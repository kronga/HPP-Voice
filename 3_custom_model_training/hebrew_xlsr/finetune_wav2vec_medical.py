import glob
import re
import time

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2Config
import pandas as pd
import os
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from tqdm import tqdm
import logging
import wandb

# from torchvision.ops.focal_loss import sigmoid_focal_loss
from focal_loss.focal_loss import FocalLoss
from memory_analysis import full_memory_analysis

import torch.cuda.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity
import gc

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AudioSegmentDataset(Dataset):
    def __init__(
        self, audio_dir, metadata_file, target_condition, processor, sample_rate=16000
    ):
        self.audio_dir = audio_dir
        self.processor = processor
        self.sample_rate = sample_rate

        logger.info(f"Initializing AudioSegmentDataset with {audio_dir}")

        # Read the metadata file
        self.metadata = pd.read_csv(metadata_file)
        logger.info(f"Loaded metadata from {metadata_file}")

        # Ensure the target condition exists in the metadata
        if target_condition not in self.metadata.columns:
            raise ValueError(
                f"Target condition '{target_condition}' not found in metadata."
            )

        # Get all audio files in the directory
        all_files = [f for f in os.listdir(audio_dir) if f.endswith((".flac", ".wav"))]
        logger.info(f"Found {len(all_files)} audio files")

        # Create a mapping of person ID to all their segment files
        person_segments = {}
        for file in all_files:
            person_id = file.split("_")[0]
            if person_id not in person_segments:
                person_segments[person_id] = []
            person_segments[person_id].append(file)

        # Create a list of all segments and their corresponding labels
        self.segments = []
        self.labels = []
        skipped_persons = 0
        for person_id, segments in tqdm(
            person_segments.items(), desc="Processing persons"
        ):
            person_data = self.metadata[
                self.metadata["filename"].str.startswith(person_id)
            ]
            if not person_data.empty:
                label = person_data[target_condition].iloc[0]
                if pd.notna(label):
                    self.segments.extend(segments)
                    # Convert label to integer
                    int_label = int(label)
                    self.labels.extend([int_label] * len(segments))
                else:
                    skipped_persons += 1
                    # logger.info(f"Skipping person {person_id} due to None/NaN label for {target_condition}")

        logger.info(f"Dataset initialized with {len(self.segments)} segments")
        logger.info(f"Skipped {skipped_persons} persons due to None/NaN labels")

        self.num_classes = len(np.unique(self.labels))
        logger.info(f"Number of classes: {self.num_classes}")

    def __len__(self):
        return len(self.segments)

    def find_audio_file(self, file_name):
        # Check for both .flac and .wav extensions
        for ext in [".flac", ".wav"]:
            full_path = os.path.join(self.audio_dir, file_name)
            if full_path.endswith(ext):
                return full_path
            if os.path.exists(full_path + ext):
                return full_path + ext
        raise FileNotFoundError(f"No .flac or .wav file found for {file_name}")

    def __getitem__(self, idx):
        segment_file = self.segments[idx]
        label = self.labels[idx]

        try:
            # Find and load audio file
            audio_path = self.find_audio_file(segment_file)
            waveform, _ = librosa.load(audio_path, sr=self.sample_rate)

            # Preprocess audio
            audio_input = self.processor(
                waveform,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding="longest",
            )
            return audio_input.input_values.squeeze(), torch.tensor(
                label, dtype=torch.long
            )
        except Exception as e:
            logger.error(f"Error loading file {segment_file}: {str(e)}")
            # Return a dummy tensor and the label
            return torch.zeros(self.sample_rate * 10), torch.tensor(
                label, dtype=torch.long
            )


def collate_fn(batch):
    # Separate inputs and labels
    inputs, labels = zip(*batch)

    # Pad inputs
    inputs_padded = torch.nn.utils.rnn.pad_sequence(
        inputs, batch_first=True, padding_value=0
    )

    # Create attention masks
    attention_mask = torch.zeros_like(inputs_padded, dtype=torch.long)
    for i, inp in enumerate(inputs):
        attention_mask[i, : len(inp)] = 1

    # Convert labels to tensor
    labels = torch.stack(labels)

    return inputs_padded, attention_mask, labels


import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class Wav2VecMedicalClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        hidden_size=1024,
        dropout_rate=0.3,
        activation="gelu",
        norm_types=None,  # Changed to list
        pooling_type="avg",
        init_type="xavier",
        layer_dims=None,
    ):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large")
        self.pooling = (
            nn.AdaptiveAvgPool1d(1)
            if pooling_type == "avg"
            else nn.AdaptiveMaxPool1d(1)
        )

        activation_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "swiglu": nn.SiLU()}
        activation_fn = activation_map[activation]

        layers = []
        current_dim = hidden_size

        # if layer_dims is None:
        #     layer_dims = [512] * (num_layers - 1)
        # else:
        num_layers = len(layer_dims) + 1
        for i in range(num_layers):
            output_dim = layer_dims[i] if i < len(layer_dims) else num_classes
            layers.append(nn.Linear(current_dim, output_dim))

            if i < num_layers - 1:
                # Add multiple norm layers if specified
                for norm_type in norm_types:
                    layers.append(self._get_norm_layer(norm_type, output_dim))
                layers.extend([activation_fn, nn.Dropout(dropout_rate)])
            current_dim = output_dim

        self.prediction_head = nn.Sequential(self.pooling, nn.Flatten(), *layers)

        self._init_weights(init_type)

    def _get_norm_layer(self, norm_type, dim):
        norm_map = {
            "layer": nn.LayerNorm(dim),
            "batch": nn.BatchNorm1d(dim),
            "rmsnorm": RMSNorm(dim),
        }
        return norm_map[norm_type]

    def _init_weights(self, init_type):
        init_fn = (
            nn.init.xavier_uniform_
            if init_type == "xavier"
            else nn.init.kaiming_normal_
        )

        for module in self.prediction_head.modules():
            if isinstance(module, nn.Linear):
                init_fn(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, attention_mask=None):
        wav2vec_output = self.wav2vec(
            x, attention_mask=attention_mask
        ).last_hidden_state
        wav2vec_output = wav2vec_output.transpose(1, 2)
        return self.prediction_head(wav2vec_output)

    def freeze_wav2vec(self):
        """Freeze all wav2vec parameters"""
        for param in self.wav2vec.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.parameters():
            param.requires_grad = True

    def freeze_feature_encoder(self):
        """Freeze the wav2vec feature encoder layers"""
        self.wav2vec.freeze_feature_encoder()


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class MemoryProfiler:
    @staticmethod
    def print_memory_stats(prefix=""):
        """Print current GPU memory statistics"""
        if torch.cuda.is_available():
            print(f"\n{prefix} Memory Stats:")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            print(
                f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
            )
            print(
                f"Max Reserved: {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB\n"
            )

    @staticmethod
    def clear_memory():
        """Clear GPU cache and run garbage collection"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


class ThreePhaseTrainer:
    def __init__(
        self,
        model,
        criterion,
        train_loader,
        val_loader,
        device,
        phase_1_learning_rate=1e-4,
        phase_2_learning_rate=1e-4,
        weight_decay=0.01,
        data_dir=".",
        seed=42,
        config=None,
    ):
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.train_step = 0
        self.val_step = 0
        self.global_epoch = 0
        self.learning_rate_phase_1 = phase_1_learning_rate
        self.learning_rate_phase_2 = phase_2_learning_rate
        self.data_dir = data_dir
        self.memory_profiler = MemoryProfiler()
        self.seed = seed
        self.config = config

        # Initialize optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate_phase_1,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2, verbose=True
        )

    def setup_wandb_logging(self, phase_name):
        """Setup wandb metric definitions"""
        # Define training metrics to use train_step
        wandb.define_metric("train_step")
        wandb.define_metric(f"{phase_name}_train_step_loss", step_metric="train_step")
        wandb.define_metric(f"{phase_name}_train_step_auc", step_metric="train_step")
        wandb.define_metric(f"{phase_name}_learning_rate", step_metric="train_step")

        # Define epoch-level metrics
        wandb.define_metric("epoch")
        wandb.define_metric(f"{phase_name}_train_epoch_loss", step_metric="epoch")
        wandb.define_metric(f"{phase_name}_train_epoch_auc", step_metric="epoch")
        wandb.define_metric(f"{phase_name}_val_epoch_loss", step_metric="epoch")
        wandb.define_metric(f"{phase_name}_val_epoch_auc", step_metric="epoch")

        wandb.define_metric("global_epoch")
        wandb.define_metric(f"train_epoch_auc", step_metric="global_epoch")
        wandb.define_metric(f"train_epoch_loss", step_metric="global_epoch")
        wandb.define_metric(f"val_epoch_auc", step_metric="global_epoch")
        wandb.define_metric(f"val_epoch_loss", step_metric="global_epoch")
        # wandb.define_metric(f"{phase_name}_val_epoch_f1", step_metric="epoch")

    def get_arch_desc(self):
        return f"layer_dims_{'-'.join(map(str, self.config.layer_dims))}_activation_{self.config.activation}_norm_types_{''.join(self.config.norm_types)}_pooling_{self.config.pooling_type}_init_{self.config.init_type}"

    def calculate_metrics(self, all_preds, all_labels):
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Calculate AUC-ROC
        if all_preds.shape[1] == 2:  # Binary classification
            auc = roc_auc_score(all_labels, all_preds[:, 1])
        else:  # Multi-class
            auc = roc_auc_score(
                all_labels, all_preds, multi_class="ovr", average="weighted"
            )

        # Calculate F1 score
        f1 = f1_score(all_labels, all_preds.argmax(axis=1), average="weighted")

        return auc, f1

    def _step(self, batch, training=True):
        """Single training/validation step"""
        inputs, attention_mask, labels = [x.to(self.device) for x in batch]

        if training:
            self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(inputs, attention_mask=attention_mask)
        log_probs = torch.log_softmax(outputs, dim=1)
        probs = torch.exp(log_probs)  # If you still need probabilities
        try:
            loss = self.criterion(probs, labels)
        except ValueError as e:
            print(f"got error: {e}")
            print(f"probs: {probs}")
            raise e

        # Backward pass if training
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        return {
            "loss": loss.item(),
            "probs": probs.detach().cpu().numpy(),
            "labels": labels.cpu().numpy(),
        }

    def print_profiler_summary(self, prof, description=""):
        """Print CUDA memory usage tables"""
        table_kwargs = {"row_limit": 20, "max_name_column_width": 70}

        print(f"\n{description}")
        print("\nBy self memory:")
        print(
            prof.key_averages().table(sort_by="self_cuda_memory_usage", **table_kwargs)
        )
        print("\nBy total memory:")
        print(prof.key_averages().table(sort_by="cuda_memory_usage", **table_kwargs))

    def train_epoch(self, epoch, phase):
        """Complete training epoch with profiling"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        step_preds = []
        step_labels = []

        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch} ({phase})")

        # Profile memory at epoch start
        self.memory_profiler.print_memory_stats(f"Start of epoch {epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            # Profile first batch
            if batch_idx == 0:
                inputs, attention_mask, labels = [x.to(self.device) for x in batch]
                # full_memory_analysis(self.model, inputs)
                step_output = self._step(batch, training=True)
            else:
                step_output = self._step(batch, training=True)

            # Update tracking metrics
            total_loss += step_output["loss"]
            all_preds.extend(step_output["probs"])
            all_labels.extend(step_output["labels"])
            step_preds.extend(step_output["probs"])
            step_labels.extend(step_output["labels"])

            # Wandb logging
            wandb.log(
                {
                    f"{phase}_train_step_loss": step_output["loss"],
                    "train_step": self.train_step,
                    f"{phase}_learning_rate": self.optimizer.param_groups[0]["lr"],
                }
            )

            # Calculate and log periodic metrics
            if (batch_idx + 1) % 10 == 0:
                step_auc, step_f1 = self.calculate_metrics(step_preds, step_labels)
                wandb.log(
                    {
                        f"{phase}_train_step_auc": step_auc,
                        f"{phase}_train_step_f1": step_f1,
                        "train_step": self.train_step,
                    }
                )
                step_preds = []
                step_labels = []

            # # Periodic memory profiling
            # if batch_idx % 100 == 0:
            #     self.memory_profiler.print_memory_stats(f"Batch {batch_idx}")

            progress_bar.set_postfix(
                {
                    "loss": f"{step_output['loss']:.4f}",
                    "gpu_mem": f"{torch.cuda.memory_allocated() / 1024 ** 2:.0f}MB",
                }
            )
            self.train_step += 1

        # Calculate final metrics
        epoch_auc, epoch_f1 = self.calculate_metrics(all_preds, all_labels)
        epoch_loss = total_loss / len(self.train_loader)

        # Final memory profile for epoch
        self.memory_profiler.print_memory_stats(f"End of epoch {epoch}")

        return epoch_loss, epoch_auc, epoch_f1

    def validate(self, phase):
        self.model.eval()
        self.memory_profiler.print_memory_stats("Start of validation")

        total_loss = 0
        all_preds = []
        all_labels = []
        step_preds = []
        step_labels = []

        progress_bar = tqdm(self.val_loader, desc=f"Validating ({phase})")
        with torch.no_grad():
            for batch_idx, (inputs, attention_mask, labels) in enumerate(progress_bar):
                inputs = inputs.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs, attention_mask=attention_mask)
                log_probs = torch.log_softmax(outputs, dim=1)
                probs = torch.exp(log_probs)  # If you still need probabilities
                try:
                    loss = self.criterion(probs, labels)
                except ValueError as e:
                    print(f"got error: {e}")
                    print(f"probs: {probs}")
                    raise e
                total_loss += loss.item()

                # Calculate probabilities and store predictions

                batch_preds = probs.detach().cpu().numpy()
                batch_labels = labels.cpu().numpy()

                # Store for both step-wise and epoch metrics
                all_preds.extend(batch_preds)
                all_labels.extend(batch_labels)
                step_preds.extend(batch_preds)
                step_labels.extend(batch_labels)

                # Store for step-wise metrics
                step_preds.extend(probs.cpu().numpy())
                step_labels.extend(labels.cpu().numpy())

                progress_bar.set_postfix({"loss": loss.item()})
                self.val_step += 1

        # Calculate final metrics
        auc, f1 = self.calculate_metrics(all_preds, all_labels)
        self.memory_profiler.print_memory_stats("End of validation")
        return total_loss / len(self.val_loader), auc, f1

    def train_phase(self, num_epochs, phase_name, early_stopping_patience=3):
        """Complete training phase with profiling"""
        # Reset step counters
        self.train_step = 0
        self.val_step = 0

        # wandb.disabled = True
        # os.environ['WANDB_DISABLED'] = 'true'

        # Initialize wandb
        wandb.init(
            project="three_phase_training",
            name=f"{phase_name}",
            group="training_phases",
            reinit=True,
            config={
                "phase": phase_name,
                "num_epochs": num_epochs,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "seed": self.seed,
                # "weight_decay": 0.01,
                # "early_stopping_patience": early_stopping_patience,
                # "loss_type": "focal_loss",
                # "focal_loss_weights": self.criterion.weights,
                # "focal_loss_gamma": self.criterion.gamma,
                # "initial_gpu_memory": f"{torch.cuda.memory_allocated() / 1024 ** 2:.2f}MB",
            },
        )
        self.setup_wandb_logging(phase_name)

        best_val_auc = 0
        patience_counter = 0

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            # Training
            train_loss, train_auc, train_f1 = self.train_epoch(epoch, phase_name)

            # Validation
            self.model.eval()
            val_loss, val_auc, val_f1 = self.validate(phase_name)

            epoch_time = time.time() - epoch_start_time

            # Logging
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Time: {epoch_time:.2f}s")
            logger.info(
                f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Train F1: {train_f1:.4f}"
            )
            logger.info(
                f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}"
            )

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "global_epoch": self.global_epoch + 1,
                    f"{phase_name}_train_epoch_loss": train_loss,
                    f"{phase_name}_train_epoch_auc": train_auc,
                    # f"{phase_name}_train_epoch_f1": train_f1,
                    f"{phase_name}_val_epoch_loss": val_loss,
                    f"{phase_name}_val_epoch_auc": val_auc,
                    f"train_epoch_auc": train_auc,
                    f"train_epoch_loss": train_loss,
                    f"val_epoch_auc": val_auc,
                    f"val_epoch_loss": val_loss,
                    # f"{phase_name}_val_epoch_f1": val_f1,
                    # f"{phase_name}_epoch_time": epoch_time,
                    f"{phase_name}_gpu_memory": torch.cuda.memory_allocated() / 1024**2,
                }
            )

            round_val_auc = round(val_auc, 2)

            # Learning rate scheduling
            self.scheduler.step(round_val_auc)

            # Model saving and early stopping
            if round_val_auc > best_val_auc:
                best_val_auc = round_val_auc
                patience_counter = 0

                # Save checkpoint
                checkpoint_dir = os.path.join(self.data_dir, "Finetuned_checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)

                checkpoint_pattern = (
                    f"best_model__phase_{phase_name}__auc_*__model_*.pt"
                )
                prev_checkpoints = glob.glob(
                    os.path.join(checkpoint_dir, checkpoint_pattern)
                )
                assert (
                    len(prev_checkpoints) <= 1
                ), f"Expected at most one checkpoint file, found {len(prev_checkpoints)}"

                if len(prev_checkpoints) == 1:
                    auc_pattern = re.compile(
                        f"best_model__phase_{phase_name}__auc_([0-9.]+)__model_.*\.pt$"
                    )
                    match = re.search(auc_pattern, prev_checkpoints[0])
                    prev_auc = match.group(1)
                    prev_auc = float(prev_auc)
                else:
                    # no prev checkpoint
                    prev_auc = 0
                best_val_auc = max(best_val_auc, prev_auc)

                if round_val_auc > prev_auc:
                    # remove previous checkpoint
                    if len(prev_checkpoints) == 1:
                        os.remove(prev_checkpoints[0])

                    checkpoint_filename = f"best_model__phase_{phase_name}__auc_{round_val_auc}__model_{self.get_arch_desc()}.pt"
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

                    torch.save(
                        {
                            "epoch": self.global_epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "scheduler_state_dict": self.scheduler.state_dict(),
                            "val_auc": round_val_auc,
                            "train_step": self.train_step,
                            "val_step": self.val_step,
                        },
                        checkpoint_path,
                    )

                    logger.info(f"Saved new best model with Val AUC: {round_val_auc}")
                    # move global epoch counter
                    self.global_epoch += 1
            else:
                patience_counter += 1
                # move global epoch counter
                self.global_epoch += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered in {phase_name} after {epoch + 1} epochs"
                    )
                    break

        # Final memory cleanup
        self.memory_profiler.clear_memory()
        self.memory_profiler.print_memory_stats(f"End of {phase_name}")

        return best_val_auc

    def reset_optimizer(self):
        """Reset optimizer with a new learning rate"""
        # reset optimizer to phase 2 learning rate
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.learning_rate_phase_2, weight_decay=0.01
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3, verbose=True
        )

    def load_checkpoint(self, checkpoint_path):
        """
        Load a saved checkpoint including model, optimizer, scheduler states and training progress.

        Args:
            checkpoint_path (str): Path to the checkpoint file

        Returns:
            dict: Loaded checkpoint data including epoch, val_auc, and steps
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        try:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)

            # Load model state
            self.model.load_state_dict(checkpoint["model_state_dict"])

            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Load scheduler state
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            # Restore training progress
            self.train_step = checkpoint["train_step"]
            self.val_step = checkpoint["val_step"]

            checkpoint_info = {
                "epoch": checkpoint["epoch"],
                "val_auc": checkpoint["val_auc"],
                "train_step": checkpoint["train_step"],
                "val_step": checkpoint["val_step"],
            }

            logger.info(
                f"Successfully loaded checkpoint from epoch {checkpoint['epoch']} "
                f"with validation AUC: {checkpoint['val_auc']:.4f}"
            )

            return checkpoint_info

        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise Exception(f"Failed to load checkpoint: {str(e)}")

    def train_with_phases(
        self,
        num_epochs_phase1=3,
        num_epochs_phase2=0,
        num_epochs_phase3=0,
        early_stopping_patience=3,
        start_phase=1,
        checkpoint_path=None,
    ):
        """
        Train the model in phases with the ability to resume from checkpoints.

        Args:
            num_epochs_phase1 (int): Number of epochs for phase 1
            num_epochs_phase2 (int): Number of epochs for phase 2
            num_epochs_phase3 (int): Number of epochs for phase 3
            early_stopping_patience (int): Number of epochs to wait before early stopping
            start_phase (int): Phase to start training from (1, 2, or 3)
            checkpoint_path (str): Path to checkpoint to load from (if starting from phase 2 or 3)

        Returns:
            tuple: Best AUC scores for each phase (None for skipped phases)
        """
        phase1_best_auc = None
        phase2_best_auc = None
        phase3_best_auc = None

        # Phase 1: Train with frozen encoder and feature extractor
        if start_phase <= 1 and num_epochs_phase1 > 0:
            phase_name = "phase1"
            logger.info(
                "Starting Phase 1: Training with frozen encoder and feature extractor"
            )
            self.model.freeze_wav2vec()
            phase1_best_auc = self.train_phase(
                num_epochs_phase1, phase_name, early_stopping_patience
            )

        # Phase 2: Train with only feature extractor frozen
        if start_phase <= 2 and num_epochs_phase2 > 0:
            phase_name = "phase2"
            if start_phase == 2 and checkpoint_path:
                logger.info(f"Loading checkpoint from {checkpoint_path} for Phase 2")
                self.load_checkpoint(checkpoint_path)

            # Reset optimizer with reduced learning rate for phase 2
            self.reset_optimizer()
            logger.info("\nStarting Phase 2: Training with frozen feature extractor")
            self.model.unfreeze_all()
            self.model.freeze_feature_encoder()
            phase2_best_auc = self.train_phase(
                num_epochs_phase2, phase_name, early_stopping_patience
            )

        # Phase 3: Full fine-tuning
        if start_phase <= 3 and num_epochs_phase3 > 0:
            phase_name = "phase3"
            if start_phase == 3 and checkpoint_path:
                logger.info(f"Loading checkpoint from {checkpoint_path} for Phase 3")
                self.load_checkpoint(checkpoint_path)

            # Reset optimizer with further reduced learning rate for phase 3
            self.reset_optimizer()
            logger.info("\nStarting Phase 3: Full fine-tuning")
            self.model.unfreeze_all()
            phase3_best_auc = self.train_phase(
                num_epochs_phase3, phase_name, early_stopping_patience
            )

        return phase1_best_auc, phase2_best_auc, phase3_best_auc


def main():
    # Set up your data and hyperparameters
    deep_dir = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder"
    audio_dir = os.path.join(deep_dir, "raw_aggregated_audio_5s_segments")
    metadata_file = os.path.join(
        deep_dir, "subject_details_table_w_medical_conditions.csv"
    )
    target_condition = "has_SA"  # Change this to the condition you want to classify
    batch_size = 133  # max stable batch size for 24GB GPU is *130*
    learning_rate = 1e-3

    # Initialize memory profiling
    memory_profiler = MemoryProfiler()
    memory_profiler.print_memory_stats("Initial state")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # F U wandb
    # wandb.disabled = True
    # os.environ['WANDB_DISABLED'] = 'true'

    # Initialize wandb with additional config
    wandb.init(
        project="wav2vec-medical-classification",
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs_phase1": 10,
            "num_epochs_phase2": 5,
            "num_epochs_phase3": 0,
            "target_condition": target_condition,
            "model": "Wav2VecMedicalClassifier",
            "weight_decay": 0.01,
        },
    )
    logger.info(f"Using device: {device}")

    # Initialize Wav2Vec2 processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    logger.info("Initialized Wav2Vec2 processor")

    # Create dataset
    dataset = AudioSegmentDataset(audio_dir, metadata_file, target_condition, processor)

    # Prepare data for stratified split
    groups = [segment.split("_")[0] for segment in dataset.segments]  # Person IDs
    labels = dataset.labels

    # Use StratifiedGroupKFold for splitting
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_indices, val_indices = next(splitter.split(dataset.segments, labels, groups))

    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    logger.info(
        f"Dataset split: {len(train_indices)} training, {len(val_indices)} validation"
    )

    # Log class distribution
    train_labels = np.array(labels)[train_indices]
    val_labels = np.array(labels)[val_indices]
    logger.info(f"Training set class distribution: {np.bincount(train_labels)}")
    logger.info(f"Validation set class distribution: {np.bincount(val_labels)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Initialize model, loss function, and optimizer
    num_classes = len(np.unique(labels))

    # Initialize model and move to device
    model = Wav2VecMedicalClassifier(num_classes=num_classes, norm_types=["batch"]).to(
        device
    )
    # torch.set_float32_matmul_precision('medium')
    # model = model.compile()
    logger.info("Model initialized")
    memory_profiler.print_memory_stats("After model initialization")

    # Log model architecture to wandb
    wandb.watch(model)

    weights = torch.FloatTensor([1, 3]).to(device)
    criterion = FocalLoss(gamma=0.7, weights=weights)

    # Create trainer with the three-phase training approach
    trainer = ThreePhaseTrainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        weight_decay=0.01,
        data_dir=deep_dir,
    )

    # Start three-phase training
    logger.info("Starting three-phase training")
    phase1_auc, phase2_auc, phase3_auc = trainer.train_with_phases(
        num_epochs_phase1=10,
        num_epochs_phase2=5,
        num_epochs_phase3=5,
        early_stopping_patience=3,
    )

    logger.info("Training completed.")
    logger.info(
        f"""Training completed.
                Best Phase 1 AUC: {phase1_auc:.4f}
                Best Phase 2 AUC: {phase2_auc:.4f}
                Best Phase 3 AUC: {phase3_auc:.4f}"""
    )

    # Close wandb run
    wandb.finish()

    # Clean up at the end
    memory_profiler.clear_memory()
    memory_profiler.print_memory_stats("End of training")


if __name__ == "__main__":
    main()
