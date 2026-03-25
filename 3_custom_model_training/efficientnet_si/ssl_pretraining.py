import warnings
import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(parent_dir)

import librosa
import wandb
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchsummary import summary
from tqdm import tqdm
from transformers import ViTMAEForPreTraining, AutoModel
import gc

import models
from src.utils.utils import *

# set dataset globals
IS_SEGMENTED = True
IS_REMOVE_BEGINNING = False
# set hyperparameters for model
IS_CLASSIFICATION_TASK = True
ARCHITECTURE = "EfficientNet"
SCHEDULER = "ReduceLROnPlateau_MaxTrainRecall@1"  # "ReduceLROnPlateau_MaxTrainRecall@1" or "ReduceLROnPlateau_MaxTestRecall@1" or "ReduceLROnPlateau_MinTestLoss"
LR = 1e-4
WD = 0.0
EPOCHS = 10
BATCH_SIZE = 10
QUANTILES = None
PATH_FOR_DATA = "/net/mraid20/export/genie/LabData/Analyses/sarahk/voice/"
setup_seed(SEED)


class SegmentsDataset(Dataset):
    def __init__(self, directory, architecture, split="train"):
        """
        Args:
            directory (string): Directory with all the voice data.
            architecture (string): 'EfficientNet' or other to define which type of data to retrieve.
            split (string): 'train', 'val' or 'test' to define which part of the data to retrieve.
        """
        self.architecture = architecture
        if "EfficientNet" in self.architecture:
            self.directory = os.path.join(directory, f"all_segments_5secs_{split}")
        self.file_names = [f for f in os.listdir(self.directory) if f.endswith(f".npy")]
        self.ids = self._extract_ids(self.file_names)
        self.recordings = self._extract_recordings(self.file_names)

    def _extract_ids(self, file_names):
        """
        Extracts and returns unique IDs from file names.
        """
        ids = []
        for file_name in file_names:
            person_id = file_name.split("_")[
                0
            ]  # Assumes ID is the first part of the filename before the first '_'
            ids.append(int(person_id))
        unique_ids = list(set(ids))
        return unique_ids

    def _extract_recordings(self, file_names):
        """
        Extracts and returns unique recordings (in format ID_date) from file names.
        """
        ids_and_dates = []
        for file_name in file_names:
            person_id = file_name.split("_")[0]
            visit_date = file_name.split("_")[1]
            ids_and_dates.append(f"{person_id}_{visit_date}")
        unique_recordings = list(set(ids_and_dates))
        return unique_recordings

    def __len__(self):
        return len(self.recordings)

    def __getitem__(self, idx):
        # get all segments of a person
        # person_id = self.ids[idx]
        # person_files = [f for f in self.file_names if f.startswith(f'{person_id}_')]

        # get all segments of a recording
        num_segments = 4
        recording_name = self.recordings[idx]
        segments_files = [
            f for f in self.file_names if f.startswith(f"{recording_name}_")
        ]
        if len(segments_files) < num_segments:
            return None, None
        data_segments = [
            np.load(os.path.join(self.directory, f))
            for f in segments_files[:num_segments]
        ]
        data_tensors = [
            torch.from_numpy(data.squeeze()).float() for data in data_segments
        ]

        # data_tensors: Any = []
        # for seg in range(5):
        #     data_tensors.append(segments_files[seg])
        return data_tensors, recording_name


class SSLModel:
    def __init__(self, config, train_ds, test_ds):
        """Define train / test dataset, model architecture, loss function and optimizer
        according to the desired task"""
        self.config = config
        self.train_dl = DataLoader(
            train_ds, batch_size=self.config["batch_size"], shuffle=True
        )
        self.test_dl = DataLoader(
            test_ds, batch_size=self.config["batch_size"], shuffle=False
        )
        self.model = self._define_model()
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(DEVICE)
        self.optimizer, self.scheduler = self._define_optimizer()
        self.test_losses = []
        self.test_performances = []
        self.train_performances = []
        self.num_segments = 4
        # Load checkpoint if available
        self.start_epoch = self.load_checkpoint(
            file_path=f"checkpoint_{wandb.run.id}.pth"
        )

    def _define_model(self):
        # Load the desired model:
        if "MAE" in self.config["architecture"]:
            model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

        if "EfficientNet" in self.config["architecture"]:
            self.n_ftrs = 512
            model = models.EffNet(in_channel=1, stride=2, dilation=1)
            model.fc = torch.nn.Linear(model.fc.in_features, self.n_ftrs)

        if "Wav2vec" in self.config["architecture"]:
            model = AutoModel.from_pretrained("facebook/wav2vec2-base-960h")
            self.n_ftrs = 768

        total_params = sum(p.numel() for p in model.parameters())
        wandb.log({f"parameters": total_params})
        print(f"Total number of parameters: {total_params}")

        return model

    def _define_optimizer(
        self,
    ):  # consider changing to SGD with StepLR scheduler as in SleepFM
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        if "ReduceLROnPlateau_Max" in self.config["scheduler"]:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max")
        elif "ReduceLROnPlateau_Min" in self.config["scheduler"]:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        return optimizer, scheduler

    def save_checkpoint(self, epoch, file_path="checkpoint.pth"):
        """Save the model, optimizer and scheduler state, and current epoch"""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            # 'test_losses': self.test_losses.state_dict(),
            # 'test_performances': self.test_performances.state_dict(),
            # 'train_performances': self.train_performances.state_dict(),
        }
        torch.save(state, file_path)

    def load_checkpoint(self, file_path="checkpoint.pth"):
        """Load the checkpoint if available"""
        if os.path.isfile(file_path):
            state = torch.load(file_path)
            self.model.load_state_dict(state["model_state_dict"])
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
            # self.test_losses.load_state_dict(state['test_losses'])
            # self.test_performances.load_state_dict(state['test_performances'])
            # self.train_performances.load_state_dict(state['train_performances'])
            epoch = state["epoch"]
            print(f"Resuming from epoch {epoch + 1}")
            return epoch + 1
        else:
            print("No checkpoint found, starting from scratch")
            return 0

    def compute_loss_on_batch(self, segments):
        """Compute the loss on a batch of segments"""

        # retrieve batch size
        n = segments[0].shape[0]

        if "EfficientNet" in self.config["architecture"]:
            # create list of embeddings for each recording
            segments_list = [
                segments[0][:, idx, :].to(DEVICE, dtype=torch.float)
                for idx in range(self.num_segments)
            ]

            embeddings = [
                self.model(segments_list[idx].unsqueeze(1))
                for idx in range(self.num_segments)
            ]

            embeddings = [
                torch.nn.functional.normalize(embeddings[idx])
                for idx in range(self.num_segments)
            ]

            # compute leave-one-out loss
            loss = 0.0
            pairwise_loss = np.zeros((self.num_segments, 2), dtype=float)
            correct = np.zeros((self.num_segments, 2), dtype=int)
            pairs = np.zeros((self.num_segments, 2), dtype=int)

            for idx in range(self.num_segments):
                # take the average of the embeddings of all other segments
                other_emb = torch.stack(
                    [
                        embeddings[j]
                        for j in list(range(idx))
                        + list(range(idx + 1, self.num_segments))
                    ]
                ).sum(0) / (self.num_segments - 1)

                logits = torch.matmul(
                    embeddings[idx], other_emb.transpose(0, 1)
                )  # * torch.exp(self.temperature))
                labels = torch.arange(logits.shape[0], device=DEVICE)

                cur_loss = torch.nn.functional.cross_entropy(
                    logits, labels, reduction="sum"
                )
                loss += cur_loss
                pairwise_loss[idx, 0] = cur_loss.item()
                if len(logits) != 0:
                    correct[idx, 0] = (
                        (torch.argmax(logits, axis=0) == labels).sum().item()
                    )
                else:
                    correct[idx, 0] = 0
                pairs[idx, 0] = n  # batch_size

                cur_loss = torch.nn.functional.cross_entropy(
                    logits.transpose(0, 1), labels, reduction="sum"
                )
                loss += cur_loss
                pairwise_loss[idx, 1] = cur_loss.item()
                if len(logits) != 0:
                    correct[idx, 1] = (
                        (torch.argmax(logits, axis=1) == labels).sum().item()
                    )
                else:
                    correct[idx, 1] = 0
                pairs[idx, 1] = n  # batch_size

        loss /= self.num_segments * 2 * n

        # Log recall@1 per batch
        recall_1_per_batch = [
            100 * correct[i, j] / pairs[i, j]
            for i in range(self.num_segments)
            for j in [0, 1]
        ]
        # print(((self.num_segments * 2) * "{:.3f}\t").format(*recall_1_per_batch))
        wandb.log({f"Train recall@1 per batch": np.mean(recall_1_per_batch)})

        return n, loss, recall_1_per_batch, correct, pairs, embeddings

    def train_one_epoch(self):
        """Loop over batches and update model's weights"""
        self.model.train()
        total_n = 0
        total_correct = np.zeros((self.num_segments, 2), dtype=int)
        total_pairs = np.zeros((self.num_segments, 2), dtype=int)
        losses = []
        accuracies = []
        full_trainset_embs = [
            torch.empty(0, self.n_ftrs) for _ in range(self.num_segments)
        ]
        for i, segments in enumerate(self.train_dl):
            # print("Step", i + 1)
            self.optimizer.zero_grad()

            # for i in range(torch.cuda.device_count()):
            #     self.print_memory_usage(i)

            n, loss, recall_1_per_batch, correct, pairs, embeddings = (
                self.compute_loss_on_batch(segments)
            )

            total_correct += correct
            total_n += n  # batch_size
            total_pairs += pairs
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())  # list of all losses within one epoch
            accuracies.append(
                np.mean(recall_1_per_batch)
            )  # list of all accuracies within one epoch

            if (i + 1) % 10 == 0:
                print(f"Step [{i + 1}/{len(self.train_dl)}], Loss: {loss.item():.4f}")
                print(
                    "Acc (recall@1 per batch): {}; ".format(
                        " ".join(
                            map(
                                "{:.1f}".format,
                                [
                                    100
                                    * (correct[j, 0] + correct[j, 1])
                                    / (pairs[j, 0] + pairs[j, 1])
                                    for j in range(self.num_segments)
                                ],
                            )
                        )
                    )
                )
                # wandb.log({f"Train Loss": loss.item()})

            # concatenate embedings
            full_trainset_embs = [
                torch.cat((t1, t2.cpu()), dim=0)
                for t1, t2 in zip(full_trainset_embs, embeddings)
            ]

            # Free up memory by deleting tensors
            del embeddings, loss, correct, pairs, recall_1_per_batch

            # Run garbage collector
            gc.collect()

            # Empty CUDA cache
            torch.cuda.empty_cache()

        wandb.log({f"Train Contrastive Loss by epoch": np.mean(losses)})

        # metric per batch, at epoch-level
        wandb.log({f"Train recall@1 per batch, by epoch": np.mean(accuracies)})

        # metric for full train set
        recall_x = {}
        n = len(full_trainset_embs[0])
        for x in [1, 10, 50]:
            recall_x[x] = self.eval_recall_at_x(full_trainset_embs, x=x)
            wandb.log({f"Train recall@{x} (N={n})": recall_x[x]})
            wandb.log({f"Random recall@{x} (N={n})": 100 * x / n})
        if self.config["scheduler"] == "ReduceLROnPlateau_MaxTrainRecall@1":
            self.scheduler.step(recall_x[1])

        # metric for N (length of test set)
        recall_x = {}
        n = len(self.test_dl.dataset)
        embs = [torch.empty(0, self.n_ftrs) for _ in range(self.num_segments)]
        # sample n random embeddings from the full train set
        random_indices = random.sample(range(0, len(full_trainset_embs[0])), n)
        for i in range(self.num_segments):
            embs[i] = torch.cat((embs[i], full_trainset_embs[i][random_indices]), dim=0)
        for x in [1, 10, 50]:
            recall_x[x] = self.eval_recall_at_x(embs, x=x)
            wandb.log({f"Train recall@{x} (N={n})": recall_x[x]})
            wandb.log({f"Random recall@{x} (N={n})": 100 * x / n})

    def eval_one_epoch(self):
        """Loop over batches and evaluate model"""
        # Evaluation on test set
        self.model.eval()
        total_n = 0
        total_correct = np.zeros((self.num_segments, 2), dtype=int)
        total_pairs = np.zeros((self.num_segments, 2), dtype=int)
        losses = []
        accuracies = []
        full_testset_embs = [
            torch.empty(0, self.n_ftrs) for _ in range(self.num_segments)
        ]
        for i, segments in enumerate(self.test_dl):
            with torch.no_grad():
                n, loss, recall_1_per_batch, correct, pairs, embeddings = (
                    self.compute_loss_on_batch(segments)
                )

                total_correct += correct
                total_n += segments[0].shape[0]  # == batch_size
                total_pairs += pairs

                losses.append(loss.item())  # list of all losses within one  epoch
                accuracies.append(
                    np.mean(recall_1_per_batch)
                )  # list of all accuracies within one  epoch

                if (i + 1) % 10 == 0:
                    print(
                        f"Step [{i + 1}/{len(self.train_dl)}], Loss: {loss.item():.4f}"
                    )
                    print(
                        "Acc (recall@1 per batch): {}; ".format(
                            " ".join(
                                map(
                                    "{:.1f}".format,
                                    [
                                        100
                                        * (correct[j, 0] + correct[j, 1])
                                        / (pairs[j, 0] + pairs[j, 1])
                                        for j in range(self.num_segments)
                                    ],
                                )
                            )
                        )
                    )

                # concatenate embedings
                full_testset_embs = [
                    torch.cat((t1, t2.cpu()), dim=0)
                    for t1, t2 in zip(full_testset_embs, embeddings)
                ]

        # Log epoch-level loss
        wandb.log({f"Test Contrastive Loss by epoch": np.mean(losses)})

        # metric per batch, at epoch-level
        wandb.log({f"Test recall@1 per batch, by epoch": np.mean(accuracies)})

        # metric among full test set
        recall_x = {}
        n = len(full_testset_embs[0])
        for x in [1, 10, 50]:
            recall_x[x] = self.eval_recall_at_x(full_testset_embs, x=x)
            wandb.log({f"Test recall@{x} (N={n})": recall_x[x]})
            wandb.log({f"Random recall@{x} (N={n})": 100 * x / n})
        if self.config["scheduler"] == "ReduceLROnPlateau_MaxTestRecall@1":
            self.scheduler.step(recall_x[1])

        return np.mean(losses), [np.mean(recall_1_per_batch)] + list(recall_x.values())

    def eval_recall_at_x(self, embeddings, x=1):
        """
        Evaluate recall@x for the given embeddings
        Args:
            embeddings (list): list of torch.tensor of embeddings
            x (int): number of top embeddings to consider
        Returns:
            recall@x (float)
        """
        with torch.no_grad():
            num_segments = len(embeddings)
            correct = np.zeros((num_segments, 2), dtype=int)
            for idx in range(num_segments):
                # take the average of the embeddings of all other segments
                other_emb = torch.stack(
                    [
                        embeddings[j]
                        for j in list(range(idx)) + list(range(idx + 1, num_segments))
                    ]
                ).sum(0) / (num_segments - 1)
                # compute similarity matrix
                logits = torch.matmul(embeddings[idx], other_emb.transpose(0, 1))
                # define labels
                labels = torch.arange(logits.shape[0])
                # check correctness of top x predictions
                top_x = torch.topk(logits, x, dim=0).indices
                correct[idx, 0] = (top_x == labels.unsqueeze(0)).sum().item()
                top_x = torch.topk(logits, x, dim=1).indices
                correct[idx, 1] = (top_x == labels.unsqueeze(1)).sum().item()
            recall_x = 100 * correct / embeddings[0].shape[0]
        return recall_x.mean()

    def train_and_eval(self):
        """Training Loop with metric calculation and WandB logging.
        Returns best test score over epochs, and save best model (early stopping)"""
        # Main loop
        for epoch in range(self.start_epoch, self.config["epochs"]):
            print(f'Epoch [{epoch + 1}/{str(self.config["epochs"])}]')
            wandb.log({"epoch": epoch + 1})
            train_performance = self.train_one_epoch()
            test_loss, test_performance = self.eval_one_epoch()
            if self.config["scheduler"] == "ReduceLROnPlateau_MinTestLoss":
                self.scheduler.step(test_loss)
            self.test_losses.append(test_loss)
            self.test_performances.append(test_performance)
            self.train_performances.append(train_performance)
            # Save checkpoint
            if self.config["epochs"] > 200:
                self.save_checkpoint(
                    epoch=epoch, file_path=f"checkpoint_{wandb.run.id}.pth"
                )
            # Save best model
            if (epoch == self.start_epoch) or (
                self.test_performances[-1] > max(self.test_performances[:-1])
            ):
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        "models",
                        f"model_{wandb.run.id}.pth",
                    ),
                )
        return (
            min(self.test_losses),
            max(self.test_performances),
            max(self.train_performances),
        )

    def print_memory_usage(self, device=0):
        # Ensure that device index is valid
        if device >= torch.cuda.device_count():
            print(f"Invalid device index: {device}")
            return

        # Get device properties
        props = torch.cuda.get_device_properties(device)
        # Total memory in bytes
        total_memory = props.total_memory
        # Currently reserved memory in bytes
        reserved_memory = torch.cuda.memory_reserved(device)
        # Currently allocated memory in bytes
        allocated_memory = torch.cuda.memory_allocated(device)
        # Currently cached memory in bytes
        cached_memory = torch.cuda.memory_reserved(device)

        print(f"CUDA Device: {torch.cuda.get_device_name(device)}_{device}")
        print(f"Total Memory: {total_memory / 1024**3:.2f} GB")
        print(f"Reserved Memory: {reserved_memory / 1024**3:.2f} GB")
        print(f"Allocated Memory: {allocated_memory / 1024**3:.2f} GB")
        print(f"Cached Memory: {cached_memory / 1024**3:.2f} GB")


def create_embeddings(model, weights_file: str, input_path: str, output_path: str):
    mkdirifnotexists(output_path)
    # load pretrained weights
    model.load_state_dict(torch.load(weights_file, map_location=torch.device(DEVICE)))
    model.eval()
    target_sr = 16000
    target_duration = 5
    for root, dirs, files in os.walk(input_path):
        for file in tqdm(files, desc="Processing files", unit="files"):
            # Check if the file is an audio file (e.g., .wav, .flac, .mp3)
            if file.lower().endswith((".wav", ".flac", ".mp3")):
                file_path = os.path.join(root, file)
                try:
                    audio, sr = librosa.load(file_path, sr=None)
                    audio_npy = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                    audio_tensor = torch.tensor(audio_npy, dtype=torch.float32).to(
                        DEVICE, dtype=torch.float
                    )
                    embedding = torch.nn.functional.normalize(
                        model(audio_tensor.unsqueeze(0).unsqueeze(1))
                    )
                    # Save the embeddings into a numpy file  in the output directory if includes full duration
                    if len(audio_npy) == target_duration * target_sr:
                        filename = os.path.splitext(file)[0] + "_emb.npy"
                        np.save(
                            os.path.join(output_path, filename),
                            embedding.detach().numpy(),
                        )
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    run_config = {
        "architecture": ARCHITECTURE,
        "learning_rate": LR,
        "scheduler": SCHEDULER,
        "weight_decay": WD,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "quantiles": QUANTILES,
        "is_classification_task": IS_CLASSIFICATION_TASK,
    }

    # Initialize WandB
    wandb_config = run_config.copy()
    wandb_config["Pytorch"] = torch.__version__
    if EPOCHS > 200:
        wandb.init(
            project=f"train_SSL",
            config=wandb_config,
            resume="allow",
            id=f"SSL_{ARCHITECTURE}_b{BATCH_SIZE}_lr{LR}_sc{SCHEDULER}",
        )
    else:
        wandb.init(project=f"train_SSL", config=wandb_config)

    # Create datasets
    print("Load train dataset")
    dataset = SegmentsDataset(PATH_FOR_DATA, run_config["architecture"], split="train")
    data_numpy = np.array([data for data, _ in tqdm(dataset) if data is not None])
    train_dataset = TensorDataset(torch.tensor(data_numpy, dtype=torch.float32))
    wandb.log({"Train N": len(dataset)})
    print("Load test dataset")
    dataset = SegmentsDataset(PATH_FOR_DATA, run_config["architecture"], split="test")
    data_numpy = np.array([data for data, _ in tqdm(dataset) if data is not None])
    test_dataset = TensorDataset(torch.tensor(data_numpy, dtype=torch.float32))
    wandb.log({"Test N": len(dataset)})

    # train model and save weights
    model_test_loss, model_test_performance, model_train_performance = SSLModel(
        run_config, train_dataset, test_dataset
    ).train_and_eval()
    print("Training done - model weights saved in project directory")

    # Log and close WandB
    wandb.log(
        {f"Best Test score (recall@1 among {BATCH_SIZE})": model_test_performance[0]}
    )
    wandb.log({f"Best Test score (recall@1 among all)": model_test_performance[1]})
    wandb.log({f"Best Test score (recall@10 among all)": model_test_performance[2]})
    wandb.log({f"Best Test score (recall@50 among all)": model_test_performance[3]})

    wandb.finish()

    # # create embeddings
    # model = models.EffNet(in_channel=1, stride=2, dilation=1)
    # model.fc = torch.nn.Linear(model.fc.in_features, 512)
    # if DEVICE == "cuda":
    #     model = torch.nn.DataParallel(model)
    # model.to(DEVICE)
    # create_embeddings(
    #     model=model,
    #     weights_file="models/SSL_EfficientNet_model.pth",
    #     input_path="/net/mraid20/export/genie/LabData/Analyses/sarahk/voice/processed_segments_5secs",
    #     output_path="/net/mraid20/export/genie/LabData/Analyses/sarahk/voice/all_embeddings_5secs",
    # )
    # print("Embeddings created and saved")
