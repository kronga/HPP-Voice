# HPP-Voice: A Large-Scale Evaluation of Speech Embeddings for Multi-Phenotypic Classification

This repository contains the computational pipeline accompanying the manuscript:

> **HPP-Voice: A Large-Scale Evaluation of Speech Embeddings for Multi-Phenotypic Classification**
> David Krongauz, Hido Pinto, Sarah Kohn, Yanir Marmor, Eran Segal
> arXiv:2505.16490 — https://doi.org/10.48550/arXiv.2505.16490

---

## Overview

The pipeline converts voice recordings into fixed-length embeddings using 14 pre-trained and custom-trained speech models, then trains gender-stratified LightGBM classifiers to predict health phenotypes. All code corresponds directly to the Methods section of the paper.

---

## Repository Structure

```
.
├── 1_preprocessing/            # Audio normalization, silence trimming, segmentation, and QC
│   ├── preprocess_voices.py        # Peak-normalize and trim silence (librosa)
│   ├── segment_audio.py            # Segment recordings into 5-second chunks
│   └── quality_control/
│       ├── extract_features_for_classifier.py  # Acoustic features for the QC classifier
│       └── train_classifier.py                 # Random Forest audio-fault classifier (AUC 0.95)
│
├── 2_embeddings/               # Embedding extraction for all 14 models + MFCC
│   ├── audio_embedding_pipeline.py  # Unified batch-friendly extractor (all model families)
│   ├── embeddings.py                # Individual embedder classes and mean-pooling logic
│   └── mfcc_extraction.py           # MFCC baseline feature extraction (13 coefficients)
│
├── 3_custom_model_training/    # Training of the three in-house models
│   ├── efficientnet_si/
│   │   ├── models.py                # EfficientNet encoder architecture
│   │   └── ssl_pretraining.py       # Contrastive self-supervised training on HPP-Voice
│   └── hebrew_xlsr/
│       ├── finetune_wav2vec_medical.py                              # XLSR Hebrew-PT continued pretraining + pyannote-FT
│       └── Fine_Tune_XLSR_Wav2Vec2_on_Hebrew_ASR_non_multi_lingual.ipynb  # XLSR Hebrew-FT (ASR fine-tuning)
│
├── 4_classification/           # LightGBM classifier, 4-fold CV, Optuna HPO, 20 seeds
│   ├── predict_downstream_tasks.py      # Main classification pipeline (gender-stratified)
│   ├── config_predict_downstream_tasks.yaml  # Hyperparameter search configuration
│   └── utils/
│       ├── stratified_split.py          # Stratified train/test splitting utilities
│       └── utils.py                     # Shared helpers (device setup, seeding, I/O)
│
├── 5_statistical_analysis/     # Wilcoxon signed-rank test + Benjamini-Hochberg FDR correction
│   └── sensitivity_analysis.ipynb
│
└── 6_visualization/            # Radar plots and per-condition performance figures
    └── plot_results.py
```

---

## Pipeline Summary

```
Raw Audio (HPP-Voice cohort)
        │
        ▼
1. Preprocessing & QC
   • Peak-normalize (max RMS = 1) and trim leading/trailing silence
   • Random Forest QC classifier trained on 488 manually labeled recordings
     (5-fold CV, mean AUC = 0.95); exclude recordings with fault probability > 50%
   • Segment each recording into 5-second chunks
        │
        ▼
2. Embedding Extraction
   • 14 speech models across 5 families (see Table 1 in paper):
     - Speech foundation:  wav2vec2-Base/Large, WavLM-Base/Large, XLSR-53
     - Hebrew-specific:    XLSR Hebrew-PT, XLSR Hebrew-FT
     - Speaker diarization: WavLM-SD, pyannote
     - Speaker identification: x-vector, EfficientNet (custom), pyannote-FT (custom)
     - Emotion: wav2vec2-SER, WavLM-SED
   • MFCC (13 coefficients) as classical baseline
   • Frame-level outputs → mean pooling → single 1×d vector per recording
   • Only the first 5-second segment is used for classifier training/evaluation
        │
        ▼
3. Custom Model Training (three in-house models)
   • EfficientNet SI: contrastive self-supervised training on HPP-Voice (no speaker labels)
   • pyannote-FT: pyannote encoder fine-tuned for SI on HPP-Voice via same contrastive framework
   • XLSR Hebrew-PT/FT: continued pretraining on ivrit.ai corpus + ASR fine-tuning on Common Voice/he
        │
        ▼
4. Classification
   • Gender-stratified (male / female subsets: 1,993 / 2,150 unique participants)
   • LightGBM classifier with Optuna HPO (20 trials per fold)
   • 4-fold cross-validation × 20 random seeds
   • Age included as additional input feature to control for confounding
   • Baseline: age-only model
        │
        ▼
5. Statistical Analysis
   • Pairwise Wilcoxon signed-rank tests (embedding model vs. age-only baseline)
   • Multiple testing correction: Benjamini-Hochberg (q < 0.05)
        │
        ▼
6. Visualization
   • Radar plots of best AUC per model family across all health phenotypes
   • Per-condition performance figures
```

---

## Requirements

Core dependencies:

```
torch>=2.2
torchaudio
transformers
librosa
lightgbm
optuna
scikit-learn
numpy
pandas
scipy
matplotlib
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Data Availability

The HPP-Voice dataset used in this study is not publicly available due to privacy regulations. To request access, please contact [corresponding author email].

---

## Citation

If you use this code, please cite:

```bibtex
@article{krongauz2025hppvoice,
  title   = {HPP-Voice: A Large-Scale Evaluation of Speech Embeddings for Multi-Phenotypic Classification},
  author  = {Krongauz, David and Pinto, Hido and Kohn, Sarah and Marmor, Yanir and Segal, Eran},
  journal = {arXiv preprint arXiv:2505.16490},
  year    = {2025},
  doi     = {10.48550/arXiv.2505.16490}
}
```
