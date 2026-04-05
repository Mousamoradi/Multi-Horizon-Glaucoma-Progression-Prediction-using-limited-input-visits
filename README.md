# Multi-Horizon-Glaucoma-Progression-Prediction-using-limited-input-visits

This repository contains code for sparse-observation, multimodal deep learning for longitudinal prediction of glaucoma progression from only two clinical visits. The pipeline integrates circumpapillary retinal nerve fiber layer (cpRNFL) OCT data, visual field (VF) data, and clinical covariates to generate prediction sequences and train multi-horizon models for forecasting progression at 2, 3, and 4 years. The repository currently includes three main scripts for data preparation, sequence generation, and model training. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}

## Overview

The workflow is organized into three stages:

1. **Data preparation**
   - Matches VF and cpRNFL records by patient, eye, and exam date
   - Applies quality-control filters
   - Computes mean deviation (MD) slope
   - Assigns progression categories for downstream analysis :contentReference[oaicite:3]{index=3}

2. **Sequence generation**
   - Creates sparse-observation two-visit sequences `(T0, T1)`
   - Assigns horizon-specific labels for years 2, 3, and 4
   - Masks missing horizons instead of discarding partially observed sequences :contentReference[oaicite:4]{index=4}

3. **Model training**
   - Trains a multimodal Bi-LSTM framework with shared-weight image encoders
   - Supports ConvNeXt, ViT, MobileNet, and EfficientNet backbones
   - Uses masked multi-horizon binary cross-entropy loss
   - Performs patient-level 5-fold cross-validation to prevent leakage :contentReference[oaicite:5]{index=5}

## Repository Structure

```text
.
├── data_preparation.py
├── sequence_generation.py
├── model_training.py
└── README.md
