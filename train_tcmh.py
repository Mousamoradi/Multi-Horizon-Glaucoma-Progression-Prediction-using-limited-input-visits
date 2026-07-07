"""
train_tcmh.py
==============
Training pipeline for the proposed TC-MMH (Temporally-Consistent Multi-Multi-
Horizon) framework for multi-horizon glaucoma progression prediction.

This script mirrors the baseline pipeline (`model_training.py`) exactly — same
cohort construction, same dataset, same patient-level GroupKFold splitting, and
same masked multi-horizon supervision — so that any performance gain over the
ConvNeXt / ViT / MobileNet / EfficientNet baselines is attributable to the two
methodological contributions and nothing else:

    1. ConvNeXt-V2 (`convnextv2_base.fcmae_ft_in1k`) image encoder with a
       lightweight tabular MLP and a *shared* prediction head conditioned on a
       learned horizon embedding (Y2 / Y3 / Y4).
    2. Masked TC-MMH loss:
           L = L_bce_masked + lambda_mono * L_mono
       where L_mono penalises non-monotone horizon predictions
       (P_{Y2} > P_{Y3} or P_{Y3} > P_{Y4}) and L_bce_masked is the same
       masked multi-horizon BCE used by the baselines.

Usage:
    python train_tcmh.py --sequences_path sequences.csv \
                         --image_dir /path/to/images \
                         --output_dir ./results_tcmh
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from PIL import Image


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# Constants  (identical to baseline unless marked TC-MMH)
# ─────────────────────────────────────────────

HORIZONS       = [2, 3, 4]
IMAGE_SIZE     = 224
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]
TABULAR_FEATS  = ["t0_md", "t1_md", "t0_rnfl", "t1_rnfl", "t0_age"]
CAT_FEATS      = ["t0_sex", "t0_race"]

# TC-MMH specific
TIMM_MODEL_NAME   = "convnextv2_base.fcmae_ft_in1k"
HORIZON_EMBED_DIM = 32
TAB_ENCODER_DIM   = 128
LAMBDA_MONO       = 0.5


# ─────────────────────────────────────────────
# 1. Dataset  (identical to baseline)
# ─────────────────────────────────────────────

class GlaucomaSequenceDataset(Dataset):
    """
    PyTorch Dataset for paired (T0, T1) multimodal glaucoma sequences.

    Each sample provides:
      - cpRNFL image at T0 and T1  (3 × 224 × 224)
      - VF TD image at T0 and T1   (3 × 224 × 224)
      - Tabular covariates          (float tensor)
      - Label vector [y2, y3, y4]  (float tensor, NaN for masked horizons)
    """

    def __init__(self,
                 df: pd.DataFrame,
                 image_dir: str,
                 scaler: StandardScaler = None,
                 fit_scaler: bool = False,
                 augment: bool = False):
        self.df        = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.augment   = augment

        aug_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ] if augment else []

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            *aug_transforms,
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        tab_cols = TABULAR_FEATS + [f"{c}_{v}" for c in CAT_FEATS
                                    for v in df[c].dropna().unique()]
        df_encoded = pd.get_dummies(df, columns=CAT_FEATS, drop_first=False)
        self.tab_cols = [c for c in df_encoded.columns if c in tab_cols or
                         any(c.startswith(f"{cat}_") for cat in CAT_FEATS)]
        tab_data = df_encoded[self.tab_cols].fillna(0).values.astype(np.float32)

        if fit_scaler:
            self.scaler = StandardScaler()
            self.tab_data = self.scaler.fit_transform(tab_data)
        elif scaler is not None:
            self.scaler   = scaler
            self.tab_data = scaler.transform(tab_data)
        else:
            self.scaler   = None
            self.tab_data = tab_data

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, filepath: str) -> torch.Tensor:
        path = self.image_dir / filepath
        if not path.exists():
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        rnfl_t0 = self._load_image(str(row.get("rnfl_img_t0", "")))
        rnfl_t1 = self._load_image(str(row.get("rnfl_img_t1", "")))
        vf_t0   = self._load_image(str(row.get("vf_img_t0",   "")))
        vf_t1   = self._load_image(str(row.get("vf_img_t1",   "")))

        tab = torch.tensor(self.tab_data[idx], dtype=torch.float32)

        labels = torch.tensor(
            [row.get(f"label_y{h}", float("nan")) for h in HORIZONS],
            dtype=torch.float32
        )

        return {
            "rnfl_t0": rnfl_t0, "rnfl_t1": rnfl_t1,
            "vf_t0":   vf_t0,   "vf_t1":   vf_t1,
            "tabular": tab,
            "labels":  labels,
        }


# ─────────────────────────────────────────────
# 2. TC-MMH model
# ─────────────────────────────────────────────

class TCMH_ConvNeXt(nn.Module):
    """
    ConvNeXt-V2 + Bi-LSTM with a horizon-conditioned shared prediction head.

    For each visit (T0, T1):
      1. Encode cpRNFL and VF images via a shared-weight ConvNeXt-V2 encoder
         (timm: `convnextv2_base.fcmae_ft_in1k`).
      2. Encode tabular covariates through a small MLP.
      3. Concatenate [rnfl_emb | vf_emb | tab_emb] into a per-visit feature.

    The two visit features form a length-2 sequence into a two-layer Bi-LSTM
    (256 hidden units). The last-timestep context is concatenated with a
    learned horizon embedding (one per Y2/Y3/Y4) and passed through a shared
    prediction head. This yields three sigmoid outputs from a single head,
    unlike the baseline's independent three-way Linear.
    """

    def __init__(self,
                 tab_dim:            int,
                 timm_model_name:    str   = TIMM_MODEL_NAME,
                 lstm_hidden:        int   = 256,
                 lstm_layers:        int   = 2,
                 horizon_embed_dim:  int   = HORIZON_EMBED_DIM,
                 tab_encoder_dim:    int   = TAB_ENCODER_DIM,
                 dropout:            float = 0.3,
                 pretrained:         bool  = True):
        super().__init__()

        self.H = len(HORIZONS)

        # Image encoder (shared weights across all four images)
        self.image_encoder = timm.create_model(
            timm_model_name, pretrained=pretrained, num_classes=0
        )
        img_dim = self.image_encoder.num_features

        # Tabular MLP
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tab_dim, 64), nn.LayerNorm(64),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, tab_encoder_dim), nn.GELU(),
        )

        # Per-visit feature: 2 images × img_dim + tab_encoder_dim
        visit_dim = 2 * img_dim + tab_encoder_dim

        self.lstm = nn.LSTM(
            input_size=visit_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        # Horizon-conditioned shared head
        self.horizon_embeddings = nn.Embedding(self.H, horizon_embed_dim)
        self.shared_head = nn.Sequential(
            nn.Linear(lstm_hidden * 2 + horizon_embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_layer = nn.Linear(128, 1)

    def encode_visit(self,
                     rnfl: torch.Tensor,
                     vf:   torch.Tensor,
                     tab:  torch.Tensor) -> torch.Tensor:
        """Encode one visit: [rnfl_emb | vf_emb | tab_emb]."""
        e_rnfl = self.image_encoder(rnfl)
        e_vf   = self.image_encoder(vf)
        e_tab  = self.tabular_encoder(tab)
        return torch.cat([e_rnfl, e_vf, e_tab], dim=-1)

    def forward(self, batch: dict) -> torch.Tensor:
        tab = batch["tabular"]

        v0 = self.encode_visit(batch["rnfl_t0"], batch["vf_t0"], tab)
        v1 = self.encode_visit(batch["rnfl_t1"], batch["vf_t1"], tab)

        seq, _ = self.lstm(torch.stack([v0, v1], dim=1))   # (B, 2, 2*hidden)
        ctx    = self.dropout(seq[:, -1, :])               # (B, 2*hidden)

        # Broadcast context against each horizon embedding, then predict
        B     = ctx.size(0)
        h_ids = torch.arange(self.H, device=ctx.device)
        h_emb = self.horizon_embeddings(h_ids)             # (H, embed)

        ctx_r   = ctx.unsqueeze(1).expand(-1, self.H, -1)          # (B, H, 2*hidden)
        h_emb_r = h_emb.unsqueeze(0).expand(B, -1, -1)             # (B, H, embed)
        joint   = torch.cat([ctx_r, h_emb_r], dim=-1)              # (B, H, 2*hidden+embed)

        logits = self.output_layer(self.shared_head(joint)).squeeze(-1)  # (B, H)
        return torch.sigmoid(logits)                                     # (B, H)


# ─────────────────────────────────────────────
# 3. Masked TC-MMH loss
# ─────────────────────────────────────────────

class MaskedTCMHLoss(nn.Module):
    """
    TC-MMH loss with the baseline's masked-BCE compatibility.

        L = L_bce_masked + lambda_mono * L_mono

    - L_bce_masked: identical to the baseline's `MaskedMultiHorizonBCE`
      (NaN targets are ignored; optional per-class weighting).
    - L_mono: mean over consecutive horizon pairs of
        relu(P_{h} - P_{h+1})^2
      averaged over samples where BOTH horizons have observed labels, so the
      penalty is not driven by predictions that lack any supervision.
    """

    def __init__(self,
                 class_weights: torch.Tensor = None,
                 lambda_mono:   float        = LAMBDA_MONO):
        super().__init__()
        self.bce           = nn.BCELoss(reduction="none")
        self.class_weights = class_weights
        self.lambda_mono   = lambda_mono

    def masked_bce(self,
                   preds:   torch.Tensor,
                   targets: torch.Tensor) -> torch.Tensor:
        mask   = ~torch.isnan(targets)
        safe_t = targets.clone()
        safe_t[~mask] = 0.0

        loss = self.bce(preds, safe_t)
        if self.class_weights is not None:
            w = torch.where(safe_t == 1,
                            self.class_weights[1],
                            self.class_weights[0])
            loss = loss * w
        loss = loss * mask.float()
        return loss.sum() / mask.float().sum().clamp(min=1.0)

    def temporal_monotonicity(self,
                              preds:   torch.Tensor,
                              targets: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(targets)                       # (B, H)
        total, n_pairs = torch.tensor(0.0, device=preds.device), 0
        for h in range(preds.size(1) - 1):
            pair_mask = (mask[:, h] & mask[:, h + 1]).float()
            if pair_mask.sum() == 0:
                continue
            diff = F.relu(preds[:, h] - preds[:, h + 1]).pow(2)
            total = total + (diff * pair_mask).sum() / pair_mask.sum().clamp(min=1.0)
            n_pairs += 1
        return total / max(n_pairs, 1)

    def forward(self,
                preds:   torch.Tensor,
                targets: torch.Tensor) -> dict:
        l_bce  = self.masked_bce(preds, targets)
        l_mono = self.temporal_monotonicity(preds, targets)
        total  = l_bce + self.lambda_mono * l_mono
        return {"total": total, "bce": l_bce.detach(), "mono": l_mono.detach()}


# ─────────────────────────────────────────────
# 4. Training loop
# ─────────────────────────────────────────────

def train_one_epoch(model:     nn.Module,
                    loader:    DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device:    torch.device) -> dict:
    model.train()
    running = {"total": 0.0, "bce": 0.0, "mono": 0.0}
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        preds  = model(batch)
        losses = criterion(preds, batch["labels"])
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()
        for k in running:
            running[k] += losses[k].item()
    return {k: v / len(loader) for k, v in running.items()}


@torch.no_grad()
def evaluate(model:  nn.Module,
             loader: DataLoader,
             device: torch.device) -> dict:
    """Compute AUROC, accuracy, and F1 averaged across available horizons."""
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = model(batch).cpu().numpy()
        lbls  = batch["labels"].cpu().numpy()
        all_preds.append(preds)
        all_labels.append(lbls)

    preds  = np.vstack(all_preds)
    labels = np.vstack(all_labels)

    metrics = {}
    aucs, accs, f1s = [], [], []

    for i, h in enumerate(HORIZONS):
        mask = ~np.isnan(labels[:, i])
        if mask.sum() < 10:
            continue
        y_true = labels[mask, i].astype(int)
        y_prob = preds[mask, i]
        y_pred = (y_prob >= 0.5).astype(int)

        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float("nan")

        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, zero_division=0)

        metrics[f"auc_y{h}"] = auc
        metrics[f"acc_y{h}"] = acc
        metrics[f"f1_y{h}"]  = f1
        aucs.append(auc); accs.append(acc); f1s.append(f1)

    metrics["auc_mean"] = float(np.nanmean(aucs)) if aucs else float("nan")
    metrics["acc_mean"] = float(np.nanmean(accs)) if accs else float("nan")
    metrics["f1_mean"]  = float(np.nanmean(f1s))  if f1s  else float("nan")
    return metrics


# ─────────────────────────────────────────────
# 5. Cross-validation  (patient-level, identical to baseline)
# ─────────────────────────────────────────────

def run_cross_validation(df:            pd.DataFrame,
                         image_dir:     str,
                         n_folds:       int   = 5,
                         epochs:        int   = 200,
                         lr:            float = 2e-5,
                         batch_size:    int   = 16,
                         weight_decay:  float = 1e-2,
                         warmup_epochs: int   = 10,
                         patience:      int   = 11,
                         lambda_mono:   float = LAMBDA_MONO,
                         output_dir:    str   = "./results_tcmh") -> list:
    """
    Patient-level K-fold cross-validation (GroupKFold on patient_id) —
    identical to the baseline pipeline so that TC-MMH performance is directly
    comparable to ConvNeXt / ViT / MobileNet / EfficientNet baselines.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    set_seed(42)

    groups   = df["patient_id"].values
    splitter = GroupKFold(n_splits=n_folds)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(df, groups=groups), 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/{n_folds}")
        print(f"{'='*60}")

        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val   = df.iloc[val_idx].reset_index(drop=True)

        # Class weights from training fold (identical to baseline)
        y_all = pd.concat([df_train[f"label_y{h}"] for h in HORIZONS]).dropna()
        n_pos = (y_all == 1).sum()
        n_neg = (y_all == 0).sum()
        w_pos = n_neg / n_pos if n_pos > 0 else 1.0
        cw    = torch.tensor([1.0, w_pos], dtype=torch.float32).to(device)

        train_ds = GlaucomaSequenceDataset(df_train, image_dir,
                                           fit_scaler=True, augment=True)
        val_ds   = GlaucomaSequenceDataset(df_val,   image_dir,
                                           scaler=train_ds.scaler, augment=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                                  num_workers=4, pin_memory=True)

        tab_dim = train_ds.tab_data.shape[1]
        model   = TCMH_ConvNeXt(tab_dim=tab_dim).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                      weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        criterion = MaskedTCMHLoss(class_weights=cw, lambda_mono=lambda_mono)

        best_auc, patience_ctr = 0.0, 0
        best_state = None

        for epoch in range(1, epochs + 1):
            train_stats = train_one_epoch(model, train_loader, optimizer,
                                          criterion, device)
            if epoch <= warmup_epochs:
                scheduler.step()

            val_metrics = evaluate(model, val_loader, device)
            auc_mean    = val_metrics.get("auc_mean", 0.0)

            print(f"  Epoch {epoch:3d} | "
                  f"loss {train_stats['total']:.4f} "
                  f"(bce {train_stats['bce']:.4f} mono {train_stats['mono']:.4f}) | "
                  f"val AUC {auc_mean:.4f} | "
                  f"acc {val_metrics.get('acc_mean', 0):.4f} | "
                  f"F1 {val_metrics.get('f1_mean', 0):.4f}")

            if auc_mean > best_auc:
                best_auc   = auc_mean
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

        ckpt_path = Path(output_dir) / f"fold{fold}_tcmh_best.pt"
        torch.save(best_state, ckpt_path)
        print(f"  Best val AUC: {best_auc:.4f} → saved to {ckpt_path}")

        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        final_metrics = evaluate(model, val_loader, device)
        final_metrics["fold"] = fold
        fold_results.append(final_metrics)

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS (mean ± std)")
    print("=" * 60)
    results_df = pd.DataFrame(fold_results)
    for col in ["auc_mean", "acc_mean", "f1_mean"]:
        if col in results_df.columns:
            print(f"  {col}: {results_df[col].mean():.4f} ± {results_df[col].std():.4f}")

    results_path = Path(output_dir) / "tcmh_cv_results.json"
    results_df.to_json(results_path, orient="records", indent=2)
    print(f"\nFull results saved → {results_path}")

    return fold_results


# ─────────────────────────────────────────────
# 6. Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the proposed TC-MMH ConvNeXt-V2 model for multi-horizon "
                    "glaucoma progression prediction. Uses the same data pipeline, "
                    "cohort, and patient-level CV as `model_training.py`."
    )
    parser.add_argument("--sequences_path", required=True,
                        help="Path to sequences.csv from sequence_generation.py.")
    parser.add_argument("--image_dir",      required=True,
                        help="Root directory containing cpRNFL and VF images.")
    parser.add_argument("--output_dir",     default="./results_tcmh",
                        help="Directory to save checkpoints and results.")
    parser.add_argument("--n_folds",        type=int,   default=5)
    parser.add_argument("--epochs",         type=int,   default=200)
    parser.add_argument("--lr",             type=float, default=2e-5)
    parser.add_argument("--batch_size",     type=int,   default=16)
    parser.add_argument("--weight_decay",   type=float, default=1e-2)
    parser.add_argument("--warmup_epochs",  type=int,   default=10)
    parser.add_argument("--patience",       type=int,   default=11)
    parser.add_argument("--lambda_mono",    type=float, default=LAMBDA_MONO,
                        help="Weight for the temporal-monotonicity penalty.")
    args = parser.parse_args()

    df = pd.read_csv(args.sequences_path)
    print(f"Loaded {len(df):,} sequences from {args.sequences_path}")

    run_cross_validation(
        df            = df,
        image_dir     = args.image_dir,
        n_folds       = args.n_folds,
        epochs        = args.epochs,
        lr            = args.lr,
        batch_size    = args.batch_size,
        weight_decay  = args.weight_decay,
        warmup_epochs = args.warmup_epochs,
        patience      = args.patience,
        lambda_mono   = args.lambda_mono,
        output_dir    = args.output_dir,
    )