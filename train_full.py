"""
Full Training Pipeline – Dual IMU Exercise Recognition + Rep Counting
======================================================================
Loads all CSV files from the data directory, applies data augmentation,
trains a CNN+BiLSTM classifier, and saves all artifacts needed for live
inference.

Features  : ax,ay,az,gx,gy,gz,qw,qx,qy,qz  (10 per band = 20 total)
Bands     : ARM + FOOT (aligned by timestamp)
Exercises : Push ups, Kniebeuge, Hampelmanner, Situp, Montainclimbers, Rauschen

Usage:
    python train_full.py [--data_dir ./] [--out_dir ./artifacts] [--epochs 60]
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    data_dir: str = "."                # directory with real CSV files
    extra_data_dirs: tuple = ()        # additional directories (e.g. synthetic)
    out_dir: str = "artifacts"
    window_size: int = 100             # ~1 s @ ~100 Hz
    stride: int = 20                   # heavy overlap → many windows
    batch_size: int = 64
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    val_size: float = 0.2
    sync_tolerance_ms: int = 50
    aug_factor: int = 8                # how many augmented copies per real window
    min_sys_cal: int = 2
    min_gyro_cal: int = 2

FEATURES = ["ax", "ay", "az", "gx", "gy", "gz", "qw", "qx", "qy", "qz"]   # 10 features
N_FEATURES = len(FEATURES)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING & ALIGNMENT
# ──────────────────────────────────────────────────────────────────────────────

def load_all_csvs(data_dirs: list[str]) -> pd.DataFrame:
    """Load and concatenate all CSV files found in one or more directories."""
    dfs = []
    total_files = 0
    for data_dir in data_dirs:
        p = Path(data_dir)
        files = sorted(p.glob("*.csv"))
        for f in files:
            df = pd.read_csv(f)
            df["source_file"] = f.stem
            dfs.append(df)
            total_files += 1

    if not dfs:
        raise FileNotFoundError(f"No CSV files found in: {data_dirs}")

    merged = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(merged)} rows from {total_files} files across {len(data_dirs)} director{'y' if len(data_dirs)==1 else 'ies'}")
    return merged


def preprocess(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    df["band"] = df["band"].astype(str).str.upper().str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp_dt"])

    for col in ["millis", *FEATURES]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["sysCal", "gyroCal", "accelCal", "magCal"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    return df


def align_streams(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Align ARM↔FOOT rows by nearest timestamp per source file."""
    arm = df[df["band"] == "ARM"].sort_values("timestamp_dt").reset_index(drop=True)
    foot = df[df["band"] == "FOOT"].sort_values("timestamp_dt").reset_index(drop=True)

    aligned = pd.merge_asof(
        arm, foot,
        on="timestamp_dt",
        direction="nearest",
        tolerance=pd.Timedelta(milliseconds=cfg.sync_tolerance_ms),
        suffixes=("_arm", "_foot"),
    )

    foot_cols = [f"{c}_foot" for c in FEATURES]
    aligned = aligned.dropna(subset=foot_cols).reset_index(drop=True)

    # Calibration filter
    for band in ("arm", "foot"):
        sys_col = f"sysCal_{band}"
        gyro_col = f"gyroCal_{band}"
        if sys_col in aligned.columns:
            aligned = aligned[aligned[sys_col] >= cfg.min_sys_cal]
        if gyro_col in aligned.columns:
            aligned = aligned[aligned[gyro_col] >= cfg.min_gyro_cal]

    print(f"  Aligned rows after calibration filter: {len(aligned)}")
    return aligned.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# WINDOWING
# ──────────────────────────────────────────────────────────────────────────────

def make_windows(data: np.ndarray, labels: np.ndarray, window: int, stride: int):
    """Return (windows, window_labels) using majority vote per window."""
    X, y = [], []
    for i in range(0, len(data) - window + 1, stride):
        seg = data[i : i + window]
        seg_labels = labels[i : i + window]
        vals, counts = np.unique(seg_labels, return_counts=True)
        majority = vals[np.argmax(counts)]
        X.append(seg)
        y.append(majority)
    return np.array(X, dtype=np.float32), np.array(y)


# ──────────────────────────────────────────────────────────────────────────────
# DATA AUGMENTATION
# ──────────────────────────────────────────────────────────────────────────────

def _jitter(x: np.ndarray, sigma: float = 0.025) -> np.ndarray:
    return x + np.random.normal(0, sigma, x.shape).astype(np.float32)


def _scale(x: np.ndarray, lo: float = 0.85, hi: float = 1.15) -> np.ndarray:
    factor = np.random.uniform(lo, hi)
    return (x * factor).astype(np.float32)


def _time_warp(x: np.ndarray) -> np.ndarray:
    """Random stretch/compress via linear interpolation."""
    T, F = x.shape
    warp = np.random.uniform(0.85, 1.15)
    orig_t = np.arange(T)
    new_len = int(T * warp)
    new_t = np.linspace(0, T - 1, new_len)
    warped = np.stack(
        [np.interp(new_t, orig_t, x[:, f]) for f in range(F)], axis=1
    ).astype(np.float32)
    # Resize back to T via interpolation
    t_final = np.linspace(0, new_len - 1, T)
    t_src = np.arange(new_len)
    return np.stack(
        [np.interp(t_final, t_src, warped[:, f]) for f in range(F)], axis=1
    ).astype(np.float32)


def _mag_perturb(x: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """Per-channel multiplicative noise."""
    factors = 1.0 + np.random.normal(0, sigma, (1, x.shape[1])).astype(np.float32)
    return (x * factors).astype(np.float32)


def _flip_gravity(x: np.ndarray) -> np.ndarray:
    """Randomly negate gravity axis in accelerometer (simulate sensor flip)."""
    out = x.copy()
    axis = np.random.randint(0, 3)   # ax, ay, or az  (indices 0-2)
    out[:, axis] *= -1
    return out


def augment_window(arm: np.ndarray, foot: np.ndarray, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate n augmented copies of a single (arm, foot) window pair."""
    AUG_FNS = [_jitter, _scale, _time_warp, _mag_perturb, _flip_gravity]
    results = []
    for _ in range(n):
        # Pick 1-3 random augmentations to apply in sequence
        fns = np.random.choice(AUG_FNS, size=np.random.randint(1, 4), replace=False)  # type: ignore[arg-type]
        a, f = arm.copy(), foot.copy()
        for fn in fns:
            a = fn(a)
            f = fn(f)
        results.append((a, f))
    return results


def build_augmented_dataset(
    X_arm: np.ndarray,
    X_foot: np.ndarray,
    y: np.ndarray,
    aug_factor: int,
    exclude_label: str = "Rauschen",
    encoder: LabelEncoder | None = None,
):
    """
    Augment training data.
    - 'Rauschen' (rest/noise) is augmented less aggressively (factor//2)
      because we don't want the model to over-fit on noise patterns.
    """
    rauschen_idx = None
    if encoder is not None:
        try:
            rauschen_idx = list(encoder.classes_).index(exclude_label)
        except ValueError:
            pass

    aug_arm, aug_foot, aug_y = [X_arm], [X_foot], [y]
    for i in range(len(y)):
        n = aug_factor // 2 if (rauschen_idx is not None and y[i] == rauschen_idx) else aug_factor
        copies = augment_window(X_arm[i], X_foot[i], n)
        for a, f in copies:
            aug_arm.append(a[np.newaxis])
            aug_foot.append(f[np.newaxis])
            aug_y.append(np.array([y[i]]))

    return (
        np.concatenate(aug_arm, axis=0),
        np.concatenate(aug_foot, axis=0),
        np.concatenate(aug_y, axis=0),
    )


# ──────────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────────

class IMUDataset(Dataset):
    def __init__(self, arm, foot, labels):
        self.arm = torch.tensor(arm, dtype=torch.float32)
        self.foot = torch.tensor(foot, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.arm[idx], self.foot[idx], self.labels[idx]


# ──────────────────────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────────────────────

class SpatialAttention(nn.Module):
    """Simple 1-D temporal attention over LSTM output."""
    def __init__(self, hidden: int):
        super().__init__()
        self.attn = nn.Linear(hidden, 1)

    def forward(self, x):           # [B, T, H]
        weights = torch.softmax(self.attn(x), dim=1)   # [B, T, 1]
        return (x * weights).sum(dim=1)                # [B, H]


class IMUStream(nn.Module):
    def __init__(self, n_features: int = N_FEATURES):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.attn = SpatialAttention(256)   # bidirectional → 256

    def forward(self, x):           # x: [B, T, F]
        x = x.permute(0, 2, 1)     # [B, F, T]
        x = self.cnn(x)             # [B, 128, T//4]
        x = x.permute(0, 2, 1)     # [B, T//4, 128]
        x, _ = self.lstm(x)         # [B, T//4, 256]
        return self.attn(x)         # [B, 256]


class DualIMUNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.arm_stream = IMUStream()
        self.foot_stream = IMUStream()
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, arm, foot):
        a = self.arm_stream(arm)
        f = self.foot_stream(foot)
        return self.fc(torch.cat([a, f], dim=1))


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for arm, foot, lbl in loader:
            pred = model(arm.to(device), foot.to(device))
            correct += (pred.argmax(1) == lbl.to(device)).sum().item()
            total += lbl.size(0)
    return correct / total if total else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run(cfg: Config):
    _set_seeds(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print("\n[1/6] Loading CSV files …")
    all_dirs = [cfg.data_dir] + list(cfg.extra_data_dirs)
    raw = load_all_csvs(all_dirs)
    raw = preprocess(raw, cfg)
    aligned = align_streams(raw, cfg)

    arm_data  = aligned[[f"{c}_arm"  for c in FEATURES]].to_numpy(dtype=np.float32)
    foot_data = aligned[[f"{c}_foot" for c in FEATURES]].to_numpy(dtype=np.float32)
    labels    = aligned["label_arm"].astype(str).to_numpy()

    print(f"  Label distribution:\n{pd.Series(labels).value_counts().to_string()}")

    # ── 2. Windows ───────────────────────────────────────────────────────────
    print("\n[2/6] Creating windows …")
    combined = np.concatenate([arm_data, foot_data], axis=1)   # [N, 20]
    X_all, y_str = make_windows(combined, labels, cfg.window_size, cfg.stride)
    X_arm_all  = X_all[:, :, :N_FEATURES]
    X_foot_all = X_all[:, :, N_FEATURES:]
    print(f"  Windows: {len(y_str)}  →  shape {X_arm_all.shape}")

    # ── 3. Encode labels ──────────────────────────────────────────────────────
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y_str)
    print(f"  Classes: {list(encoder.classes_)}")

    # ── 4. Train/Val split ────────────────────────────────────────────────────
    print("\n[3/6] Splitting train/val …")
    idx = np.arange(len(y_enc))
    train_idx, val_idx = train_test_split(
        idx, test_size=cfg.val_size, random_state=cfg.seed,
        stratify=y_enc,
    )
    X_arm_tr,  X_arm_val  = X_arm_all[train_idx],  X_arm_all[val_idx]
    X_foot_tr, X_foot_val = X_foot_all[train_idx], X_foot_all[val_idx]
    y_tr, y_val           = y_enc[train_idx],       y_enc[val_idx]

    # ── 5. Normalise (fit on train) ───────────────────────────────────────────
    arm_scaler  = StandardScaler().fit(X_arm_tr.reshape(-1, N_FEATURES))
    foot_scaler = StandardScaler().fit(X_foot_tr.reshape(-1, N_FEATURES))

    def scale(windows, sc):
        flat = sc.transform(windows.reshape(-1, N_FEATURES))
        return flat.reshape(windows.shape).astype(np.float32)

    X_arm_tr   = scale(X_arm_tr,  arm_scaler)
    X_arm_val  = scale(X_arm_val, arm_scaler)
    X_foot_tr  = scale(X_foot_tr,  foot_scaler)
    X_foot_val = scale(X_foot_val, foot_scaler)

    # ── 6. Augment train set ──────────────────────────────────────────────────
    print(f"\n[4/6] Augmenting training set (×{cfg.aug_factor}) …")
    X_arm_tr, X_foot_tr, y_tr = build_augmented_dataset(
        X_arm_tr, X_foot_tr, y_tr, cfg.aug_factor, encoder=encoder
    )
    # Shuffle
    perm = np.random.permutation(len(y_tr))
    X_arm_tr, X_foot_tr, y_tr = X_arm_tr[perm], X_foot_tr[perm], y_tr[perm]
    print(f"  Train: {len(y_tr)}  Val: {len(y_val)}")

    # ── 7. DataLoaders ────────────────────────────────────────────────────────
    train_ds = IMUDataset(X_arm_tr, X_foot_tr, y_tr)
    val_ds   = IMUDataset(X_arm_val, X_foot_val, y_val)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    # ── 8. Model ──────────────────────────────────────────────────────────────
    print(f"\n[5/6] Training on {DEVICE} …")
    model = DualIMUNet(num_classes=len(encoder.classes_)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # Class weights to handle any imbalance
    class_counts = np.bincount(y_tr, minlength=len(encoder.classes_))
    weights      = 1.0 / (class_counts + 1)
    weights      = torch.tensor(weights / weights.sum() * len(encoder.classes_), dtype=torch.float32).to(DEVICE)
    loss_fn   = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for arm, foot, lbl in train_dl:
            arm, foot, lbl = arm.to(DEVICE), foot.to(DEVICE), lbl.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(arm, foot), lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        tr_acc  = accuracy(model, train_dl, DEVICE)
        val_acc = accuracy(model, val_dl,   DEVICE)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{cfg.epochs} | loss {total_loss:7.3f} | train {tr_acc:.3f} | val {val_acc:.3f}")

    print(f"\n  Best val accuracy: {best_val_acc:.4f}")

    # ── 9. Evaluate best model ────────────────────────────────────────────────
    model.load_state_dict(best_state)  # type: ignore[arg-type]
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for arm, foot, lbl in val_dl:
            pred = model(arm.to(DEVICE), foot.to(DEVICE)).argmax(1).cpu().numpy()
            all_preds.extend(pred)
            all_true.extend(lbl.numpy())

    print("\n[6/6] Classification report (validation set):")
    print(classification_report(all_true, all_preds, target_names=encoder.classes_))

    cm = confusion_matrix(all_true, all_preds)
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_).to_string())

    # ── 10. Save artifacts ────────────────────────────────────────────────────
    model_path = out_dir / "dual_imu_model.pt"
    torch.save(best_state, model_path)

    enc_path = out_dir / "label_encoder.json"
    enc_path.write_text(
        json.dumps({"classes": encoder.classes_.tolist()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    scaler_path = out_dir / "scalers.json"
    scaler_path.write_text(
        json.dumps(
            {
                "arm":  {"mean": arm_scaler.mean_.tolist(),  "scale": arm_scaler.scale_.tolist()},
                "foot": {"mean": foot_scaler.mean_.tolist(), "scale": foot_scaler.scale_.tolist()},
                "features": FEATURES,
                "window_size": cfg.window_size,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nArtifacts saved to: {out_dir.resolve()}")
    print(f"  {model_path.name}")
    print(f"  {enc_path.name}")
    print(f"  {scaler_path.name}")

    return model, encoder


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dual-IMU exercise classifier with augmentation")
    parser.add_argument("--data_dir",        default=".", help="Directory containing real CSV files")
    parser.add_argument("--extra_data_dirs", nargs="*",  default=[], help="Additional directories (e.g. synthetic/)")
    parser.add_argument("--out_dir",         default="artifacts")
    parser.add_argument("--window",   type=int,   default=100)
    parser.add_argument("--stride",   type=int,   default=20)
    parser.add_argument("--epochs",   type=int,   default=60)
    parser.add_argument("--batch",    type=int,   default=64)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--aug",      type=int,   default=8,  help="Augmentation multiplier")
    args = parser.parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        extra_data_dirs=tuple(args.extra_data_dirs),
        out_dir=args.out_dir,
        window_size=args.window,
        stride=args.stride,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        aug_factor=args.aug,
    )
    run(cfg)
