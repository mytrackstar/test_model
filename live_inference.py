"""
Live Inference – Exercise Detection + Rep Counting
===================================================
Reads new IMU rows (stdin / CSV / direct call) and runs a sliding-window
classifier to identify the current exercise and count repetitions in real time.

Usage examples:
    # Replay a recorded CSV to simulate live inference
    python live_inference.py --replay path/to/recording.csv

    # Use as a library (see LiveSession class below)
    session = LiveSession(artifacts_dir="artifacts")
    for row in row_stream:
        result = session.push_row(row)
        if result:
            print(result)   # {"exercise": "Push ups", "reps": 3, "confidence": 0.97}
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt, find_peaks

# ──────────────────────────────────────────────────────────────────────────────
# Duplicate the model definition here so this file is self-contained
# ──────────────────────────────────────────────────────────────────────────────

FEATURES = ["ax", "ay", "az", "gx", "gy", "gz", "qw", "qx", "qy", "qz"]
N_FEATURES = len(FEATURES)
DEVICE = "cpu"   # inference on CPU is fast enough


class SpatialAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Linear(hidden, 1)

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return (x * w).sum(dim=1)


class IMUStream(nn.Module):
    def __init__(self, n_features=N_FEATURES):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, 5, padding=2), nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU(),
        )
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.attn = SpatialAttention(256)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        return self.attn(x)


class DualIMUNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.arm_stream  = IMUStream()
        self.foot_stream = IMUStream()
        self.fc = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, arm, foot):
        return self.fc(torch.cat([self.arm_stream(arm), self.foot_stream(foot)], dim=1))


# ──────────────────────────────────────────────────────────────────────────────
# SCALER (reproduces StandardScaler without sklearn dependency)
# ──────────────────────────────────────────────────────────────────────────────

class NumpyScaler:
    def __init__(self, mean: list, scale: list):
        self.mean  = np.array(mean,  dtype=np.float32)
        self.scale = np.array(scale, dtype=np.float32)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# REP COUNTER
# ──────────────────────────────────────────────────────────────────────────────

# Maps exercise → which feature index (in FEATURES) carries the rep signal,
# and whether to look for peaks (+1) or valleys (-1).
# These are empirically chosen based on the anatomy of each exercise.
REP_SIGNAL_CONFIG: dict[str, dict] = {
    # min_dist in ARM samples @ ~50 Hz.  prominence in normalised g-units.
    # Conservative values so sub-movements (e.g. partial bounce) don't count.
    "Push ups":         {"feature": 2, "direction":  1, "min_dist": 60, "prominence": 0.5},  # az_arm peak = top position
    "Kniebeuge":        {"feature": 2, "direction": -1, "min_dist": 60, "prominence": 0.5},  # az_foot valley = bottom squat
    "Hampelmanner":     {"feature": 0, "direction":  1, "min_dist": 40, "prominence": 0.4},  # ax_arm peak = arms out
    "Sit Up":           {"feature": 1, "direction":  1, "min_dist": 60, "prominence": 0.5},  # ay_arm peak = upright
    "Montain climbers": {"feature": 5, "direction":  1, "min_dist": 40, "prominence": 0.4},  # gz_arm peak = alternating legs
    "Rauschen":         None,
}

_DEFAULT_REP = {"feature": 2, "direction": 1, "min_dist": 60, "prominence": 0.5}


def _bandpass(signal: np.ndarray, lowcut: float = 0.5, highcut: float = 5.0, fs: float = 50.0) -> np.ndarray:
    """2nd-order Butterworth band-pass, handles short signals gracefully."""
    nyq = fs / 2.0
    low,  high = lowcut / nyq, min(highcut / nyq, 0.99)
    if low >= high or len(signal) < 27:
        return signal
    try:
        b, a = butter(2, [low, high], btype="band")
        return filtfilt(b, a, signal)
    except Exception:
        return signal


class RepCounter:
    """
    Counts reps using peak detection on a low-pass-filtered acceleration signal.

    Uses an absolute sample counter so peaks are never double-counted even
    as the ring buffer scrolls.
    """

    BUFFER_SIZE = 500   # ~5 s @ 100 Hz
    FS = 50.0           # approximate sample rate after ARM-only stream

    def __init__(self):
        self._buf: deque[np.ndarray] = deque(maxlen=self.BUFFER_SIZE)
        self._exercise: str = ""
        self._rep_count: int = 0
        self._total_pushed: int = 0          # absolute ARM sample counter
        self._last_counted_abs: int = -1     # absolute index of last counted peak

    def reset(self):
        self._buf.clear()
        self._exercise = ""
        self._rep_count = 0
        self._total_pushed = 0
        self._last_counted_abs = -1

    def push(self, arm_row: np.ndarray, exercise: str) -> int:
        """
        Push one ARM sensor frame (shape [N_FEATURES]) and the current exercise
        classification.  Returns the cumulative rep count.
        """
        if exercise != self._exercise:
            self._exercise = exercise
            self._rep_count = 0
            self._buf.clear()
            self._total_pushed = 0
            self._last_counted_abs = -1

        if exercise == "Rauschen" or exercise not in REP_SIGNAL_CONFIG:
            return 0

        self._buf.append(arm_row)
        self._total_pushed += 1

        # Only re-count every 10 new samples to avoid thrashing
        if len(self._buf) < 30 or self._total_pushed % 10 != 0:
            return self._rep_count

        cfg = REP_SIGNAL_CONFIG.get(exercise) or _DEFAULT_REP
        feat_idx  = cfg["feature"]
        direction = cfg["direction"]
        min_dist  = cfg["min_dist"]
        prom      = cfg["prominence"]

        buf_data = list(self._buf)
        sig = np.array([r[feat_idx] for r in buf_data], dtype=np.float32)
        sig = _bandpass(sig, fs=self.FS) * direction

        peaks, _ = find_peaks(sig, distance=min_dist, prominence=prom)

        # Convert buffer-relative peak indices to absolute stream positions.
        # buffer start in absolute coords = total_pushed - len(buf)
        buf_start_abs = self._total_pushed - len(buf_data)

        for p in peaks:
            abs_pos = buf_start_abs + int(p)
            # Only count peaks strictly newer than the last counted one
            if abs_pos > self._last_counted_abs:
                self._rep_count += 1
                self._last_counted_abs = abs_pos

        return self._rep_count


# ──────────────────────────────────────────────────────────────────────────────
# LIVE SESSION
# ──────────────────────────────────────────────────────────────────────────────

class LiveSession:
    """
    Manages real-time exercise detection + rep counting.

    Push rows one at a time via push_row().  When enough data has
    accumulated (window_size rows per band), a classification is returned.

    A 5-frame majority-vote smoother prevents brief misclassifications
    (e.g. a single Rauschen frame) from resetting the rep counter.
    """

    SMOOTH_WINDOW = 5   # number of consecutive predictions to smooth over

    def __init__(self, artifacts_dir: str = "artifacts", confidence_threshold: float = 0.65):
        self.threshold = confidence_threshold
        self._load_artifacts(artifacts_dir)
        self._arm_buf:  deque[np.ndarray] = deque(maxlen=self.window_size)
        self._foot_buf: deque[np.ndarray] = deque(maxlen=self.window_size)
        self._rep_counter = RepCounter()
        self._pred_history: deque[str] = deque(maxlen=self.SMOOTH_WINDOW)
        self._stable_exercise = "Rauschen"
        self._step = 0   # for stride-based triggering

    def _load_artifacts(self, artifacts_dir: str):
        d = Path(artifacts_dir)
        scalers = json.loads((d / "scalers.json").read_text())
        enc     = json.loads((d / "label_encoder.json").read_text())

        self.classes     = enc["classes"]
        self.window_size = scalers.get("window_size", 100)
        self.arm_scaler  = NumpyScaler(scalers["arm"]["mean"],  scalers["arm"]["scale"])
        self.foot_scaler = NumpyScaler(scalers["foot"]["mean"], scalers["foot"]["scale"])

        model = DualIMUNet(num_classes=len(self.classes))
        state = torch.load(str(d / "dual_imu_model.pt"), map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        self.model = model

    def push_row(self, row: dict) -> Optional[dict]:
        """
        Push a single sensor row (dict with keys matching FEATURES + 'band').
        Returns a result dict when a classification fires, else None.
        """
        band = str(row.get("band", "")).upper().strip()
        try:
            vec = np.array([float(row[f]) for f in FEATURES], dtype=np.float32)
        except (KeyError, ValueError):
            return None

        if band == "ARM":
            self._arm_buf.append(vec)
        elif band == "FOOT":
            self._foot_buf.append(vec)
        else:
            return None

        self._step += 1
        stride = max(1, self.window_size // 5)   # fire every 20% of a window

        if (len(self._arm_buf) < self.window_size or
                len(self._foot_buf) < self.window_size or
                self._step % stride != 0):
            # Still feed rep counter even without a new classification
            if band == "ARM":
                self._rep_counter.push(vec, self._stable_exercise)
            return None

        # ── Build tensors ─────────────────────────────────────────────────────
        arm_win  = self.arm_scaler.transform(np.array(self._arm_buf,  dtype=np.float32))
        foot_win = self.foot_scaler.transform(np.array(self._foot_buf, dtype=np.float32))

        arm_t  = torch.tensor(arm_win[np.newaxis],  dtype=torch.float32)
        foot_t = torch.tensor(foot_win[np.newaxis], dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(arm_t, foot_t)
            probs  = F.softmax(logits, dim=1).squeeze().numpy()

        pred_idx    = int(np.argmax(probs))
        confidence  = float(probs[pred_idx])
        raw_exercise = self.classes[pred_idx] if confidence >= self.threshold else "Rauschen"

        # ── Smooth: majority vote over last SMOOTH_WINDOW predictions ─────────
        self._pred_history.append(raw_exercise)
        counts: dict[str, int] = {}
        for p in self._pred_history:
            counts[p] = counts.get(p, 0) + 1
        self._stable_exercise = max(counts, key=lambda k: counts[k])

        if band == "ARM":
            reps = self._rep_counter.push(vec, self._stable_exercise)
        else:
            reps = self._rep_counter._rep_count

        return {
            "exercise":   self._stable_exercise,
            "confidence": round(confidence, 4),
            "reps":       reps,
            "probs":      {c: round(float(p), 4) for c, p in zip(self.classes, probs)},
        }

    def reset(self):
        self._arm_buf.clear()
        self._foot_buf.clear()
        self._rep_counter.reset()
        self._pred_history.clear()
        self._stable_exercise = "Rauschen"
        self._step = 0


# ──────────────────────────────────────────────────────────────────────────────
# REPLAY MODE (CLI)
# ──────────────────────────────────────────────────────────────────────────────

def replay(csv_path: str, artifacts_dir: str, realtime: bool = False, threshold: float = 0.65):
    df = pd.read_csv(csv_path)

    # Normalise label for comparison
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip()

    session = LiveSession(artifacts_dir=artifacts_dir, confidence_threshold=threshold)

    # Approximate inter-row delay
    row_delay = 1 / 100.0  # 100 Hz

    print(f"\nReplaying {csv_path} ({len(df)} rows) …\n")
    print(f"{'#':>4}  {'Exercise':<22} {'Conf':>6}  {'Reps':>4}  {'TrueLabel':<22}")
    print("-" * 70)

    last_print = {"exercise": None, "reps": -1}
    row_count  = 0

    for _, row in df.iterrows():
        row_count += 1
        result = session.push_row(row.to_dict())

        if realtime:
            time.sleep(row_delay)

        if result is None:
            continue

        ex   = result["exercise"]
        reps = result["reps"]
        true = row.get("label", "?")

        # Only print when something changes
        if ex != last_print["exercise"] or reps != last_print["reps"]:
            print(f"{row_count:4d}  {ex:<22} {result['confidence']:6.3f}  {reps:4d}  {true:<22}")
            last_print = {"exercise": ex, "reps": reps}

    print("\nDone.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live exercise inference from CSV replay")
    parser.add_argument("--replay",    required=True, help="CSV file to replay")
    parser.add_argument("--artifacts", default="artifacts", help="Directory with saved model artifacts")
    parser.add_argument("--threshold", type=float, default=0.65, help="Minimum confidence to accept a classification")
    parser.add_argument("--realtime",  action="store_true", help="Sleep between rows to simulate real-time")
    args = parser.parse_args()

    replay(args.replay, args.artifacts, realtime=args.realtime, threshold=args.threshold)
