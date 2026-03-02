"""
Synthetic IMU Data Generator
=============================
Exploits the periodicity of workout exercises to synthesise realistic
new training recordings.

Strategy
--------
1.  Load each labelled CSV and split into ARM / FOOT streams.
2.  For each stream, find the feature whose FFT has the highest peak
    (= dominant periodic axis for that exercise) and extract its
    individual cycles via peak detection.
3.  Build a library of clean single-cycle segments per (exercise, band).
4.  For each desired synthetic recording:
    a.  Pick a random cycle from the library (or blend two).
    b.  Resample it to a slightly different duration (±15 %) to vary tempo.
    c.  Add per-channel Gaussian noise, multiplicative amplitude jitter,
        and a random DC offset shift.
    d.  Repeat for N_reps repetitions and stitch into a full sequence.
    e.  Re-align ARM + FOOT by inserting matching FOOT samples at the
        same timestamps.
5.  Write the result out as a new CSV in the same format as the originals.

Usage
-----
    python generate_synthetic.py [--data_dir ./] [--out_dir ./synthetic]
                                 [--n_recordings 10] [--min_reps 8]
                                 [--max_reps 20]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

# ──────────────────────────────────────────────────────────────────────────────

FEATURES = ["ax", "ay", "az", "gx", "gy", "gz", "qw", "qx", "qy", "qz"]
N_FEAT   = len(FEATURES)
FS       = 50.0   # effective sample rate per band (ARM or FOOT)

# Which feature index carries the strongest periodic signal per exercise.
# Used only to FIND cycle boundaries; all 10 features are synthesised.
DOMINANT_FEAT: dict[str, int] = {
    "Push ups":         2,   # az_arm
    "Kniebeuge":        2,   # az_foot
    "Hampelmanner":     0,   # ax_arm
    "Sit Up":           1,   # ay_arm
    "Montain climbers": 5,   # gz_arm
    "Rauschen":         2,   # unused – Rauschen handled separately
}
DEFAULT_DOMINANT = 2


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _bandpass(sig: np.ndarray, lo: float = 0.3, hi: float = 6.0) -> np.ndarray:
    nyq = FS / 2.0
    lo_n, hi_n = lo / nyq, min(hi / nyq, 0.99)
    if lo_n >= hi_n or len(sig) < 27:
        return sig
    b, a = butter(2, [lo_n, hi_n], btype="band")
    return filtfilt(b, a, sig).astype(np.float32)


def _dominant_frequency(sig: np.ndarray) -> float:
    """Return dominant frequency (Hz) of a 1-D signal."""
    fft = np.abs(np.fft.rfft(sig - sig.mean()))
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / FS)
    mask = (freqs >= 0.3) & (freqs <= 6.0)
    if not mask.any():
        return 1.0
    idx = np.argmax(fft[mask])
    return float(freqs[mask][idx])


def _extract_cycles(
    data: np.ndarray,           # [N, N_FEAT]
    feat_idx: int,
    min_reps_in_recording: int = 3,
) -> list[np.ndarray]:
    """
    Extract individual exercise cycles as numpy arrays of shape [T_cycle, N_FEAT].
    Returns an empty list if the signal is too noisy or aperiodic.
    """
    sig = _bandpass(data[:, feat_idx].astype(np.float32))
    dom_freq = _dominant_frequency(sig)
    min_dist = max(10, int(FS / dom_freq * 0.6))   # at least 60 % of one period

    # Normalise so prominence is scale-independent
    sig_norm = (sig - sig.mean()) / (sig.std() + 1e-6)
    peaks, props = find_peaks(sig_norm, distance=min_dist, prominence=0.3)

    if len(peaks) < min_reps_in_recording:
        return []

    cycles = []
    for i in range(len(peaks) - 1):
        start, end = peaks[i], peaks[i + 1]
        if end - start > 5:
            cycles.append(data[start:end].astype(np.float32))

    return cycles


def _resample_cycle(cycle: np.ndarray, new_len: int) -> np.ndarray:
    """Linearly resample a [T, F] cycle to a different length."""
    T, F = cycle.shape
    old_t = np.linspace(0, 1, T)
    new_t = np.linspace(0, 1, new_len)
    return np.stack(
        [np.interp(new_t, old_t, cycle[:, f]) for f in range(F)], axis=1
    ).astype(np.float32)


def _augment_cycle(cycle: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply small random perturbations to a single cycle."""
    c = cycle.copy()

    # Per-channel amplitude jitter
    amp = rng.uniform(0.85, 1.15, (1, c.shape[1])).astype(np.float32)
    c *= amp

    # DC offset shift (simulates different body positions)
    dc = rng.normal(0, 0.03, (1, c.shape[1])).astype(np.float32)
    c += dc

    # Gaussian noise
    c += rng.normal(0, 0.02, c.shape).astype(np.float32)

    return c


def _build_synthetic_sequence(
    cycles: list[np.ndarray],
    rep_plan: list[tuple[int, float]],   # (cycle_idx, tempo_variation) per rep
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Stitch reps into one continuous sequence using a pre-computed plan so that
    ARM and FOOT streams always share the same per-rep tempo.
    """
    parts = []
    for cycle_idx, variation in rep_plan:
        cycle = cycles[cycle_idx % len(cycles)]
        new_len = max(5, int(len(cycle) * variation))
        resampled = _resample_cycle(cycle, new_len)
        augmented = _augment_cycle(resampled, rng)
        parts.append(augmented)

    return np.concatenate(parts, axis=0)


def _make_timestamps(n: int, start: datetime, dt_ms: float = 1000.0 / FS) -> list[str]:
    ts = []
    t = start
    for _ in range(n):
        ts.append(t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z")
        t += timedelta(milliseconds=dt_ms)
    return ts


# ──────────────────────────────────────────────────────────────────────────────
# RAUSCHEN SYNTHESIS (rest / noise – no periodicity needed)
# ──────────────────────────────────────────────────────────────────────────────

def _synthesise_rauschen(
    real_arm: np.ndarray,
    real_foot: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic resting data by randomly sampling stretches of the
    real Rauschen data and adding slight jitter.
    """
    def _sample(real: np.ndarray) -> np.ndarray:
        out = []
        while len(out) < n_samples:
            start = rng.integers(0, max(1, len(real) - 50))
            chunk = real[start: start + rng.integers(20, 80)]
            chunk = chunk + rng.normal(0, 0.01, chunk.shape).astype(np.float32)
            out.append(chunk)
        return np.concatenate(out)[:n_samples].astype(np.float32)

    return _sample(real_arm.astype(np.float32)), _sample(real_foot.astype(np.float32))


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def generate(
    data_dir: str,
    out_dir: str,
    n_recordings: int,
    min_reps: int,
    max_reps: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    in_p  = Path(data_dir)
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(in_p.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {data_dir}")

    # ── 1.  Load real data and build cycle library ────────────────────────────
    # exercise → band → list of cycles (each cycle is [T, N_FEAT])
    cycle_library: dict[str, dict[str, list[np.ndarray]]] = {}
    # exercise → band → raw array  (for Rauschen)
    raw_library:   dict[str, dict[str, np.ndarray]]       = {}
    # real calibration values per band (we'll reuse them)
    cal_library:   dict[str, dict[str, np.ndarray]]       = {}

    CAL_COLS = ["sysCal", "gyroCal", "accelCal", "magCal"]

    for f in csv_files:
        df = pd.read_csv(f)
        df["band"]  = df["band"].astype(str).str.upper().str.strip()
        df["label"] = df["label"].astype(str).str.strip()
        for col in FEATURES:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=FEATURES)

        label = df["label"].mode()[0]
        if label not in cycle_library:
            cycle_library[label] = {}
            raw_library[label]   = {}
            cal_library[label]   = {}

        feat_idx = DOMINANT_FEAT.get(label, DEFAULT_DOMINANT)

        for band in ("ARM", "FOOT"):
            bdf = df[df["band"] == band].reset_index(drop=True)
            if len(bdf) < 50:
                continue

            data = bdf[FEATURES].to_numpy(dtype=np.float32)
            raw_library[label].setdefault(band, [])
            raw_library[label][band] = (
                np.concatenate([raw_library[label][band], data], axis=0)
                if len(raw_library[label][band]) > 0 else data
            )

            # calibration: keep typical values
            cal_data = np.ones((len(bdf), 4), dtype=np.float32) * 3.0
            for i, c in enumerate(CAL_COLS):
                if c in bdf.columns:
                    cal_data[:, i] = pd.to_numeric(bdf[c], errors="coerce").fillna(3).to_numpy()
            cal_library[label].setdefault(band, cal_data)

            if label == "Rauschen":
                continue   # no cycle extraction for rest

            cycles = _extract_cycles(data, feat_idx)
            if cycles:
                existing = cycle_library[label].get(band, [])
                cycle_library[label][band] = existing + cycles
                print(f"  {label:20s} {band}  {len(cycles):3d} cycles extracted from {f.name}")

    # ── 2.  Generate synthetic recordings per exercise ────────────────────────
    generated = 0
    start_time = datetime(2026, 3, 1, 22, 0, 0, tzinfo=timezone.utc)

    for label in cycle_library:
        arm_cycles  = cycle_library[label].get("ARM",  [])
        foot_cycles = cycle_library[label].get("FOOT", [])

        if label == "Rauschen":
            arm_raw  = raw_library[label].get("ARM",  None)
            foot_raw = raw_library[label].get("FOOT", None)
            if arm_raw is None or foot_raw is None:
                print(f"  Skipping Rauschen (missing ARM or FOOT data)")
                continue

            for i in range(n_recordings):
                n_samples = rng.integers(400, 800)
                arm_seq, foot_seq = _synthesise_rauschen(arm_raw, foot_raw, int(n_samples), rng)
                rows = _interleave_to_df(arm_seq, foot_seq, label, start_time, rng, cal_library.get(label, {}))
                fname = out_p / f"syn_{label.replace(' ','_')}_{i:03d}.csv"
                rows.to_csv(fname, index=False)
                generated += 1

            print(f"  {label}: {n_recordings} synthetic recordings")
            continue

        if not arm_cycles or not foot_cycles:
            print(f"  Skipping {label} – not enough cycles (arm={len(arm_cycles)} foot={len(foot_cycles)})")
            continue

        print(f"  {label}: {len(arm_cycles)} arm cycles, {len(foot_cycles)} foot cycles → generating {n_recordings} recordings …")

        for i in range(n_recordings):
            n_reps  = int(rng.integers(min_reps, max_reps + 1))
            tempo_v = rng.uniform(0.05, 0.18)

            # Build one rep plan: (cycle_index, tempo_variation) per rep.
            # Both ARM and FOOT will use the same per-rep tempo so their
            # timing stays correlated, matching real dual-sensor behaviour.
            arm_plan  = [(int(rng.integers(len(arm_cycles))),
                          float(rng.uniform(1.0 - tempo_v, 1.0 + tempo_v)))
                         for _ in range(n_reps)]
            # FOOT uses the same tempo values, but maps to its own cycle library
            foot_plan = [(idx % len(foot_cycles), var) for idx, var in arm_plan]

            arm_seq  = _build_synthetic_sequence(arm_cycles,  arm_plan,  rng)
            foot_seq = _build_synthetic_sequence(foot_cycles, foot_plan, rng)

            rows = _interleave_to_df(arm_seq, foot_seq, label, start_time, rng, cal_library.get(label, {}))
            fname = out_p / f"syn_{label.replace(' ','_')}_{i:03d}.csv"
            rows.to_csv(fname, index=False)
            generated += 1

    print(f"\nDone. {generated} synthetic CSV files written to {out_p.resolve()}")


def _interleave_to_df(
    arm_seq:   np.ndarray,    # [N_arm,  N_FEAT]
    foot_seq:  np.ndarray,    # [N_foot, N_FEAT]
    label:     str,
    start:     datetime,
    rng:       np.random.Generator,
    cal:       dict[str, np.ndarray],
) -> pd.DataFrame:
    """Merge ARM + FOOT samples into a single interleaved DataFrame."""
    dt_arm  = 1000.0 / FS
    dt_foot = 1000.0 / FS

    rows = []
    # Millis values separate for each band (independent sensor clocks)
    millis_base_arm  = int(rng.integers(100_000, 500_000))
    millis_base_foot = int(rng.integers(100_000, 500_000))

    cal_arm  = cal.get("ARM",  np.ones((1, 4), dtype=np.float32) * 3.0)
    cal_foot = cal.get("FOOT", np.ones((1, 4), dtype=np.float32) * 3.0)

    def _sample_cal(arr, idx):
        row = arr[idx % len(arr)]
        return [int(row[j]) for j in range(4)]

    # Create ARM rows
    t_arm = start
    for k in range(len(arm_seq)):
        ts = t_arm.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t_arm.microsecond // 1000:03d}Z"
        cal_vals = _sample_cal(cal_arm, k)
        row = {
            "timestamp": ts,
            "band": "ARM",
            "millis": millis_base_arm + int(k * dt_arm),
        }
        for j, feat in enumerate(FEATURES):
            row[feat] = round(float(arm_seq[k, j]), 6)
        row.update(dict(zip(["sysCal","gyroCal","accelCal","magCal"], cal_vals)))
        row["label"] = label
        rows.append(row)
        t_arm += timedelta(milliseconds=dt_arm)

    # Create FOOT rows  (offset start by half a period to interleave naturally)
    t_foot = start + timedelta(milliseconds=dt_foot / 2)
    for k in range(len(foot_seq)):
        ts = t_foot.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t_foot.microsecond // 1000:03d}Z"
        cal_vals = _sample_cal(cal_foot, k)
        row = {
            "timestamp": ts,
            "band": "FOOT",
            "millis": millis_base_foot + int(k * dt_foot),
        }
        for j, feat in enumerate(FEATURES):
            row[feat] = round(float(foot_seq[k, j]), 6)
        row.update(dict(zip(["sysCal","gyroCal","accelCal","magCal"], cal_vals)))
        row["label"] = label
        rows.append(row)
        t_foot += timedelta(milliseconds=dt_foot)

    df = pd.DataFrame(rows)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp_dt").drop(columns="timestamp_dt").reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic periodic IMU exercise data")
    parser.add_argument("--data_dir",     default=".",         help="Directory with real CSV files")
    parser.add_argument("--out_dir",      default="synthetic", help="Output directory for synthetic CSVs")
    parser.add_argument("--n_recordings", type=int, default=20, help="Synthetic recordings per exercise")
    parser.add_argument("--min_reps",     type=int, default=8,  help="Min reps per synthetic recording")
    parser.add_argument("--max_reps",     type=int, default=25, help="Max reps per synthetic recording")
    parser.add_argument("--seed",         type=int, default=0)
    args = parser.parse_args()

    generate(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        n_recordings=args.n_recordings,
        min_reps=args.min_reps,
        max_reps=args.max_reps,
        seed=args.seed,
    )
