"""
Create Workout CSV
==================
Stitches real IMU exercise recordings into a single "complete workout" CSV
by interleaving exercise blocks with configurable rest (Rauschen) periods.

The output CSV has the same format as a live recording and can be fed
directly into plot_workout.py or live_inference.py --replay.

Usage:
    # Default: 1 set of each exercise, 30 s rest
    python create_workout.py

    # 2 sets of each exercise, 45 s rest
    python create_workout.py --sets 2 --rest 45

    # Custom plan as JSON list of [exercise, n_sets] pairs
    python create_workout.py --plan '[["Push ups",2],["Kniebeuge",2],["Sit Up",1]]'

    # Custom output path
    python create_workout.py --out my_workout.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────

FS_PER_BAND = 50.0   # approximate sample rate per band (Hz)

# Preferred workout order when auto-building the default plan
NATURAL_ORDER = [
    "Hampelmanner",
    "Push ups",
    "Kniebeuge",
    "Sit Up",
    "Montain climbers",
]


# ──────────────────────────────────────────────────────────────────────────────
# LIBRARY LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_library(data_dir: Path) -> dict[str, list[pd.DataFrame]]:
    """
    Load all real (non-synthetic) CSV files from data_dir.
    Returns: {label: [df, ...]}  (Rauschen included separately for rest blocks)
    """
    library: dict[str, list[pd.DataFrame]] = {}
    for f in sorted(data_dir.glob("*.csv")):
        if f.stem.startswith("syn_") or f.stem.startswith("workout"):
            continue
        df = pd.read_csv(f)
        if "label" not in df.columns or "band" not in df.columns:
            continue
        df["band"]  = df["band"].astype(str).str.upper().str.strip()
        df["label"] = df["label"].astype(str).str.strip()
        label = df["label"].mode()[0]
        if label in ("?", ""):
            continue
        library.setdefault(label, []).append(df)
        dur = _estimate_duration(df)
        print(f"  {f.name:45s}  →  {label}  ({dur:.0f} s)")
    return library


def _estimate_duration(df: pd.DataFrame) -> float:
    """Return approximate recording duration in seconds."""
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dropna()
    if len(ts) < 2:
        return len(df) / (FS_PER_BAND * 2)
    return (ts.iloc[-1] - ts.iloc[0]).total_seconds()


# ──────────────────────────────────────────────────────────────────────────────
# TIMESTAMP MANIPULATION
# ──────────────────────────────────────────────────────────────────────────────

def _retimestamp(df: pd.DataFrame, new_start: datetime) -> pd.DataFrame:
    """
    Shift all timestamps in df so the earliest sample aligns with new_start.
    """
    df = df.copy()
    parsed = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    valid  = parsed.dropna()
    if valid.empty:
        return df

    t0    = valid.iloc[0]
    delta = new_start - t0

    def _shift(ts_str: str) -> str:
        try:
            t = pd.to_datetime(ts_str, utc=True) + delta
            return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"
        except Exception:
            return ts_str

    df["timestamp"] = df["timestamp"].apply(_shift)
    return df


def _block_end(df: pd.DataFrame) -> datetime:
    """
    Return the datetime one sample-period after the last sample in df.
    This gives the start time for the very next block.
    """
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dropna()
    if ts.empty:
        return datetime.now(timezone.utc)
    one_sample = timedelta(milliseconds=1000.0 / FS_PER_BAND)
    return ts.iloc[-1] + one_sample


# ──────────────────────────────────────────────────────────────────────────────
# REST BLOCK BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def _build_rest_block(rauschen_pool: pd.DataFrame,
                      duration_s: float,
                      start: datetime) -> pd.DataFrame:
    """
    Tile real Rauschen data to fill exactly duration_s seconds.
    Returns a re-timestamped block with label="Rauschen".
    """
    if rauschen_pool.empty or duration_s <= 0:
        return pd.DataFrame()

    rows_needed = max(1, int(duration_s * FS_PER_BAND * 2))   # *2: ARM + FOOT
    repeats     = rows_needed // len(rauschen_pool) + 2
    block       = pd.concat([rauschen_pool] * repeats, ignore_index=True)
    block       = block.iloc[:rows_needed].copy()
    block["label"] = "Rauschen"
    return _retimestamp(block, start)


# ──────────────────────────────────────────────────────────────────────────────
# WORKOUT ASSEMBLY
# ──────────────────────────────────────────────────────────────────────────────

def build_workout(library:  dict[str, list[pd.DataFrame]],
                  plan:     list[tuple[str, int]],
                  rest_s:   float) -> pd.DataFrame:
    """
    Build the complete workout DataFrame.

    Each (exercise, n_sets) entry in plan produces n_sets blocks.
    Every block (including the last one) is followed by a Rauschen rest block.
    """
    # Pool all Rauschen recordings for use as rest data
    rauschen_dfs = library.get("Rauschen", [])
    if rauschen_dfs:
        rauschen_pool = pd.concat(rauschen_dfs, ignore_index=True)
        rauschen_pool["label"] = "Rauschen"
    else:
        print("  WARNING: keine Rauschen-Aufnahme gefunden – Pausen werden übersprungen")
        rauschen_pool = pd.DataFrame()

    cursor = datetime(2026, 3, 1, 22, 0, 0, tzinfo=timezone.utc)
    blocks: list[pd.DataFrame] = []

    for ex_label, n_sets in plan:
        recs = library.get(ex_label, [])
        if not recs:
            print(f"  WARNING: keine Aufnahmen für '{ex_label}' – übersprungen")
            continue

        for set_idx in range(n_sets):
            # Cycle through available recordings if n_sets > len(recs)
            rec   = recs[set_idx % len(recs)].copy()
            block = _retimestamp(rec, cursor)
            block["label"] = ex_label
            blocks.append(block)

            cursor = _block_end(block)
            dur    = _estimate_duration(block)
            print(f"  + Satz {set_idx + 1} – {ex_label:<22}  "
                  f"({dur:.0f}s / {len(block):,} Zeilen)  → {cursor.strftime('%H:%M:%S')} UTC")

            # Always add rest after every set
            rest = _build_rest_block(rauschen_pool, rest_s, cursor)
            if not rest.empty:
                blocks.append(rest)
                cursor = _block_end(rest)

    if not blocks:
        raise ValueError("Workout ist leer – keine passenden Daten gefunden.")

    workout = pd.concat(blocks, ignore_index=True)

    # Re-index millis per band to be workout-relative (ascending)
    dt_ms = 1000.0 / FS_PER_BAND
    for band in ("ARM", "FOOT"):
        mask = workout["band"] == band
        n    = int(mask.sum())
        workout.loc[mask, "millis"] = (np.arange(n) * dt_ms + 100_000).astype(int)

    # Final sort by timestamp
    workout["_ts"] = pd.to_datetime(workout["timestamp"], errors="coerce", utc=True)
    workout = (workout
               .sort_values("_ts")
               .drop(columns="_ts")
               .reset_index(drop=True))

    return workout


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main(data_dir: str,
         plan_json: str | None,
         sets_per_exercise: int,
         rest_s: float,
         out: str):

    in_p = Path(data_dir)
    print("\n[1/3] Übungsaufnahmen laden …")
    library = load_library(in_p)

    available = sorted(k for k in library if k != "Rauschen")
    print(f"\n  Verfügbare Übungen: {available}")

    # ── Build workout plan ────────────────────────────────────────────────────
    if plan_json:
        raw  = json.loads(plan_json)
        plan = [(entry[0], int(entry[1])) for entry in raw]
    else:
        ordered   = [ex for ex in NATURAL_ORDER if ex in available]
        remainder = [ex for ex in available if ex not in ordered]
        plan      = [(ex, sets_per_exercise) for ex in ordered + remainder]

    print("\n[2/3] Workout-Plan:")
    total_sets = sum(s for _, s in plan)
    for ex, sets in plan:
        n_recs = len(library.get(ex, []))
        print(f"  {sets}× {ex}  (aus {n_recs} Aufnahme{'n' if n_recs != 1 else ''})")
    estimated_duration = sum(
        _estimate_duration(library[ex][0]) * sets
        for ex, sets in plan if ex in library
    ) + total_sets * rest_s
    print(f"  Pause: {rest_s:.0f}s nach jedem Satz")
    print(f"  Geschätzte Gesamtdauer: ~{estimated_duration:.0f}s ({estimated_duration/60:.1f} min)")

    print("\n[3/3] Workout zusammenstellen …")
    workout = build_workout(library, plan, rest_s)

    out_p = Path(out)
    workout.to_csv(out_p, index=False)

    n_exercise = int((workout["label"] != "Rauschen").sum())
    n_rest     = int((workout["label"] == "Rauschen").sum())
    total_s    = n_exercise / (FS_PER_BAND * 2)
    total_rest = n_rest     / (FS_PER_BAND * 2)

    print(f"\nFertig. Workout gespeichert: {out_p.resolve()}")
    print(f"  {len(workout):,} Zeilen gesamt")
    print(f"  {n_exercise:,} Zeilen Übung   (~{total_s:.0f}s)")
    print(f"  {n_rest:,} Zeilen Rauschen (~{total_rest:.0f}s)")
    print(f"  → plot_workout.py --workout {out_p.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine real IMU exercise recordings into a complete workout CSV"
    )
    parser.add_argument("--data_dir", default=".",
                        help="Verzeichnis mit echten CSV-Dateien")
    parser.add_argument("--plan",     default=None,
                        help='Workout-Plan als JSON, z.B. \'[["Push ups",2],["Kniebeuge",2]]\'')
    parser.add_argument("--sets",     type=int, default=1,
                        help="Sätze pro Übung (wenn --plan nicht angegeben)")
    parser.add_argument("--rest",     type=float, default=30.0,
                        help="Pausendauer zwischen Sätzen in Sekunden (default: 30)")
    parser.add_argument("--out",      default="workout.csv",
                        help="Ausgabe-CSV-Pfad (default: workout.csv)")
    args = parser.parse_args()

    main(args.data_dir, args.plan, args.sets, args.rest, args.out)
