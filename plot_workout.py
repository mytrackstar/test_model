"""
Workout Inference Plot
======================
Feeds a combined workout CSV (from create_workout.py) through LiveSession,
automatically detects exercise sets and rep counts per set, and produces
three visualisation plots.

Plots saved to --out_dir:
  01_workout_timeline   – Full workout as a coloured timeline + confidence
  02_set_summary        – Rep counts per exercise per set (bar chart)
  03_rep_progression    – Rep counter with set boundaries + annotations

Usage:
    # Build workout first (if not yet done):
    python create_workout.py --sets 2 --rest 30 --out workout.csv

    # Then plot:
    python plot_workout.py --workout workout.csv
    python plot_workout.py --workout workout.csv --min_rest 8 --threshold 0.65
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).parent))
from live_inference import LiveSession

# ──────────────────────────────────────────────────────────────────────────────

EXERCISE_COLORS = {
    "Push ups":         "#e74c3c",
    "Kniebeuge":        "#3498db",
    "Hampelmanner":     "#2ecc71",
    "Sit Up":           "#f39c12",
    "Montain climbers": "#9b59b6",
    "Rauschen":         "#4a4a5a",
}
DEFAULT_COLOR = "#606070"

STYLE = {
    "figure.facecolor":  "#1a1a2e",
    "axes.facecolor":    "#16213e",
    "axes.edgecolor":    "#0f3460",
    "axes.labelcolor":   "#e0e0e0",
    "xtick.color":       "#a0a0a0",
    "ytick.color":       "#a0a0a0",
    "text.color":        "#e0e0e0",
    "grid.color":        "#0f3460",
    "grid.linewidth":    0.6,
    "axes.grid":         True,
    "axes.titlesize":    11,
    "axes.labelsize":    9,
    "legend.fontsize":   8,
    "font.family":       "monospace",
}

INFERENCE_DT = 20 / 50.0   # 0.4 s per inference stride


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(csv_path: Path,
                  artifacts_dir: Path,
                  threshold: float) -> pd.DataFrame:
    """
    Feed every row of csv_path through LiveSession.
    Returns a DataFrame with one row per inference firing:
      t, exercise, confidence, reps
    """
    df      = pd.read_csv(csv_path)
    session = LiveSession(artifacts_dir=str(artifacts_dir), confidence_threshold=threshold)

    results = []
    t       = 0.0

    for _, row in df.iterrows():
        res = session.push_row(row.to_dict())
        if res is None:
            continue
        entry = {"t": t, "exercise": res["exercise"],
                 "confidence": res["confidence"], "reps": res["reps"]}
        results.append(entry)
        t += INFERENCE_DT

    if not results:
        raise RuntimeError(f"Keine Inferenz-Ergebnisse für {csv_path.name}")

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────────────
# SET DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def detect_sets(res: pd.DataFrame,
                min_rest_s: float = 8.0,
                min_set_s: float  = 8.0) -> list[dict]:
    """
    Segment the inference timeline into exercise sets.

    Two blocks of the same exercise separated by a Rauschen gap >= min_rest_s
    are treated as different sets.  Short gaps (< min_rest_s) are treated as
    within-set classification noise and merged into the surrounding exercise.

    For rep counting:
      - Different exercises → RepCounter resets → start_reps = 0
      - Same exercise across rest → RepCounter accumulates → delta approach:
            reps_this_set = end_reps − start_reps

    Returns list of dicts:
      exercise, set_num, start_t, end_t, duration, start_reps, end_reps, reps
    """
    times    = res["t"].values
    exs      = res["exercise"].values
    reps_arr = res["reps"].values

    # ── Step 1: build raw contiguous blocks ──────────────────────────────────
    raw: list[dict] = []
    i = 0
    while i < len(exs):
        ex = exs[i]
        j  = i + 1
        while j < len(exs) and exs[j] == ex:
            j += 1
        raw.append(dict(exercise=ex,
                        start_i=i, end_i=j - 1,
                        start_t=float(times[i]), end_t=float(times[j - 1]),
                        duration=float(times[j - 1] - times[i])))
        i = j

    # ── Step 2: merge short Rauschen blips back into surrounding exercise ──
    # A Rauschen block shorter than min_rest_s that is surrounded by the same
    # exercise on both sides is considered a momentary misclassification.
    merged: list[dict] = []
    for b in raw:
        if (b["exercise"] == "Rauschen"
                and b["duration"] < min_rest_s
                and len(merged) > 0
                and merged[-1]["exercise"] != "Rauschen"):
            # Absorb into previous block
            merged[-1]["end_i"] = b["end_i"]
            merged[-1]["end_t"] = b["end_t"]
            merged[-1]["duration"] = merged[-1]["end_t"] - merged[-1]["start_t"]
        elif (len(merged) > 0
              and merged[-1]["exercise"] == b["exercise"]
              and b["exercise"] != "Rauschen"):
            # Extend previous exercise block (happens after blip absorption)
            merged[-1]["end_i"] = b["end_i"]
            merged[-1]["end_t"] = b["end_t"]
            merged[-1]["duration"] = merged[-1]["end_t"] - merged[-1]["start_t"]
        else:
            merged.append(dict(b))

    # ── Step 3: assign set numbers and rep counts ─────────────────────────────
    exercise_counters: dict[str, int] = {}
    sets: list[dict] = []

    for b in merged:
        if b["exercise"] == "Rauschen":
            continue
        if b["duration"] < min_set_s:
            continue

        ex  = b["exercise"]
        exercise_counters[ex] = exercise_counters.get(ex, 0) + 1
        set_num = exercise_counters[ex]

        start_reps = int(reps_arr[b["start_i"]])
        end_reps   = int(reps_arr[b["end_i"]])
        reps_count = max(0, end_reps - start_reps)

        sets.append(dict(
            exercise=ex,
            set_num=set_num,
            start_t=b["start_t"],
            end_t=b["end_t"],
            duration=b["duration"],
            start_reps=start_reps,
            end_reps=end_reps,
            reps=reps_count,
        ))

    return sets


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 1 – WORKOUT TIMELINE
# ──────────────────────────────────────────────────────────────────────────────

def plot_workout_timeline(res: pd.DataFrame,
                          sets: list[dict],
                          out: Path):
    """
    Top panel  – confidence over time (coloured by predicted exercise)
    Middle     – exercise colour bar (imshow-style) for quick visual scan
    Bottom     – detected sets as annotated coloured spans
    """
    plt.rcParams.update(STYLE)

    t    = res["t"].values / 60.0          # convert to minutes
    conf = res["confidence"].values
    exs  = res["exercise"].values

    fig, axes = plt.subplots(3, 1, figsize=(16, 8),
                             gridspec_kw={"height_ratios": [3, 0.6, 2]})
    fig.suptitle("Workout Timeline – KI Inferenz", fontsize=13, y=1.01)

    # ── Confidence line, coloured by exercise ────────────────────────────────
    ax = axes[0]
    for i in range(len(t) - 1):
        col = EXERCISE_COLORS.get(exs[i], DEFAULT_COLOR)
        ax.plot(t[i:i + 2], conf[i:i + 2], color=col, linewidth=1.2, alpha=0.9)
    ax.set_ylabel("Konfidenz")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.65, color="#ffffff", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.set_xlim(t[0], t[-1])
    ax.set_xticklabels([])

    # Set boundaries on confidence panel
    for s in sets:
        ax.axvline(s["start_t"] / 60.0, color="#ffffff", linewidth=0.7,
                   linestyle=":", alpha=0.5)

    # ── Exercise colour bar ───────────────────────────────────────────────────
    ax2 = axes[1]
    for i in range(len(t) - 1):
        col = EXERCISE_COLORS.get(exs[i], DEFAULT_COLOR)
        ax2.axvspan(t[i], t[i + 1], color=col, alpha=0.9)
    ax2.set_yticks([])
    ax2.set_ylabel("Übung", fontsize=7)
    ax2.set_xlim(t[0], t[-1])
    ax2.set_xticklabels([])
    ax2.grid(False)

    # ── Detected sets as annotated spans ─────────────────────────────────────
    ax3 = axes[2]
    ax3.set_xlim(t[0], t[-1])
    ax3.set_ylim(0, 1)
    ax3.set_yticks([])
    ax3.set_xlabel("Zeit (min)")
    ax3.set_ylabel("Sätze")
    ax3.grid(False)

    y_pos = 0.5
    bar_height = 0.6

    for s in sets:
        x0  = s["start_t"] / 60.0
        x1  = s["end_t"]   / 60.0
        col = EXERCISE_COLORS.get(s["exercise"], DEFAULT_COLOR)
        ax3.barh(y_pos, x1 - x0, left=x0, height=bar_height,
                 color=col, alpha=0.85, edgecolor="#ffffff", linewidth=0.5)
        mid = (x0 + x1) / 2.0
        label = f"{s['exercise'][:4]}×{s['set_num']}\n{s['reps']} rep"
        ax3.text(mid, y_pos, label, ha="center", va="center",
                 fontsize=6.5, color="white", fontweight="bold")

    # ── Legend ────────────────────────────────────────────────────────────────
    patches = [mpatches.Patch(color=v, label=k)
               for k, v in EXERCISE_COLORS.items() if k != "Rauschen"]
    axes[0].legend(handles=patches, loc="upper right",
                   ncol=3, framealpha=0.3, fontsize=7)

    fig.tight_layout()
    _save(fig, out)


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 2 – SET SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

def plot_set_summary(sets: list[dict], out: Path):
    """
    Grouped horizontal bar chart: reps per set, grouped by exercise.
    Also shows total reps per exercise.
    """
    if not sets:
        print("  (keine Sätze erkannt – kein Summary-Plot)")
        return

    plt.rcParams.update(STYLE)

    # Build labels and values
    labels = [f"{s['exercise']}  Satz {s['set_num']}" for s in sets]
    reps   = [s["reps"] for s in sets]
    colors = [EXERCISE_COLORS.get(s["exercise"], DEFAULT_COLOR) for s in sets]

    fig, ax = plt.subplots(figsize=(10, max(4, len(sets) * 0.55 + 1.5)))
    fig.suptitle("Workout Ergebnis – Reps pro Satz", fontsize=13)

    y_pos = np.arange(len(labels))
    bars  = ax.barh(y_pos, reps, color=colors, alpha=0.85,
                    edgecolor="#ffffff", linewidth=0.4, height=0.6)

    # Annotate bars with rep count
    for bar, rep in zip(bars, reps):
        if rep > 0:
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    str(rep), va="center", ha="left", fontsize=9, color="#e0e0e0")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Wiederholungen")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Total reps per exercise in title or margin
    totals: dict[str, int] = {}
    for s in sets:
        totals[s["exercise"]] = totals.get(s["exercise"], 0) + s["reps"]
    summary_lines = [f"{ex}: {n} total" for ex, n in totals.items()]
    ax.set_title("\n".join(summary_lines), fontsize=8, loc="right", color="#a0a0a0")

    fig.tight_layout()
    _save(fig, out)


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 3 – REP PROGRESSION
# ──────────────────────────────────────────────────────────────────────────────

def plot_rep_progression(res: pd.DataFrame,
                         sets: list[dict],
                         out: Path):
    """
    Step function of accumulated rep counts.
    Each exercise is shown in its own colour segment.
    Vertical dashed lines mark set boundaries with annotations.
    """
    plt.rcParams.update(STYLE)

    t    = res["t"].values / 60.0
    reps = res["reps"].values
    exs  = res["exercise"].values

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.suptitle("Workout Rep-Verlauf", fontsize=13)

    # Coloured step line per exercise
    for i in range(len(t) - 1):
        col = EXERCISE_COLORS.get(exs[i], DEFAULT_COLOR)
        ax.step([t[i], t[i + 1]], [reps[i], reps[i]], where="post",
                color=col, linewidth=2.0, alpha=0.85)

    # Set boundary lines + annotations
    for s in sets:
        x0 = s["start_t"] / 60.0
        x1 = s["end_t"]   / 60.0
        col = EXERCISE_COLORS.get(s["exercise"], DEFAULT_COLOR)

        ax.axvline(x0, color=col, linewidth=0.8, linestyle="--", alpha=0.7)

        # Label at top of the start line
        ax.text(x0 + 0.02, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 1,
                f"{s['exercise'][:6]}\nS{s['set_num']} +{s['reps']}",
                va="top", ha="left", fontsize=6.5, color=col, alpha=0.9)

    ax.set_xlabel("Zeit (min)")
    ax.set_ylabel("Kumulative Reps")
    ax.set_xlim(t[0], t[-1])
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    _save(fig, out)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {path.name}")


def _print_workout_table(sets: list[dict]):
    """Print a structured workout summary to stdout."""
    if not sets:
        print("  Keine Sätze erkannt.")
        return

    print()
    print(f"  {'Übung':<22} {'Satz':>5} {'Reps':>6} {'Dauer':>8}")
    print("  " + "─" * 45)

    totals: dict[str, int] = {}
    for s in sets:
        print(f"  {s['exercise']:<22} {s['set_num']:>5} {s['reps']:>6} "
              f"  {s['duration']:.0f}s")
        totals[s["exercise"]] = totals.get(s["exercise"], 0) + s["reps"]

    print("  " + "─" * 45)
    for ex, total in totals.items():
        n_sets = sum(1 for s in sets if s["exercise"] == ex)
        print(f"  {ex:<22} {n_sets:>2} Sätze  {total:>4} Reps gesamt")

    print(f"\n  Gesamt: {sum(totals.values())} Reps in "
          f"{len(sets)} Sätzen über {len(totals)} Übungen")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main(workout_csv: str,
         artifacts_dir: str,
         out_dir: str,
         threshold: float,
         min_rest_s: float):

    csv_p  = Path(workout_csv)
    art_p  = Path(artifacts_dir)
    out_p  = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    if not csv_p.exists():
        print(f"Fehler: {csv_p} nicht gefunden.")
        print("  → Erstelle zunächst ein Workout mit: python create_workout.py")
        sys.exit(1)

    print(f"\n[1/3] Inferenz läuft auf '{csv_p.name}' …")
    res = run_inference(csv_p, art_p, threshold)
    total_t = res["t"].iloc[-1] / 60.0
    print(f"  {len(res)} Inferenz-Schritte · {total_t:.1f} min")

    print(f"\n[2/3] Sätze erkennen (min_rest={min_rest_s}s) …")
    sets = detect_sets(res, min_rest_s=min_rest_s)
    _print_workout_table(sets)

    print(f"\n[3/3] Plots erstellen → {out_p}/")
    stem = csv_p.stem

    plot_workout_timeline(res, sets, out_p / f"{stem}_01_timeline.png")
    plot_set_summary     (sets,      out_p / f"{stem}_02_set_summary.png")
    plot_rep_progression (res, sets, out_p / f"{stem}_03_rep_progression.png")

    print(f"\nFertig. Plots gespeichert in: {out_p.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference + Set/Rep-Erkennung für ein komplettes Workout"
    )
    parser.add_argument("--workout",    default="workout.csv",
                        help="Workout-CSV Pfad (erzeugt mit create_workout.py)")
    parser.add_argument("--artifacts",  default="artifacts",
                        help="Ordner mit Modell-Artifacts")
    parser.add_argument("--out_dir",    default="workout_plots",
                        help="Ausgabeordner für Plots")
    parser.add_argument("--threshold",  type=float, default=0.65,
                        help="Konfidenz-Schwellwert für LiveSession")
    parser.add_argument("--min_rest",   type=float, default=8.0,
                        help="Mindest-Pausendauer (s) um Satzgrenzen zu erkennen")
    args = parser.parse_args()

    main(args.workout, args.artifacts, args.out_dir, args.threshold, args.min_rest)
