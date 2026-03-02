"""
Inference Plots – Dual-IMU Model Output Visualisation
======================================================
Runs LiveSession on every real CSV file and produces three plots per file:

  01_probabilities  – Softmax probability for each class over time
  02_timeline       – Predicted exercise as a coloured band + confidence
  03_rep_counter    – Rep counter step-function over time

Plus one global summary:
  00_inference_summary – Side-by-side final rep counts vs. inference timeline

Usage:
    python plot_inference.py [--data_dir ./] [--artifacts ./artifacts]
                             [--out_dir ./inference_plots] [--threshold 0.65]
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

# Import LiveSession from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from live_inference import LiveSession

# ──────────────────────────────────────────────────────────────────────────────

EXERCISE_COLORS = {
    "Push ups":         "#e74c3c",
    "Kniebeuge":        "#3498db",
    "Hampelmanner":     "#2ecc71",
    "Sit Up":           "#f39c12",
    "Montain climbers": "#9b59b6",
    "Rauschen":         "#95a5a6",
}
DEFAULT_COLOR = "#e0e0e0"

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

# Approx time between inference calls: stride=20 @ FS=50 Hz
INFERENCE_DT = 20 / 50.0   # 0.4 s


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _apply_style():
    plt.rcParams.update(STYLE)


def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {path.name}")


def run_inference(csv_path: Path, artifacts_dir: str, threshold: float):
    """
    Feed a CSV file row-by-row through LiveSession and collect all
    inference results as a DataFrame.
    Returns (results_df, true_label).
    """
    df = pd.read_csv(csv_path)
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip()

    session = LiveSession(artifacts_dir=artifacts_dir, confidence_threshold=threshold)

    records = []
    step = 0

    for _, row in df.iterrows():
        result = session.push_row(row.to_dict())
        if result is None:
            continue

        t = step * INFERENCE_DT
        rec = {
            "t":          t,
            "exercise":   result["exercise"],
            "confidence": result["confidence"],
            "reps":       result["reps"],
        }
        for cls, prob in result["probs"].items():
            rec[f"p_{cls}"] = prob
        records.append(rec)
        step += 1

    true_label = df["label"].mode()[0] if "label" in df.columns else "?"
    return pd.DataFrame(records), true_label


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 1: SOFTMAX PROBABILITIES over time
# ──────────────────────────────────────────────────────────────────────────────

def plot_probabilities(res: pd.DataFrame, true_label: str, classes: list, out: Path):
    _apply_style()
    true_color = EXERCISE_COLORS.get(true_label, DEFAULT_COLOR)

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(f"Softmax Wahrscheinlichkeiten  ·  {true_label}",
                 fontsize=13, color=true_color, fontweight="bold")

    t = res["t"].values
    for cls in classes:
        col = f"p_{cls}"
        if col not in res.columns:
            continue
        c = EXERCISE_COLORS.get(cls, DEFAULT_COLOR)
        lw = 2.0 if cls == true_label else 0.9
        alpha = 1.0 if cls == true_label else 0.55
        ax.plot(t, res[col].values, color=c, linewidth=lw,
                alpha=alpha, label=cls)

    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("Wahrscheinlichkeit")
    ax.set_xlabel("Zeit (s)")
    ax.legend(loc="upper right", framealpha=0.3, ncol=3)

    # Mark true label with horizontal dashed line at y=1
    ax.axhline(1.0, color=true_color, linewidth=0.5, linestyle="--", alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out)


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 2: PREDICTED EXERCISE TIMELINE + CONFIDENCE
# ──────────────────────────────────────────────────────────────────────────────

def plot_timeline(res: pd.DataFrame, true_label: str, out: Path):
    _apply_style()
    true_color = EXERCISE_COLORS.get(true_label, DEFAULT_COLOR)

    fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(f"Klassifikations-Timeline  ·  {true_label}",
                 fontsize=13, color=true_color, fontweight="bold")

    t = res["t"].values

    # ── Top: confidence line, background coloured by predicted exercise ───────
    ax = axes[0]
    exercises = res["exercise"].values

    # Draw coloured background segments
    prev_ex  = exercises[0]
    seg_start = t[0]
    for i in range(1, len(t)):
        if exercises[i] != prev_ex or i == len(t) - 1:
            c = EXERCISE_COLORS.get(prev_ex, DEFAULT_COLOR)
            ax.axvspan(seg_start, t[i], alpha=0.18, color=c, linewidth=0)
            prev_ex   = exercises[i]
            seg_start = t[i]

    # Confidence line
    ax.plot(t, res["confidence"].values, color="#ffffff", linewidth=0.8,
            alpha=0.9, label="Konfidenz")
    ax.axhline(0.65, color="#f1c40f", linewidth=0.8, linestyle="--",
               alpha=0.6, label="Schwellwert (0.65)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Konfidenz")
    ax.legend(loc="lower right", framealpha=0.3)
    ax.set_title("Vorhergesagte Übung (Hintergrundfarbe) + Konfidenz")

    # ── Bottom: exercise label as colour bar ──────────────────────────────────
    ax2 = axes[1]
    # Map exercises to numeric for imshow
    all_classes = list(EXERCISE_COLORS.keys())
    ex_num = np.array([all_classes.index(e) if e in all_classes else 0
                       for e in exercises], dtype=float)
    ax2.imshow(ex_num[np.newaxis, :], aspect="auto",
               extent=[t[0], t[-1], 0, 1],
               cmap=matplotlib.colors.ListedColormap(
                   [EXERCISE_COLORS.get(c, DEFAULT_COLOR) for c in all_classes]
               ),
               vmin=0, vmax=len(all_classes) - 1, interpolation="nearest")

    # Legend patches
    present = sorted(set(exercises), key=lambda e: all_classes.index(e) if e in all_classes else 99)
    patches = [mpatches.Patch(color=EXERCISE_COLORS.get(e, DEFAULT_COLOR), label=e)
               for e in present]
    ax2.legend(handles=patches, loc="upper right", framealpha=0.4, ncol=len(present),
               fontsize=7)
    ax2.set_yticks([])
    ax2.set_title("Erkannte Übung")
    ax2.set_xlabel("Zeit (s)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out)


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 3: REP COUNTER step function
# ──────────────────────────────────────────────────────────────────────────────

def plot_rep_counter(res: pd.DataFrame, true_label: str, out: Path):
    _apply_style()
    true_color = EXERCISE_COLORS.get(true_label, DEFAULT_COLOR)

    fig, ax = plt.subplots(figsize=(16, 4))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(f"Wiederholungs-Zähler  ·  {true_label}",
                 fontsize=13, color=true_color, fontweight="bold")

    t    = res["t"].values
    reps = res["reps"].values

    ax.step(t, reps, color=true_color, linewidth=1.5, where="post")
    ax.fill_between(t, reps, step="post", alpha=0.2, color=true_color)

    # Mark each rep increment
    changes = np.where(np.diff(reps) > 0)[0]
    if len(changes):
        ax.scatter(t[changes + 1], reps[changes + 1],
                   color="#f1c40f", zorder=5, s=60,
                   label=f"Rep erkannt ({reps[-1]} total)")
        ax.legend(loc="upper left", framealpha=0.3)

    final = int(reps[-1]) if len(reps) else 0
    ax.set_ylabel("Wiederholungen")
    ax.set_xlabel("Zeit (s)")
    peak = int(np.max(reps)) if len(reps) else 0
    ax.set_title(f"Gezählte Reps: {peak}  (Endwert nach Reset: {final})")
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out)


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 0: GLOBAL SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

def plot_summary(all_results: list, out: Path):
    """Bar chart of final rep counts for all recordings."""
    _apply_style()
    n = len(all_results)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, n * 0.7 + 2)))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("Inference Zusammenfassung – alle Aufnahmen",
                 fontsize=14, color="#e0e0e0", fontweight="bold")

    # Left: bar chart of final rep counts
    ax = axes[0]
    labels  = [r["fname"] for r in all_results]
    reps    = [r["final_reps"] for r in all_results]
    colors  = [EXERCISE_COLORS.get(r["true_label"], DEFAULT_COLOR) for r in all_results]
    y_pos   = np.arange(len(labels))

    bars = ax.barh(y_pos, reps, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r['fname']}\n({r['true_label']})" for r in all_results],
                       fontsize=7)
    ax.set_xlabel("Gezählte Reps")
    ax.set_title("Finale Wiederholungszahl")
    for bar, val in zip(bars, reps):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8, color="#e0e0e0")

    # Right: classification accuracy (% frames correctly classified)
    ax2 = axes[1]
    acc = [r["accuracy_pct"] for r in all_results]
    bars2 = ax2.barh(y_pos, acc, color=colors, alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([r["true_label"] for r in all_results], fontsize=8)
    ax2.set_xlabel("Richtig klassifiziert (%)")
    ax2.set_xlim(0, 105)
    ax2.set_title("Klassifikationsgenauigkeit")
    for bar, val in zip(bars2, acc):
        ax2.text(min(bar.get_width() + 0.5, 101), bar.get_y() + bar.get_height() / 2,
                 f"{val:.0f}%", va="center", fontsize=8, color="#e0e0e0")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main(data_dir: str, artifacts_dir: str, out_dir: str, threshold: float):
    in_p  = Path(data_dir)
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(in_p.glob("*.csv"))
    # Skip synthetic files if they accidentally end up here
    csv_files = [f for f in csv_files if not f.stem.startswith("syn_")]
    if not csv_files:
        raise FileNotFoundError(f"Keine CSV-Dateien in {data_dir}")

    # Load the class list from artifacts
    import json
    enc_path = Path(artifacts_dir) / "label_encoder.json"
    classes = json.loads(enc_path.read_text())["classes"]

    all_results = []

    for f in csv_files:
        print(f"\n{'─'*60}")
        print(f"  {f.name}")
        print(f"{'─'*60}")

        res, true_label = run_inference(f, artifacts_dir, threshold)
        if res.empty:
            print("  (keine Inferenz-Ergebnisse – Datei zu kurz?)")
            continue

        sub = out_p / f.stem
        sub.mkdir(exist_ok=True)

        plot_probabilities(res, true_label, classes,
                           sub / "01_probabilities.png")
        plot_timeline(res, true_label,
                      sub / "02_timeline.png")
        plot_rep_counter(res, true_label,
                         sub / "03_rep_counter.png")

        # Accuracy: fraction of inference frames where predicted == true_label
        correct     = (res["exercise"] == true_label).mean() * 100
        final_reps  = int(res["reps"].max())

        all_results.append({
            "fname":        f.stem,
            "true_label":   true_label,
            "final_reps":   final_reps,
            "accuracy_pct": correct,
        })
        print(f"  Genauigkeit: {correct:.1f}%  |  Reps erkannt: {final_reps}")

    if all_results:
        print(f"\n{'─'*60}")
        print("  Globaler Summary-Plot …")
        print(f"{'─'*60}")
        plot_summary(all_results, out_p / "00_inference_summary.png")

    total = sum(1 for _ in out_p.rglob("*.png"))
    print(f"\nFertig. {total} PNG-Dateien in: {out_p.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot model inference on IMU CSVs")
    parser.add_argument("--data_dir",    default=".",          help="Verzeichnis mit CSV-Dateien")
    parser.add_argument("--artifacts",   default="artifacts",  help="Modell-Artefakte")
    parser.add_argument("--out_dir",     default="inference_plots", help="Ausgabeverzeichnis")
    parser.add_argument("--threshold",   type=float, default=0.65)
    args = parser.parse_args()
    main(args.data_dir, args.artifacts, args.out_dir, args.threshold)
