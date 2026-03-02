"""
Signal Plotter – Dual-IMU Exercise Data
========================================
Erzeugt für jede CSV-Datei mehrere Plots und speichert sie im Ordner plots/.

Plots pro Datei:
  1. overview    – alle Kanäle (ARM + FOOT) übereinander, volle Aufnahme
  2. accel       – Beschleunigung ax/ay/az, ARM vs FOOT nebeneinander
  3. gyro        – Gyroskop gx/gy/gz, ARM vs FOOT nebeneinander
  4. quaternion  – qw/qx/qy/qz, ARM vs FOOT nebeneinander
  5. rep_signal  – gefilterter Rep-Kanal mit erkannten Peaks markiert
  6. fft         – Frequenzspektrum der dominanten Kanäle (ARM + FOOT)

Zusätzlich ein globaler Vergleichs-Plot:
  7. compare_all – Beschleunigungsbetrag aller Übungen in einem Plot

Aufruf:
    python plot_signals.py [--data_dir ./] [--out_dir ./plots]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # kein GUI-Fenster nötig
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy.signal import butter, filtfilt, find_peaks

# ──────────────────────────────────────────────────────────────────────────────

ACCEL  = ["ax", "ay", "az"]
GYRO   = ["gx", "gy", "gz"]
QUAT   = ["qw", "qx", "qy", "qz"]
ALL_FT = ACCEL + GYRO + QUAT

FS = 50.0   # effektive Abtastrate pro Band

ACCEL_COLORS  = ["#e74c3c", "#2ecc71", "#3498db"]   # rot grün blau
GYRO_COLORS   = ["#e67e22", "#1abc9c", "#9b59b6"]   # orange türkis lila
QUAT_COLORS   = ["#f39c12", "#27ae60", "#2980b9", "#8e44ad"]

# Dominanter Kanal + Richtung pro Übung (für Rep-Signal-Plot)
REP_CFG = {
    "Push ups":         {"feat": "az", "band": "ARM",  "dir":  1, "min_dist": 60, "prom": 0.5},
    "Kniebeuge":        {"feat": "az", "band": "FOOT", "dir": -1, "min_dist": 60, "prom": 0.5},
    "Hampelmanner":     {"feat": "ax", "band": "ARM",  "dir":  1, "min_dist": 40, "prom": 0.4},
    "Sit Up":           {"feat": "ay", "band": "ARM",  "dir":  1, "min_dist": 60, "prom": 0.5},
    "Montain climbers": {"feat": "gz", "band": "ARM",  "dir":  1, "min_dist": 40, "prom": 0.4},
    "Rauschen":         {"feat": "az", "band": "ARM",  "dir":  1, "min_dist": 60, "prom": 0.5},
}

EXERCISE_COLORS = {
    "Push ups":         "#e74c3c",
    "Kniebeuge":        "#3498db",
    "Hampelmanner":     "#2ecc71",
    "Sit Up":           "#f39c12",
    "Montain climbers": "#9b59b6",
    "Rauschen":         "#95a5a6",
}

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


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _bandpass(sig, lo=0.3, hi=6.0):
    nyq = FS / 2.0
    lo_n, hi_n = lo / nyq, min(hi / nyq, 0.99)
    if lo_n >= hi_n or len(sig) < 27:
        return sig
    b, a = butter(2, [lo_n, hi_n], btype="band")
    return filtfilt(b, a, sig)


def _load(path: Path):
    df = pd.read_csv(path)
    df["band"]  = df["band"].astype(str).str.upper().str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    for c in ALL_FT:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts"] + ALL_FT).sort_values("ts").reset_index(drop=True)

    arm  = df[df["band"] == "ARM" ].reset_index(drop=True)
    foot = df[df["band"] == "FOOT"].reset_index(drop=True)

    # Zeit in Sekunden ab Start
    t0 = df["ts"].iloc[0]
    arm ["t"] = (arm ["ts"] - t0).dt.total_seconds()
    foot["t"] = (foot["ts"] - t0).dt.total_seconds()

    label = df["label"].mode()[0]
    return arm, foot, label


def _accel_mag(df):
    return np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2).values


def _apply_style():
    plt.rcParams.update(STYLE)


def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 1: OVERVIEW  (alle 10 Kanäle, beide Bänder)
# ──────────────────────────────────────────────────────────────────────────────

def plot_overview(arm, foot, label, out: Path):
    _apply_style()
    color = EXERCISE_COLORS.get(label, "#e0e0e0")

    groups = [
        ("Acceleration (ARM)",       arm,  ACCEL, ACCEL_COLORS),
        ("Acceleration (FOOT)",      foot, ACCEL, ACCEL_COLORS),
        ("Gyroscope (ARM)",          arm,  GYRO,  GYRO_COLORS),
        ("Gyroscope (FOOT)",         foot, GYRO,  GYRO_COLORS),
        ("Quaternion (ARM)",         arm,  QUAT,  QUAT_COLORS),
        ("Quaternion (FOOT)",        foot, QUAT,  QUAT_COLORS),
    ]

    fig, axes = plt.subplots(len(groups), 1, figsize=(16, 14), sharex=False)
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(f"Signal Overview  ·  {label}", fontsize=14, color=color,
                 fontweight="bold", y=0.995)

    for ax, (title, df, feats, colors) in zip(axes, groups):
        t = df["t"].values
        for feat, c in zip(feats, colors):
            ax.plot(t, df[feat].values, color=c, linewidth=0.7,
                    alpha=0.9, label=feat)
        ax.set_title(title, pad=4)
        ax.set_ylabel("value")
        ax.legend(loc="upper right", ncol=len(feats), framealpha=0.3)
        ax.set_xlim(t[0], t[-1])

    axes[-1].set_xlabel("Zeit (s)")
    fig.tight_layout(rect=[0, 0, 1, 0.995])
    _save(fig, out)


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 2/3/4: ACCEL / GYRO / QUATERNION  (ARM links, FOOT rechts)
# ──────────────────────────────────────────────────────────────────────────────

def plot_sensor_group(arm, foot, label, feats, colors, group_name, out: Path):
    _apply_style()
    color = EXERCISE_COLORS.get(label, "#e0e0e0")
    n = len(feats)

    fig, axes = plt.subplots(n, 2, figsize=(16, 3 * n), sharex="col")
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(f"{group_name}  ·  {label}", fontsize=13, color=color,
                 fontweight="bold")

    if n == 1:
        axes = axes[np.newaxis, :]

    for row, (feat, c) in enumerate(zip(feats, colors)):
        for col, (df, band_name) in enumerate([(arm, "ARM"), (foot, "FOOT")]):
            ax = axes[row, col]
            t = df["t"].values
            ax.plot(t, df[feat].values, color=c, linewidth=0.8)
            ax.set_title(f"{feat}  [{band_name}]", pad=3)
            ax.set_ylabel("value")
            ax.set_xlim(t[0], t[-1])
            if row == n - 1:
                ax.set_xlabel("Zeit (s)")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, out)


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 5: REP SIGNAL  (gefilterter Kanal + erkannte Peaks)
# ──────────────────────────────────────────────────────────────────────────────

def plot_rep_signal(arm, foot, label, out: Path):
    _apply_style()
    color = EXERCISE_COLORS.get(label, "#e0e0e0")

    cfg = REP_CFG.get(label, REP_CFG["Rauschen"])
    feat     = cfg["feat"]
    band_str = cfg["band"]
    direc    = cfg["dir"]
    min_dist = cfg["min_dist"]
    prom     = cfg["prom"]

    df = arm if band_str == "ARM" else foot
    t  = df["t"].values
    raw = df[feat].values.astype(np.float32)
    filtered = _bandpass(raw) * direc

    peaks, props = find_peaks(filtered, distance=min_dist, prominence=prom)

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(f"Rep Detection  ·  {label}  ·  {feat} [{band_str}]",
                 fontsize=13, color=color, fontweight="bold")

    # Oben: Rohsignal
    axes[0].plot(t, raw * direc, color="#7f8c8d", linewidth=0.7,
                 alpha=0.6, label="raw")
    axes[0].plot(t, filtered, color=color, linewidth=1.0, label="bandpass filtered")
    if len(peaks):
        axes[0].scatter(t[peaks], filtered[peaks], color="#f1c40f",
                        zorder=5, s=50, label=f"peaks ({len(peaks)})")
    axes[0].set_ylabel("amplitude")
    axes[0].legend(loc="upper right", framealpha=0.3)
    axes[0].set_title("Filtered signal + detected peaks")

    # Unten: Inter-Peak-Intervall (Periodenzeit → Tempo)
    if len(peaks) >= 2:
        intervals = np.diff(t[peaks])
        peak_mid_t = 0.5 * (t[peaks[:-1]] + t[peaks[1:]])
        axes[1].bar(peak_mid_t, intervals, width=intervals * 0.6,
                    color=color, alpha=0.7, label="inter-peak interval (s)")
        axes[1].axhline(np.mean(intervals), color="#f1c40f", linestyle="--",
                        linewidth=1.0, label=f"mean = {np.mean(intervals):.2f} s  "
                        f"({60/np.mean(intervals):.1f} reps/min)")
        axes[1].set_ylabel("Interval (s)")
        axes[1].set_title("Tempo – Zeit zwischen Wiederholungen")
        axes[1].legend(loc="upper right", framealpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "< 2 Peaks erkannt", transform=axes[1].transAxes,
                     ha="center", va="center", color="#7f8c8d", fontsize=12)
        axes[1].set_title("Tempo")

    axes[1].set_xlabel("Zeit (s)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, out)


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 6: FFT  (Frequenzspektrum beider Bänder)
# ──────────────────────────────────────────────────────────────────────────────

def plot_fft(arm, foot, label, out: Path):
    _apply_style()
    color = EXERCISE_COLORS.get(label, "#e0e0e0")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(f"FFT Frequenzspektrum  ·  {label}", fontsize=13,
                 color=color, fontweight="bold")

    plot_pairs = [
        (axes[0, 0], arm,  "ax", "#e74c3c", "ARM"),
        (axes[0, 1], arm,  "ay", "#2ecc71", "ARM"),
        (axes[0, 2], arm,  "az", "#3498db", "ARM"),
        (axes[1, 0], foot, "ax", "#e74c3c", "FOOT"),
        (axes[1, 1], foot, "ay", "#2ecc71", "FOOT"),
        (axes[1, 2], foot, "az", "#3498db", "FOOT"),
    ]

    for ax, df, feat, c, band_name in plot_pairs:
        sig = df[feat].values.astype(np.float32)
        sig -= sig.mean()
        N   = len(sig)
        fft = np.abs(np.fft.rfft(sig)) * 2 / N
        freqs = np.fft.rfftfreq(N, d=1.0 / FS)

        mask = (freqs >= 0.1) & (freqs <= 10.0)
        ax.fill_between(freqs[mask], fft[mask], color=c, alpha=0.5)
        ax.plot(freqs[mask], fft[mask], color=c, linewidth=0.9)

        # Dominant-Frequenz markieren
        peak_idx = np.argmax(fft[mask])
        dom_f = freqs[mask][peak_idx]
        ax.axvline(dom_f, color="#f1c40f", linestyle="--", linewidth=1.0,
                   label=f"dom. {dom_f:.2f} Hz")

        ax.set_title(f"{feat}  [{band_name}]", pad=3)
        ax.set_xlabel("Frequenz (Hz)")
        ax.set_ylabel("|FFT|")
        ax.legend(framealpha=0.3)
        ax.set_xlim(0.1, 10)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, out)


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 7: ARM vs FOOT Acceleration Magnitude nebeneinander
# ──────────────────────────────────────────────────────────────────────────────

def plot_magnitude(arm, foot, label, out: Path):
    _apply_style()
    color = EXERCISE_COLORS.get(label, "#e0e0e0")

    arm_mag  = _accel_mag(arm)
    foot_mag = _accel_mag(foot)

    fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=False)
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(f"Beschleunigungsbetrag  ·  {label}", fontsize=13,
                 color=color, fontweight="bold")

    for ax, t, mag, band_name in [
        (axes[0], arm["t"].values,  arm_mag,  "ARM"),
        (axes[1], foot["t"].values, foot_mag, "FOOT"),
    ]:
        ax.fill_between(t, mag, alpha=0.3, color=color)
        ax.plot(t, mag, color=color, linewidth=0.8)
        ax.axhline(np.mean(mag), color="#f1c40f", linestyle="--",
                   linewidth=0.9, label=f"mean = {np.mean(mag):.2f}")
        ax.set_title(f"|a|  [{band_name}]")
        ax.set_ylabel("m/s²")
        ax.legend(framealpha=0.3)
        ax.set_xlim(t[0], t[-1])

    axes[-1].set_xlabel("Zeit (s)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, out)


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 8: GLOBAL COMPARE  (alle Übungen, ARM-Beschleunigungsbetrag)
# ──────────────────────────────────────────────────────────────────────────────

def plot_compare_all(all_data: list[tuple], out: Path):
    _apply_style()
    n = len(all_data)

    fig, axes = plt.subplots(n, 1, figsize=(16, 3 * n), sharex=False)
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("Alle Übungen – ARM Beschleunigungsbetrag", fontsize=14,
                 color="#e0e0e0", fontweight="bold")

    if n == 1:
        axes = [axes]

    for ax, (arm, _, label, fname) in zip(axes, all_data):
        t   = arm["t"].values
        mag = _accel_mag(arm)
        c   = EXERCISE_COLORS.get(label, "#e0e0e0")

        ax.fill_between(t, mag, alpha=0.25, color=c)
        ax.plot(t, mag, color=c, linewidth=0.8)
        ax.set_title(f"{label}  ({fname})", pad=3)
        ax.set_ylabel("|a| ARM")
        ax.set_xlim(t[0], t[-1])

    axes[-1].set_xlabel("Zeit (s)")
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    _save(fig, out)


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 9: HEATMAP – Korrelation aller Features (ARM)
# ──────────────────────────────────────────────────────────────────────────────

def plot_correlation(arm, label, out: Path):
    _apply_style()
    color = EXERCISE_COLORS.get(label, "#e0e0e0")

    corr = arm[ALL_FT].corr().values

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#1a1a2e")

    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ALL_FT)))
    ax.set_yticks(range(len(ALL_FT)))
    ax.set_xticklabels(ALL_FT, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ALL_FT, fontsize=9)

    for i in range(len(ALL_FT)):
        for j in range(len(ALL_FT)):
            val = corr[i, j]
            txt_color = "white" if abs(val) > 0.6 else "#aaaaaa"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#a0a0a0")

    ax.set_title(f"Feature Korrelation (ARM)  ·  {label}", fontsize=12,
                 color=color, fontweight="bold", pad=10)
    fig.tight_layout()
    _save(fig, out)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main(data_dir: str, out_dir: str):
    in_p  = Path(data_dir)
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(in_p.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Keine CSV-Dateien in {data_dir}")

    all_data = []

    for f in csv_files:
        print(f"\n{'─'*60}")
        print(f"  {f.name}")
        print(f"{'─'*60}")

        arm, foot, label = _load(f)
        slug = f.stem          # Dateiname ohne .csv als Präfix
        sub  = out_p / slug
        sub.mkdir(exist_ok=True)

        all_data.append((arm, foot, label, f.stem))

        plot_overview(arm, foot, label,
                      sub / "01_overview.png")

        plot_sensor_group(arm, foot, label, ACCEL, ACCEL_COLORS,
                          "Beschleunigung (m/s²)",
                          sub / "02_acceleration.png")

        plot_sensor_group(arm, foot, label, GYRO, GYRO_COLORS,
                          "Gyroskop (rad/s)",
                          sub / "03_gyroscope.png")

        plot_sensor_group(arm, foot, label, QUAT, QUAT_COLORS,
                          "Quaternion",
                          sub / "04_quaternion.png")

        plot_rep_signal(arm, foot, label,
                        sub / "05_rep_detection.png")

        plot_fft(arm, foot, label,
                 sub / "06_fft_spectrum.png")

        plot_magnitude(arm, foot, label,
                       sub / "07_magnitude.png")

        plot_correlation(arm, label,
                         sub / "08_correlation.png")

    # Globaler Vergleich
    print(f"\n{'─'*60}")
    print("  Globaler Vergleichs-Plot …")
    print(f"{'─'*60}")
    plot_compare_all(all_data, out_p / "00_compare_all_exercises.png")

    print(f"\nFertig. Alle Plots in: {out_p.resolve()}")
    total = sum(1 for _ in out_p.rglob("*.png"))
    print(f"  {total} PNG-Dateien erstellt.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot IMU exercise signals")
    parser.add_argument("--data_dir", default=".",     help="Verzeichnis mit CSV-Dateien")
    parser.add_argument("--out_dir",  default="plots", help="Ausgabeverzeichnis für Plots")
    args = parser.parse_args()
    main(args.data_dir, args.out_dir)
