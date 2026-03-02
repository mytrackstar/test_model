# Dual-IMU Workout Recognition – Modelldokumentation

## Übersicht

Ein Deep-Learning-System zur **Echtzeit-Erkennung von Calisthenics-Übungen** und **automatischen Wiederholungszählung** auf Basis von zwei BNO055-IMU-Sensoren (ARM + FOOT).

---

## 1. Datenbasis

### Echte Aufnahmen (8 CSV-Dateien)

| Datei | Übung | Zeilen | Dauer |
|---|---|---|---|
| `imu_..._Pushups.csv` × 2 | Push ups | 5 101 | ~51 s |
| `imu_..._Kniebeuge.csv` × 2 | Kniebeuge | 3 927 | ~39 s |
| `imu_..._Hampelmanner.csv` | Hampelmanner | 2 438 | ~24 s |
| `imu_..._Situp.csv` | Sit Up | 3 304 | ~33 s |
| `imu_..._Montainclimbers.csv` | Montain climbers | 2 484 | ~25 s |
| `imu_..._Rauschen.csv` | Rauschen (Ruhezustand) | 3 954 | ~40 s |

**Gesamt real:** 21 208 Zeilen, 6 Klassen

### Eingabespalten

```
timestamp, band (ARM|FOOT), millis,
qw, qx, qy, qz,          ← Quaternion-Orientierung
ax, ay, az,               ← Linearbeschleunigung
gx, gy, gz,               ← Gyroskop
sysCal, gyroCal, accelCal, magCal,   ← Kalibrierungsstatus
label
```

**Features für das Modell:** `ax, ay, az, gx, gy, gz, qw, qx, qy, qz` → **10 Features × 2 Bänder = 20 Kanäle**

---

## 2. Datenpipeline

### 2.1 Vorverarbeitung

1. ARM- und FOOT-Streams werden per **Timestamp** aufeinander ausgerichtet (`merge_asof`, Toleranz 50 ms)
2. Kalibrierungsfilter: `sysCal ≥ 2` und `gyroCal ≥ 2` – schlechte Samples werden verworfen
3. **Sliding-Window-Segmentierung**: Fenstergröße 100 Samples (~1 s @ 50 Hz), Stride 20 Samples (80 % Überlappung)
4. **Window-Label** = Mehrheitsvote aller Labels im Fenster
5. **Normalisierung**: `StandardScaler` pro Band (fit nur auf Trainingsdaten, dann auf Val übertragen)

### 2.2 Klassische Augmentierung (in `train_full.py`)

Wird auf jedes Trainingsfenster **×4** angewendet, wobei 1–3 der folgenden Transformationen zufällig kombiniert werden:

| Methode | Beschreibung | Parameter |
|---|---|---|
| **Gaussian Jitter** | Additives Rauschen auf alle Kanäle | σ = 0.025 |
| **Amplitude Scaling** | Multiplikativer Faktor auf alle Werte | 0.85 – 1.15 |
| **Time Warping** | Zeitachse strecken/stauchen via Interpolation | ±15 % |
| **Magnitude Perturb** | Per-Kanal multiplikatives Rauschen | σ = 0.1 |
| **Gravity Flip** | Zufällige Vorzeichenumkehr eines Beschleunigungskanals | Simuliert Sensor-Flip |

### 2.3 Periodizitäts-basierte Synthetisierung (in `generate_synthetic.py`)

Da alle Übungen **periodische Bewegungsmuster** sind, wird die Struktur ausgenutzt:

```
Echte Aufnahme
    ↓
FFT → dominante Frequenz bestimmen
    ↓
Bandpassfilter (0.3 – 6 Hz) auf Hauptachse
    ↓
Peak Detection → einzelne Zyklen extrahieren
    ↓
Zyklen-Bibliothek (25–100 Zyklen pro Übung/Band)
    ↓
Synthetic Recording:
  für N_reps (8–25):
    - zufälligen Zyklus wählen
    - Tempo ±15 % variieren (Resampling)
    - Amplitude jittern, DC-Offset shiften, Rauschen addieren
  → zusammennähen
    ↓
Fertiges synthetisches CSV (gleiche Spalten wie echte Daten)
```

**Extrahierte Zyklen:**
| Übung | ARM Zyklen | FOOT Zyklen |
|---|---|---|
| Push ups | 41 | 62 |
| Kniebeuge | 26 | 98 |
| Hampelmanner | 25 | 57 |
| Sit Up | 33 | 93 |
| Montain climbers | 49 | 49 |

**Ergebnis:** 120 synthetische CSVs → 159 624 Gesamtzeilen (8× mehr als real)

### 2.4 Datenbilanz nach Training mit real + synthetisch

| Klasse | Samples (aligned) |
|---|---|
| Kniebeuge | 15 979 |
| Push ups | 14 431 |
| Sit Up | 11 780 |
| Rauschen | 10 414 |
| Hampelmanner | 8 149 |
| Montain climbers | 4 320 |

---

## 3. Modellarchitektur

### 3.1 Überblick: Dual-Stream CNN + BiLSTM + Attention

```
ARM Input [B, 100, 10]          FOOT Input [B, 100, 10]
       │                                  │
   ┌───▼────────────────┐    ┌────────────▼───┐
   │     IMUStream       │    │   IMUStream     │
   │  Conv1D(10→64, k=5) │    │  Conv1D(10→64)  │
   │  BatchNorm + GELU   │    │  BatchNorm + GELU│
   │  MaxPool1D(2)       │    │  MaxPool1D(2)   │
   │  Conv1D(64→128,k=5) │    │  Conv1D(64→128) │
   │  BatchNorm + GELU   │    │  BatchNorm + GELU│
   │  MaxPool1D(2)       │    │  MaxPool1D(2)   │
   │  Conv1D(128→128,k=3)│    │  Conv1D(128→128)│
   │  BatchNorm + GELU   │    │  BatchNorm + GELU│
   │  BiLSTM(128→256)    │    │  BiLSTM(128→256) │
   │  2 Layers, Dropout  │    │  2 Layers, Dropout│
   │  Temporal Attention │    │  Temporal Attention│
   │     → [B, 256]      │    │     → [B, 256]  │
   └────────┬────────────┘    └──────────┬──────┘
            └──────────────┬─────────────┘
                      Concat [B, 512]
                           │
                   Linear(512→256)
                   GELU + Dropout(0.4)
                   Linear(256→128)
                   GELU + Dropout(0.3)
                   Linear(128→6)
                           │
                     Softmax → Klasse
```

### 3.2 Details

| Komponente | Konfiguration |
|---|---|
| **Parameter gesamt** | 1 672 968 |
| **Aktivierungsfunktion** | GELU (besser als ReLU für IMU-Daten) |
| **Normalisierung** | BatchNorm1D nach jedem Conv-Layer |
| **Regularisierung** | Dropout 0.3 (LSTM), 0.4/0.3 (FC) |
| **Attention** | Gelerntes 1D Temporal-Gewicht → gewichtete Summe der LSTM-Outputs |
| **Loss** | CrossEntropyLoss mit klassengewichteter Korrektur (Imbalance) |
| **Optimizer** | AdamW, lr=1e-3, weight_decay=1e-4 |
| **LR-Schedule** | CosineAnnealingLR (eta_min=1e-5) |
| **Gradient Clipping** | max_norm=2.0 |

### 3.3 Warum diese Architektur?

- **Conv1D:** Erkennt lokale Bewegungsmuster (z. B. der Aufwärtsstoß beim Push-up) unabhängig von der exakten Position im Fenster
- **BiLSTM:** Versteht die zeitliche Abfolge der Bewegungsphasen in beide Zeitrichtungen
- **Temporal Attention:** Gewichtet relevante Zeitpunkte stärker (z. B. Umkehrpunkt einer Bewegung)
- **Dual-Stream:** ARM und FOOT liefern komplementäre Information – ein Squat sieht im Bein komplett anders aus als im Arm

---

## 4. Live-Inferenz & Wiederholungszählung

### 4.1 Echtzeit-Pipeline

```
Eingehende Sensorzeile (ARM oder FOOT)
    ↓
Sliding Buffer (deque, 100 Samples pro Band)
    ↓
Alle 20 Samples (= 20 % Stride): Inferenz
    ↓
Softmax-Wahrscheinlichkeiten
    ↓
5-Frame Majority-Vote Smoother
(verhindert kurze Fehlklassifikationen)
    ↓
Stabile Klasse → RepCounter
```

### 4.2 Wiederholungszählung

Für jede Übung wird ein spezifischer Sensorkanal bandpassgefiltert (0.5–5 Hz) und per `scipy.find_peaks` ausgewertet:

| Übung | Kanal | Richtung | min_dist | Prominence |
|---|---|---|---|---|
| Push ups | `az_arm` | Peak ↑ | 60 | 0.5 |
| Kniebeuge | `az_foot` | Tal ↓ | 60 | 0.5 |
| Hampelmanner | `ax_arm` | Peak ↑ | 40 | 0.4 |
| Sit Up | `ay_arm` | Peak ↑ | 60 | 0.5 |
| Montain climbers | `gz_arm` | Peak ↑ | 40 | 0.4 |

**Absolute Positionsverfolgung:** Jeder gefundene Peak wird mit seiner absoluten Stream-Position (nicht Buffer-relativer Index) abgeglichen → keine Doppelzählung trotz scrollendem Buffer.

---

## 5. Ergebnisse

### 5.1 Training (nur echte Daten, 8 Dateien)

- Validierungsgenauigkeit: **100 %** (aber: Datenleck durch überlappende Fenster derselben Aufnahmen)
- Alle 6 Klassen sauber getrennt

### 5.2 Training (real + synthetisch, 128 Dateien)

- Validierungsgenauigkeit: **59 %** (ehrlicheres Bild – echte Verteilungsvielfalt)
- Live-Inferenz auf echten Aufnahmen: weiterhin **~100 % Konfidenz**

### 5.3 Live-Replay Ergebnisse

| Aufnahme | Erkannte Übung | Konfidence | Gezählte Reps |
|---|---|---|---|
| Pushups (1) | Push ups | 0.99 | 38 |
| Pushups (2) | Push ups | 0.99 | 20 |
| Kniebeuge (1) | Kniebeuge | 1.00 | 22 |
| Kniebeuge (2) | Kniebeuge | 1.00 | 20 |
| Hampelmanner | Hampelmanner | 1.00 | 43 |
| Sit Up | Sit Up | 1.00 | 38 |
| Montain climbers | Montain climbers | 1.00 | 27 |
| Rauschen | Rauschen | 1.00 | 0 |

---

## 6. Stärken

- **Robuste Architektur:** CNN+BiLSTM+Attention ist State-of-the-Art für multivariate Zeitreihensegmentierung
- **Dual-Stream:** Die Verwendung zweier Körperstellen erhöht die Diskriminierbarkeit erheblich (Squat im Bein vs. Arm)
- **Prediction Smoother:** Kurze Einzelfehlklassifikationen setzen den Rep-Counter nicht zurück
- **Skalierbar:** Neue Übungen einfach durch Aufnahme neuer CSVs und Retraining hinzufügen
- **Selbständige Artefakte:** Modell, Scaler und Label-Encoder werden separat gespeichert → portabel
- **Synthetische Datengenerierung:** Nutzt explizit die Periodizität der Übungen aus – keine blinde Augmentierung

---

## 7. Schwächen & Verbesserungspotenzial

### 7.1 Datenmenge – kritischstes Problem

| Problem | Auswirkung |
|---|---|
| Nur **1 Person** aufgenommen | Modell kennt nur eine Körpergröße, einen Bewegungsstil |
| Nur **eine Session** pro Übung | Keine Variation in Müdigkeit, Tempo, Aufwärmen |
| **Hampelmanner & Montain climbers** stark unterrepräsentiert | Modell könnte diese im echten Einsatz schlechter erkennen |

**Empfehlung:** Mindestens 3–5 Personen, 3–5 Sessions pro Person, verschiedene Tempi (langsam / mittel / schnell), verschiedene Sensor-Sitzpositionen.

### 7.2 Synthetische Daten – ARM/FOOT Entkopplung

Das aktuelle Generierungsscript erzeugt ARM- und FOOT-Zyklen **unabhängig voneinander**. In echten Aufnahmen sind die beiden Sensoren zeitlich korreliert (beide reagieren auf dieselbe Körperbewegung). Diese fehlende Korrelation erklärt den Rückgang auf 59 % Validierungsgenauigkeit.

**Verbesserung:** Zyklen als **ARM+FOOT-Paare** aus denselben Zeitfenstern extrahieren und gemeinsam augmentieren.

### 7.3 Rep-Counting – Signal-basiert, nicht ML-basiert

Die Wiederholungszählung basiert auf heuristischer Peak-Detection. Das funktioniert gut für klare periodische Signale, versagt aber bei:
- Unregelmäßigen Tempi (Pause in der Mitte)
- Halbrepetitionen (z. B. halber Squat)
- Übungsübergängen

**Verbesserung:** Phasen pro Übung als eigene Klasse trainieren (z. B. `pushup_up`, `pushup_down`) und Zustandsübergänge zählen.

### 7.4 Kalibrierungsabhängigkeit

Der Kalibrierungsfilter (`sysCal ≥ 2`) verwirft unkalibrierte Samples. Bei einem neuen Gerät oder einer neuen Umgebung kann die BNO055-Kalibrierung länger dauern → anfangs weniger Inferenzdaten.

**Verbesserung:** Kalibrierungsunabhängiges Training (Filter absenken oder ganz weglassen) oder einen Kalibrierungs-Warmup-Indikator in der App einbauen.

### 7.5 Echtzeit-Latenz

Das Modell feuert alle 20 Samples (~0.4 s). Für eine snappige App-Reaktion wäre ein kleineres/schnelleres Modell sinnvoll.

**Verbesserung:** Knowledge Distillation (kleines Schüler-Modell lernt vom großen Lehrer-Modell), oder MobileNet-Style Depthwise-Separable Convolutions.

### 7.6 Noch fehlende Übungen

Das Modell kennt nur 5 Übungen + Rauschen. Typische Calisthenics-Übungen die fehlen:
- Pull-ups / Chin-ups
- Dips
- Burpees
- Lunges
- Plank (isometrisch, kein Rep-Counting nötig)

---

## 8. Empfohlene nächste Schritte (priorisiert)

1. **Mehr echte Daten aufnehmen** (höchste Priorität)
   - Ziel: 5+ Personen × 3+ Sessions × 5+ Übungen
   - Verschiedene Tempi und Sensor-Positionen variieren

2. **ARM/FOOT-Sync in `generate_synthetic.py`** korrigieren
   - Zyklen als gekoppelte Paare extrahieren
   - Beide Bänder mit identischer Tempovariierung resamplen

3. **Phasen-basiertes Rep-Counting** implementieren
   - Pro Übung 2–3 Phasen labeln (z. B. `down`, `up`, `hold`)
   - Zustandsmaschine zählt Übergänge statt Peaks

4. **Proper Hold-Out-Evaluierung**
   - Ganze Aufnahmen als Validation reservieren (nicht zufällige Fenster)
   - Z. B. immer die zweite Aufnahme pro Übung als Testset

5. **Modell komprimieren** für mobile Nutzung
   - Zielgröße: < 500 KB, < 10 ms Inferenzzeit auf ARM-Prozessor

---

## 9. Datei-Übersicht

```
test_model/
├── train_full.py          # Haupttraining (real + synthetisch)
├── generate_synthetic.py  # Periodizitäts-basierte Datengenerierung
├── live_inference.py      # Echtzeit-Inferenz + Rep-Counter
├── artifacts/
│   ├── dual_imu_model.pt  # Trainiertes Modell (PyTorch state_dict)
│   ├── label_encoder.json # Klassen-Mapping (Index → Name)
│   └── scalers.json       # StandardScaler-Parameter (mean, scale)
├── synthetic/             # 120 generierte CSV-Dateien
└── imu_*.csv              # 8 echte Aufnahmen
```
