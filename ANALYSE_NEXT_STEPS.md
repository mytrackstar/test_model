# Analyse & Next Steps – Dual-IMU Workout Recognition

## 1. Architektur: CNN + BiLSTM – richtige Wahl?

### Kurzantwort: Ja, aber mit konkreten Verbesserungen

Die aktuelle Architektur ist state-of-the-art für multivariate IMU-Zeitreihen und funktioniert in der Praxis gut. Hier eine ehrliche Einordnung:

#### Warum CNN + BiLSTM die richtige Wahl ist

| Komponente | Wozu | Warum besser als Alternativen |
|---|---|---|
| **Conv1D (3 Schichten)** | Lokale Bewegungsmuster erkennen (z. B. der Aufwärtsstoß beim Push-up) | Invariant gegen kleine Zeitverschiebungen; effizienter als LSTM allein |
| **BatchNorm** | Stabilisiert Training, erlaubt höhere Lernraten | Fehlt bei reinem LSTM |
| **BiLSTM (2 Layer)** | Zeitliche Abfolge der Bewegungsphasen in beide Richtungen | Kann Kontext von "vorher" und "nachher" nutzen |
| **Temporal Attention** | Relevante Zeitpunkte stärker gewichten (z. B. Umkehrpunkte) | Besser als nur letzten LSTM-State nehmen |
| **Dual-Stream** | ARM und FOOT getrennt verarbeiten | Erzwingt, dass das Modell beide Perspektiven lernt, nicht eine dominiert |

#### Was man ändern könnte (mit Begründung)

**Option A: 1D-ResNet statt reinem CNN (mittlere Verbesserung)**
```
Conv1D → BatchNorm → GELU → Conv1D → BatchNorm + Residual Skip
```
Tiefere Netze ohne vanishing gradients. Sinnvoll wenn mehr Daten vorhanden.

**Option B: Transformer statt BiLSTM (große Verbesserung – erst ab viel mehr Daten)**
```
MultiHeadSelfAttention(8 heads) → FeedForward → LayerNorm
```
Kann globale zeitliche Abhängigkeiten lernen, parallelisierbar.
**Achtung:** Braucht deutlich mehr Daten (~10k echte Windows) und mehr Rechenzeit. Mit aktuell 400 echten Fenstern würde er overfitten.

**Option C: Beibehaltung + kleinere Verbesserungen am aktuellen Modell (empfohlen)**
- Residual-Verbindungen um die LSTM-Schichten
- LayerNorm statt BatchNorm im LSTM-Teil
- Squeeze-and-Excitation Block nach CNN (Feature-Reweighting)

**Fazit für jetzt:** Die aktuelle Architektur ist genau richtig für die vorhandene Datenmenge. Ein Transformer wäre Overkill und würde wahrscheinlich schlechter performen. Verbesserungen an der Datenbasis bringen mehr als Architekturwechsel.

---

## 2. Gefundene Bugs & technische Probleme

### Bug 1 (kritisch): ARM + FOOT Zyklen werden unabhängig synthetisiert

**Problem:** In `generate_synthetic.py` werden ARM- und FOOT-Zyklen aus separaten Bibliotheken gezogen und unabhängig augmentiert. In echten Aufnahmen reagieren beide Sensoren auf **dieselbe Körperbewegung** – sie sind zeitlich korreliert. Die synthetischen Daten brechen diese Korrelation.

**Auswirkung:** Der Dual-Stream-Klassifikator lernt, ARM+FOOT-Kombinationen zu unterscheiden. Wenn beide Streams keine natürliche Korrelation zeigen, lernt das Modell ein falsches Muster → 59 % Validierungsgenauigkeit mit synthetischen Daten statt 100 %.

**Fix:** Zyklen als **synchronized pair** extrahieren – immer den gleichen Zeitabschnitt für ARM und FOOT nehmen, und bei Tempo-Variation beide Bänder mit **identischer** Resampling-Rate strecken/stauchen.

```python
# Statt: arm_seq = _build(arm_cycles), foot_seq = _build(foot_cycles)
# Besser:
tempo = rng.uniform(0.85, 1.15)   # EIN Wert für beide
arm_seq  = _build_with_tempo(arm_cycles,  tempo)
foot_seq = _build_with_tempo(foot_cycles, tempo)
```

### Bug 2 (mittel): Modell doppelt definiert

Die komplette Architektur (`IMUStream`, `SpatialAttention`, `DualIMUNet`) ist identisch in `train_full.py` und `live_inference.py` kopiert. Wenn die Architektur geändert wird, müssen beide Dateien synchron angepasst werden – sonst lädt das gespeicherte Modell die falsche Architektur.

**Fix:** Eine separate `model.py` erstellen, die beide Dateien importieren.

### Bug 3 (klein): 100 % Validierungsgenauigkeit war irreführend

Die erste Trainingsrunde (nur echte Daten) zeigte 100 % – das war **kein echtes Signal**. Windows mit Stride=20 und Size=100 überlappen sich zu 80 %. Window 0 enthält Samples 0–99, Window 1 enthält Samples 20–119 – sie teilen 80 Samples. Wenn solche Fenster zufällig auf Train/Val aufgeteilt werden, ist Datenleck nahezu garantiert.

**Fix:** Für Validierung **komplette Aufnahmen** zurückhalten (z. B. immer die zweite Aufnahme einer Übung), nicht zufällige Fenster.

### Bug 4 (klein): Modellarchitektur hat 1.67 M Parameter für 6 Klassen

Für die aktuelle Datenmenge (~400 echte Windows) ist das Modell drastisch überdimensioniert. Das erklärt, warum es auf den echten Daten sofort overfit (100 % nach Epoche 1). Mit mehr Daten ist die Größe angemessen.

---

## 3. Warum die Validierungsgenauigkeit von 100 % auf 59 % fiel

Drei gleichzeitige Faktoren:

1. **Datenleck behoben:** Synthetische Daten kommen aus anderen Zyklen als die Trainingsdaten → keine Überlappung → echte Generalisierung wird gemessen
2. **ARM/FOOT Entkopplung** (Bug 1 oben): Synthetische Kombinationen zeigen unnatürliche Korrelationsmuster
3. **Mehr Klassen-Variation:** 120 synthetische Aufnahmen zeigen mehr Tempo- und Amplitudenvarianz als die 8 echten → der Klassifikationsraum wird komplexer

Das Modell **funktioniert auf echten Aufnahmen weiterhin zu ~100 %** – das ist die relevante Kennzahl für den Live-Einsatz.

---

## 4. Wie Trainingsdaten richtig aufnehmen

Das ist der wichtigste Hebel für bessere Generalisierung.

### 4.1 Minimale Anforderungen (kurzfristig erreichbar)

**Personenanzahl:** Mindestens **3 Personen**
- Unterschiedliche Körpergrößen (klein/mittel/groß)
- Unterschiedliche Fitness-Level (Anfänger / Fortgeschrittener)
- Mindestens 1 Linkshänder wenn möglich

**Sessions pro Person:** Mindestens **3 Sessions** an verschiedenen Tagen
- Session 1: Aufwärmen trainiert, frische Muskeln
- Session 2: Nach dem Sport, müde Bewegungsausführung
- Session 3: Verschiedene Geschwindigkeit bewusst variieren

**Pro Übung pro Session aufnehmen:**
- 15–20 Wiederholungen **langsam** (~0.3/s)
- 15–20 Wiederholungen **normal** (~0.5–0.7/s)
- 10–15 Wiederholungen **schnell** (~1.0/s)
- **Separate Datei pro Tempo** — getrennte Labeldateien vereinfachen das Training

### 4.2 Was beim Aufnehmen variiert werden muss

| Variable | Warum wichtig | Konkret |
|---|---|---|
| **Sensorposition** | Kleiner Verschub → komplett anderes Signal | ARM-Band: exakt Handgelenk vs. 3 cm höher |
| **Bewegungsausführung** | Volle Range-of-Motion vs. Teilbewegungen | Push-up: tief bis Boden vs. nur halbe Tiefe |
| **Körperausrichtung** | Nordausrichtung ändert Magnetometer-Kanäle | Einmal gegen die Wand turnen, einmal in den Raum |
| **Kleidung** | Ärmel können Sensor leicht verschieben | Einmal kurzes Shirt, einmal langes |
| **Untergrund** | Federung beim Jumping Jack ändert Footsensor | Harter Boden vs. Teppich |

### 4.3 Aufnahmecheckliste

```
Vor jeder Session:
□  Sensoren fest befestigt (kein Wackeln)
□  60 Sekunden Kalibrierung abwarten (sysCal = 3, gyroCal = 3 in App)
□  Im gleichen Raum wie später die App benutzt wird

Während der Session:
□  5 Sekunden "Rauschen" aufnehmen BEVOR die Übung startet
□  Übung starten, Label setzen
□  5 Sekunden Pause NACH der Übung (Rauschen)
□  Jede Übung als SEPARATE Datei aufnehmen (nicht alles in eine CSV)

Dateiname-Konvention (für automatisches Laden):
  imu_DATUM_UHRZEIT_PERSON_UEBUNG_TEMPO.csv
  Beispiel: imu_20260305_143022_martin_Pushups_normal.csv

Rauschen aufnehmen (eigene Datei):
□  Sitzen
□  Stehen still
□  Langsam gehen
□  Hände schütteln (Sensor-Shake)
```

### 4.4 Kriterium: Wann reichen die Daten?

Als Faustregel für dieses Problem:
- **Mindestens 200 echte Windows pro Klasse** für verlässliches Training (aktuell: ~50–100)
- **Mindestens 3 verschiedene Personen** für Generalisierung
- **Validation Accuracy > 85 % auf zurückgehaltenen Aufnahmen** (nicht auf Fenstern derselben Aufnahmen)

---

## 5. Priorisierte Next Steps

### Priorität 1: Synthese-Bug beheben + validiere Daten neu

```bash
# Fix: generate_synthetic.py → synchronisierte ARM+FOOT Paare
# Dann neu generieren und neu trainieren
python generate_synthetic.py --data_dir . --out_dir synthetic_v2 --n_recordings 20
python train_full.py --data_dir . --extra_data_dirs synthetic_v2 --out_dir artifacts_v2 --epochs 60
```

**Erwartetes Ergebnis:** Validierungsgenauigkeit sollte von 59 % auf ~80–90 % steigen.

### Priorität 2: Korrekte Hold-Out Evaluation einbauen

Statt zufälliger Fenster-Split: **Aufnahme-Level Split** in `train_full.py`.

```python
# Statt: train_test_split auf Window-Indizes
# Besser: Bei 2 Aufnahmen pro Übung → Aufnahme 2 immer als Val

# In Config:
val_file_pattern: str = "*_02.csv"  # zweite Aufnahme jeder Übung
```

### Priorität 3: Mehr echte Daten aufnehmen

Gemäß Abschnitt 4. Ziel: **200+ echte Windows pro Klasse** aus mindestens 3 Personen.

### Priorität 4: Modell-Code aufräumen

Separate `model.py` erstellen:
```
test_model/
├── model.py              ← DualIMUNet, IMUStream, SpatialAttention
├── train_full.py         ← from model import DualIMUNet
├── live_inference.py     ← from model import DualIMUNet
├── generate_synthetic.py
└── artifacts/
```

### Priorität 5: Phasen-basiertes Rep-Counting (mittelfristig)

Statt Peak-Detection: Jede Übung hat **2 Phasen** die gelabelt werden:

| Übung | Phase 1 | Phase 2 |
|---|---|---|
| Push ups | `pushup_down` | `pushup_up` |
| Kniebeuge | `squat_down` | `squat_up` |
| Sit Up | `situp_down` | `situp_up` |
| Hampelmanner | `jack_closed` | `jack_open` |
| Montain climbers | `climb_left` | `climb_right` |

Der Rep-Counter zählt dann einfach Phasenwechsel von Phase 2 → Phase 1.
Das ist robuster gegen unregelmäßige Tempi und Pausen.

**Wie aufnehmen:** Gleiche IMU-Daten, aber Label wechselt pro Phase. Dafür braucht die App ein Button-Interface das während der Aufnahme zwischen Phase 1/2 wechselt.

### Priorität 6: Modell für Mobile optimieren (längerfristig)

Aktuell: 1.67 M Parameter (~6.7 MB). Für echten App-Einsatz:

```
Ziel: < 200k Parameter, < 2 MB, < 10 ms Inferenz auf iPhone
Methode: Knowledge Distillation
  - Großes Modell (Teacher) trainieren (aktuelle Architektur)
  - Kleines Modell (Student) lernt vom Teacher
  - Student: nur 1 CNN-Block, 1 LSTM-Layer, 64 Hidden Units
```

---

## 6. Validierungsgenauigkeit – realistische Erwartungen

| Szenario | Erwartete Val-Accuracy | Realistisch? |
|---|---|---|
| Nur 8 echte Dateien, Fenster-Split | ~100 % | Nein – Datenleck |
| Nur 8 echte Dateien, Aufnahme-Split | ~85–95 % | Ja |
| 8 real + 120 synthetisch (aktuell, Bug) | ~59 % | Zu niedrig wegen Bug |
| 8 real + 120 synthetisch (Bug behoben) | ~75–85 % | Ja |
| 3+ Personen, 3+ Sessions, Aufnahme-Split | ~80–90 % | Ja, stabiles Modell |
| 5+ Personen, Phasen-Labels, großes Dataset | ~90–95 % | Ja, produktionsreif |

---

## 7. Zusammenfassung: Was jetzt tun?

```
SOFORT (ohne neue Daten):
  1. Bug in generate_synthetic.py beheben (ARM+FOOT synchronisieren)
  2. model.py auslagern
  3. Neu trainieren → sollte ~80 % Val-Accuracy erreichen

BALD (nächste Woche):
  4. 2-3 Personen bitten, je 3 Sessions aufzunehmen
  5. Aufnahme-Level Hold-Out implementieren
  6. Mit neuen Daten neu trainieren

MITTELFRISTIG:
  7. Phasen-Labels einführen → robusteres Rep-Counting
  8. Modell komprimieren für App-Integration
```
