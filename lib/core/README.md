# Backend Fahrererkennung

Fahrererkennung anhand von Fahrsimulator-Daten. Das System klassifiziert den Fahrer aus CSV-Recordings mittels maschinellem Lernen (RandomForest, LogisticRegression, GradientBoosting).

---

## Inhaltsverzeichnis

1. [Überblick](#1-überblick)
2. [Installation](#2-installation)
3. [Schnellstart](#3-schnellstart)
4. [Projektstruktur](#4-projektstruktur)
5. [Nutzung](#5-nutzung)
6. [Datenformate](#6-datenformate)
7. [Konfiguration](#7-konfiguration)
8. [Grafiken (plots/)](#8-grafiken-plots)

---

## 1. Überblick

### Funktionen

| Funktion | Beschreibung |
|----------|--------------|
| **Training** | Trainiert RandomForest und LogisticRegression auf gelabelten Recordings |
| **Vorhersage** | Klassifiziert unbekannte Recordings mit trainierten Modellen |
| **Pipeline-Fortschritt** | `pipeline_progress.json` / `progress_callback` für Frontend-Statusanzeige |
| **Grafiken** | Konfusionsmatrix, Feature Importance, Accuracy |

### Technik

- **Features:** Featuretools (Aggregationen) + TSFresh (Zeitreihen-Features)
- **Modelle:** RandomForest, LogisticRegression, GradientBoosting
- **Validierung:** StratifiedGroupKFold (über ganze Recordings)
- **Fenster:** standardmäßig 25 Sekunden, Schrittweite 12 Sekunden; Kann konfiguriert werden

---

## 2. Installation

### Voraussetzungen

- Python 3.10 - 3.12  (Featuretools kann in 3.13 Probleme verursachen)

### Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

### Abhängigkeiten (requirements.txt)

- pandas
- numpy
- featuretools
- tsfresh
- scikit-learn
- joblib
- matplotlib

---

## 3. Schnellstart

```bash
# 1. Training (labels.lbl und data/ müssen vorhanden sein)
python train.py

# 2. Vorhersage (test_labels.lbl und trainierte Modelle erforderlich)
python predict.py
```

---

## 4. Projektstruktur

```
core/
├── config.py              # Zentrale Konfiguration
├── data.py                # Datenladen, Fensterbildung, Labels
├── features.py            # Feature-Extraktion (Featuretools + TSFresh)
├── train.py               # Trainings-Pipeline
├── predict.py             # Vorhersage-Pipeline
├── backend_adapter.py     # GUI-Schnittstelle
├── progress.py            # pipeline_progress.json für Frontend-Status
├── plots.py               # Grafiken (Konfusionsmatrix, Feature Importance, Accuracy, Vorhersage)
├── main.py                # Einstiegspunkt (train | predict)
├── labels.lbl             # Trainings-Labels (optional für CLI-Nutzung)
├── test_labels.lbl        # Test-Labels (optional für CLI-Nutzung)
├── data/                  # CSV-Recordings (optional für CLI-Nutzung)
├── artifacts/             # Modelle und Ergebnisse
│   ├── model_randomforest.joblib
│   ├── model_logreg.joblib
│   ├── model_gradientboosting.joblib
│   ├── ergebnis.txt
│   ├── pipeline_progress.json
│   ├── test_ergebnis_*.csv
│   └── plots/             # Unterordner mit Grafiken
│       ├── confusion/     # Konfusionsmatrix pro Modell
│       ├── importance/    # Feature Importance pro Modell
│       └── accuracy/      # Modell-Genauigkeit (Balkendiagramm)
├── requirements.txt
├── README.md              # Diese Dokumentation
└── SCHNITTSTELLEN_BESCHREIBUNG.md   # API für Tkinter-Frontend
```

---

## 5. Nutzung

### CLI (train.py / predict.py)

**Training:**
```bash
python train.py [--data-dir DIR] [--labels FILE] [--artifacts DIR] [--config PATH] [--cv-splits N] [--random-state N]
```

**Vorhersage:**
```bash
python predict.py [--data-dir DIR] [--test-labels FILE] [--artifacts DIR] [--config PATH]
```

### GUI-API (backend_adapter)

Für Tkinter- und andere GUI-Frontends siehe **SCHNITTSTELLEN_BESCHREIBUNG.md**.

```python
from backend_adapter import train, predict, get_config, set_config

ok, msg = train(data_dir="data", labels=pd.DataFrame(columns=["File","Label"]), artifacts_dir="artifacts")
ok, out, ergebnisse = predict(data_dir="data", test_labels_file=pd.DataFrame(columns=["File","Label"]), artifacts_dir="artifacts")
```

---

## 6. Datenformate

### Label-Datei

CSV mit Header, Trennzeichen Komma:

```
File,Label
recording_2026_02_10__15_31_22_florian.csv,florian
recording_2026_02_10__15_38_03_matthias.csv,matthias
```

- **File:** Dateiname (relativ zu `data_dir`) oder absoluter Pfad
- **Label:** Fahrer-ID

### CSV-Recording

Erforderliche Spalten:

| Spalte | Beschreibung |
|--------|--------------|
| `timestamp` | Zeitstempel (Sekunden) |
| `wheel_position` | Lenkradposition |
| `car0_throttle_position` | Gaspedal |
| `car0_brake_position` | Bremspedal |
| `car0_velocity_vehicle` | Fahrzeuggeschwindigkeit |
| `rot_vel` | Rotationsgeschwindigkeit (Format "x,y,z"; Y = Gierrate) |

Die zweite Zeile kann Einheiten enthalten und wird übersprungen.

---

## 7. Konfiguration

### config.json

Optional im Projekt-Root. Beispiel:

```json
{
  "data_dir": "data",
  "labels_file": "labels.lbl",
  "test_labels_file": "test_labels.lbl",
  "artifacts_dir": "artifacts",
  "models": ["randomforest", "logreg", "gradientboosting"],
  "feature_set": "both",
  "window_sec": 25,
  "step_sec": 12,
  "min_points": 300,
  "max_points": 500,
  "cv_splits": 5,
  "random_state": 42,
  "use_grid_search": true
}
```

### Wichtige Parameter

| Parameter | Standard | Beschreibung |
|-----------|----------|--------------|
| `window_sec` | 25 | Fenstergröße in Sekunden |
| `step_sec` | 12 | Schrittweite in Sekunden |
| `feature_set` | "both" | "featuretools" \| "tsfresh" \| "both" |
| `cv_splits` | 5 | Folds für Cross-Validation |

---

## 8. Grafiken (plots/)

Nach Training und Vorhersage werden Grafiken in `artifacts/plots/` erzeugt:

| Ordner | Inhalt |
|--------|--------|
| `confusion/` | Konfusionsmatrix pro Modell |
| `importance/` | Feature Importance (Top 30) pro Modell |
| `accuracy/` | Balkendiagramm der Modell-Genauigkeiten |

---

## Fehlerbehandlung

- **CLI:** Bei Fehlern wird `SystemExit` mit Fehlermeldung geworfen
- **backend_api:** Gibt `(False, fehlermeldung)` zurück statt Exceptions

Typische Fehler:
- Keine gültigen Labels gefunden
- Label-Datei nicht gefunden
- Modelle nicht gefunden (vor predict muss train ausgeführt worden sein)
- Ungültige oder fehlende CSV-Spalten
