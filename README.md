# DriveIdent - Anwendung zur Fahrererkennung aus Fahrsimulator-Telemetriedaten

Fahrererkennung anhand von Fahrsimulator-Daten über eine simple graphische Nutzeroberfläche. Das System klassifiziert den Fahrer aus CSV-Recordings mittels maschinellem Lernen (RandomForest, LogisticRegression, GradientBoosting), siehe [Backend-Readme](./lib/core/README.md).

---

## Inhaltsverzeichnis

1. [Überblick](#1-überblick)
1. [Installation](#2-installation)
1. [Projektstruktur](#3-projektstruktur)
1. [Nutzung / Build](#4-nutzung--build)
1. [Datenformate](#5-datenformate)
1. [Konfiguration der Voreinstellungen](#6-konfiguration-der-voreinstellungen)

---

## 1. Überblick

### Funktionsumfang

| Funktion | Beschreibung |
|----------|--------------|
| **Training** | Trainiert Machine-Learning-Modelle auf gelabelte Recordings (CSV-Datensätze) |
| **Label-Verwaltung** | Labeln von Recordings; Exportieren und Importieren von Label-Dateien |
| **Konfiguration** | Konfiguration der Trainingsmechanismen und Auswahl von Modellen und Feature-Algorithmen |
| **Modell-Evaluation** | Auswertung der trainierten Modelle mithilfe von Konfusionsmatrizen und Feature-Importance-Diagrammen |
| **Vorhersage** | Klassifiziert unbekannte Recordings mit trainierten Modellen aus bekannten Fahrern |

---

## 2. Installation

### Python (Source-Code)

- **Download:** Source-Code aus diesem Repository klonen
- **Voraussetzung:** Python 3.10 - 3.12  (Featuretools kann in 3.13 Probleme verursachen)
- **Abhängigkeiten:** `pip install -r requirements.txt`, oder siehe [requirements.txt](./requirements.txt)

### Ausführbare Datei (.exe) für Windows

- **Download:** [DriveIdent.exe hier herunterladen](#)

---

## 3. Projektstruktur

```
DriveIdent/
├── config.json            # Konfiguration der Voreinstellungen
├── DriveIdent.py          # Haupt-Datei / Einstiegspunkt
├── lib/                   
│   ├── components/        # Frontend-Komponenten
│   ├── core/              # Backend (siehe lib/core/README.md)
│   └── windows/           # Frontend-Fenster
│       └── frames/        # Frontend-Ansichten
├── requirements.txt       # Python-Abhängigkeiten
└── README.md              # Diese Dokumentation
```

---

## 4. Nutzung / Build

### Source-Code

**Ausführen:**
```bash
python -m DriveIdent.DriveIdent
```

**Bauen zur .exe:**
```bash
pyinstaller --windowed --onefile --collect-all tsfresh --collect-all featuretools --collect-all sklearn --collect-all matplotlib --collect-all numba --collect-all woodwork --collect-all llvmlite --collect-all stumpy DriveIdent/DriveIdent.py
```

### Ausführbare Datei (DriveIdent.exe)

- Doppelklick zum Starten
- Nach kurzer Verzögerung öffnet sich das GUI
- Auswählen von Trainingsdaten
- Labeln der Trainingsdaten (durch editieren der Tabellen-Zellen, oder Import einer Label-Datei)
- Konfigurieren des Trainings
- Trainieren
- Auswählen von Recordings zur Vorhersage
- Vorhersagen

---

## 5. Datenformate

### Label-Datei

CSV mit Header, Trennzeichen Komma:

```
File,Label
recording_2026_02_10__15_31_22_florian.csv,florian
recording_2026_02_10__15_38_03_matthias.csv,matthias
```

- **File:** Dateiname
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

## 6. Konfiguration der Voreinstellungen

### config.json

Optional im Projekt-Root. Definiert die voreingestellten Optionen zum Modelltraining beim Anwendungsstart. Beispiel:

```json
{
  "artifacts_dir": "artifacts",
  "models": ["randomforest", "logreg", "gradientboosting"],
  "feature_set": "both",
  "window_sec": 25,
  "step_sec": 12,
  "min_points": 300,
  "max_points": 500,
  "cv_splits": 5,
  "use_grid_search": true
}
```

---
