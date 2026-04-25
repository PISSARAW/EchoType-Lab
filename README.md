# EchoType Lab 🎹🔊

> **Educational side-channel research** – classify keyboard keystrokes from their
> acoustic signature using machine learning.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ⚠️ Ethical Considerations & Disclaimer

> **Read this section carefully before using or modifying EchoType Lab.**

This project is an **educational demonstration** of acoustic side-channel
analysis.  It is designed to show researchers and students how keystroke
sounds can carry information about which keys are pressed – a well-studied
phenomenon in the academic security literature (cf. Zhuang *et al.*, 2009;
Berger *et al.*, 2006).

### Permitted uses
- Studying and understanding acoustic side channels on **your own hardware**.
- University or research-lab demonstrations with the **full consent** of every
  participant.
- Security awareness training to motivate defence (noise-cancelling keyboards,
  white noise, soundproofing, etc.).

### Prohibited uses
- Recording or analysing keystrokes of **any person who has not explicitly
  consented**.
- Deploying the tool on systems you do not own.
- Any use that may violate local wiretapping, eavesdropping, computer-fraud, or
  privacy laws.

**The authors accept no liability for misuse.  You use this software at your
own risk and are solely responsible for compliance with applicable laws.**

---

## Table of Contents

1. [Project overview](#project-overview)
2. [Directory layout](#directory-layout)
3. [Installation](#installation)
4. [Quick start](#quick-start)
5. [Modules](#modules)
6. [Running tests](#running-tests)
7. [Limitations](#limitations)
8. [References](#references)
9. [License](#license)

---

## Project overview

EchoType Lab records short audio clips from a microphone while the user types
on their **own** keyboard, extracts Mel-frequency Cepstral Coefficient (MFCC)
features with [librosa](https://librosa.org/), and trains a classifier
([scikit-learn](https://scikit-learn.org/) Random Forest or a small
TensorFlow neural network) to predict which key was pressed.

The full pipeline:

```
Microphone  →  AudioRecorder  →  WAV files (data/raw/)
                                        │
                              MfccExtractor (librosa)
                                        │
                              Feature matrix (X, y)
                                        │
                          SklearnClassifier / TFClassifier
                                        │
                         Evaluation metrics + plots
```

All data stays on **your local machine**.  Nothing is sent to any server.

---

## Directory layout

```
EchoType-Lab/
├── echotype_lab/               # Main Python package
│   ├── __init__.py
│   ├── consent.py              # User-consent prompt & logging
│   ├── audio/
│   │   ├── __init__.py
│   │   └── recorder.py         # AudioRecorder, KeystrokeRecorder
│   ├── features/
│   │   ├── __init__.py
│   │   └── mfcc.py             # MFCC extraction (librosa)
│   ├── models/
│   │   ├── __init__.py
│   │   └── classifier.py       # SklearnClassifier, TFClassifier
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py          # Accuracy, confusion matrix, F1 …
│   └── visualization/
│       ├── __init__.py
│       └── plots.py            # Waveform, spectrogram, confusion-matrix plots
├── scripts/
│   ├── record_keystrokes.py    # CLI: collect labelled audio clips
│   └── train_model.py          # CLI: train & evaluate classifier
├── tests/
│   ├── conftest.py
│   ├── test_audio.py
│   ├── test_evaluation.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_package.py
│   └── test_visualization.py
├── data/
│   ├── raw/                    # WAV clips (gitignored)
│   ├── processed/              # Feature arrays (gitignored)
│   └── models/                 # Saved models (gitignored)
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Installation

### Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.9 |
| PortAudio | system library (required by PyAudio) |

Install PortAudio on common platforms:

```bash
# Ubuntu / Debian
sudo apt-get install portaudio19-dev

# macOS (Homebrew)
brew install portaudio

# Windows – usually bundled with the PyAudio wheel
```

### Python dependencies

```bash
# Clone the repository
git clone https://github.com/PISSARAW/EchoType-Lab.git
cd EchoType-Lab

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install runtime dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .

# Optional: TensorFlow backend
pip install -e ".[tensorflow]"
```

---

## Quick start

### 1 – Record keystroke audio

```bash
# Record 20 clips for each of the keys: a, b, c, space
echotype-record --keys a b c space --n 20

# Or run the script directly
python -m scripts.record_keystrokes --keys a b c --n 30 --output data/raw
```

The script will:
1. Display a **consent notice** and ask you to confirm.
2. Prompt you to press each key in turn while it records 0.5 s audio clips.
3. Save WAV files to `data/raw/<key>/<key>_NNNN.wav`.

### 2 – Train a classifier

```bash
echotype-train --data data/raw --model sklearn --plot-cm data/confusion_matrix.png
```

This will:
1. Extract MFCC features from all WAV files.
2. Split 80 % / 20 % for training and testing.
3. Train a Random Forest classifier.
4. Print a classification report.
5. Save the model to `data/models/classifier.pkl`.
6. Save a confusion-matrix plot (if `--plot-cm` is specified).

### 3 – Use the Python API directly

```python
import numpy as np
from echotype_lab.audio.recorder import AudioRecorder
from echotype_lab.features.mfcc import extract_mfcc_from_array
from echotype_lab.models.classifier import SklearnClassifier
from echotype_lab.evaluation.metrics import print_report

# Record a short clip (requires microphone)
rec = AudioRecorder(clip_duration=0.5)
samples = rec.record_clip()

# Extract features
feats = extract_mfcc_from_array(samples)
print("Feature vector shape:", feats.shape)  # (78,)

# Train (assuming you have X_train, y_train, labels from extract_mfcc_batch)
clf = SklearnClassifier(labels=["a", "b", "c"])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print_report(y_test, y_pred, labels=["a", "b", "c"])
```

---

## Modules

### `echotype_lab.consent`

Manages user consent before any audio is collected.

| Function | Description |
|----------|-------------|
| `request_consent(log_path, non_interactive)` | Display notice; return `True` if accepted |

### `echotype_lab.audio.recorder`

| Class / Function | Description |
|------------------|-------------|
| `AudioRecorder` | Low-level PyAudio wrapper; `record_clip()` → `np.ndarray` |
| `KeystrokeRecorder` | High-level: links labels to clips, manages file naming |

### `echotype_lab.features.mfcc`

| Function | Description |
|----------|-------------|
| `extract_mfcc(path)` | Feature vector from a WAV file |
| `extract_mfcc_from_array(samples)` | Feature vector from a NumPy array |
| `extract_mfcc_batch(data_dir)` | Returns `(X, y, labels)` for a labelled dataset |

Feature vector composition (default `n_mfcc=13`):
`[MFCC_mean, MFCC_std, Δ_mean, Δ_std, ΔΔ_mean, ΔΔ_std]` → **78 features**.

### `echotype_lab.models.classifier`

| Class | Backend | Description |
|-------|---------|-------------|
| `SklearnClassifier` | scikit-learn | Random Forest (default); any sklearn estimator accepted |
| `TFClassifier` | TensorFlow/Keras | Small dense network; optional dependency |

Both expose: `fit(X, y)`, `predict(X)`, `predict_proba(X)`, `save(path)`, `load(path)`.

### `echotype_lab.evaluation.metrics`

| Function | Description |
|----------|-------------|
| `compute_accuracy(y_true, y_pred)` | Float in [0, 1] |
| `compute_confusion_matrix(y_true, y_pred)` | 2-D integer array |
| `classification_report_dict(y_true, y_pred, labels)` | Per-class precision/recall/F1 |
| `top_k_accuracy(y_true, y_proba, k)` | Top-k accuracy |
| `print_report(y_true, y_pred, labels)` | Pretty-print report |

### `echotype_lab.visualization.plots`

| Function | Description |
|----------|-------------|
| `plot_waveform(samples)` | Time-domain amplitude plot |
| `plot_spectrogram(samples)` | Mel-spectrogram |
| `plot_mfcc(samples)` | MFCC heatmap |
| `plot_confusion_matrix(cm, labels)` | Annotated heatmap |
| `plot_class_distribution(y, labels)` | Bar chart of class counts |

All functions return a `matplotlib.figure.Figure` and accept optional
`save_path` and `show` keyword arguments.

---

## Running tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run the full test suite
pytest

# Run with coverage report
pytest --cov=echotype_lab --cov-report=term-missing

# Run a specific test file
pytest tests/test_evaluation.py -v
```

Tests that require a microphone or GPU are **not** included; all tests run
without audio hardware using synthetic NumPy arrays.

---

## Limitations

| Limitation | Details |
|------------|---------|
| **Accuracy** | Real-world accuracy is highly dependent on keyboard model, microphone quality, room acoustics, and typing style.  Academic papers report 40–96 % top-1 accuracy depending on conditions. |
| **Generalisation** | A model trained on one keyboard / environment typically does **not** generalise to another without retraining. |
| **Key coverage** | Distinguishing all 104 keys is much harder than a small subset.  Start with 5–10 distinct-sounding keys. |
| **Data quantity** | At least 50–100 clips per key are recommended for reasonable performance. |
| **Background noise** | Loud environments significantly degrade accuracy; consider a directional microphone. |
| **Streaming / real-time** | This skeleton performs offline batch classification.  Real-time inference requires additional buffering logic not included here. |
| **OS / hardware** | Relies on PortAudio via PyAudio; may require system-level audio permissions on macOS / Linux. |

---

## References

1. Zhuang, L., Zhou, F., & Tygar, J. D. (2009).
   *Keyboard acoustic emanations revisited.*
   ACM Transactions on Information and System Security, 13(1), 1–26.

2. Berger, Y., Wool, A., & Yeredor, A. (2006).
   *Dictionary attacks using keyboard acoustic emanations.*
   Proceedings of CCS 2006.

3. Halevi, T., & Saxena, N. (2012).
   *A closer look at keyboard acoustic emanations.*
   Proceedings of ASIACCS 2012.

---

## License

MIT – see [LICENSE](LICENSE) for details.

> **Reminder:** Capability does not imply permission.  Always obtain informed
> consent and respect applicable laws before recording audio near others.
