# Dolphin Click Detector - Export Package

This `export/` folder contains the trained model, configuration files, and minimal code needed to **run inference and evaluation on new audio datasets**. No training data is included.

---

## 📁 Folder Structure

```
export/
│
├── checkpoints/                            
│   └── best.pt                            # Trained model weights
├── configs/
│   ├── eval_wav.yaml                      # Evaluation configuration
│   └── inference.yaml                     # Inference configuration
├── main.py                                # Main evaluation script
├── environment.yaml                       # Conda environment configuration
├── README.md                              # This document
│
├── models/                                # Model architecture
│   ├── __init__.py
│   └── cnn1d/
│       ├── __init__.py
│       ├── model.py                       # CNN model definition
│       └── inference.py                   # Inference wrapper
│
├── training/                              # Evaluation utilities
│   ├── __init__.py
│   ├── dataset/
│   │   ├── __init__.py
│   │   └── segments.py                    # Dataset construction (reference)
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py                     # Evaluation metrics
│   │   └── report.py                      # Report generation
│   └── augment/
│       ├── __init__.py
│       └── pipeline.py                    # Data augmentation (reference)
│
└── utils/                                 # Utility functions
    ├── __init__.py
    ├── logging/
    │   ├── __init__.py
    │   └── logger.py                      # Logging utilities
    └── metrics/
        ├── __init__.py
        └── events_tracks.py               # Event-level evaluation
```

---

## 🚀 Quick Start

### 1. Environment Setup

Create a fresh Conda environment:

```bash
conda env create -f environment.yaml
conda activate dolphin_click
```

Or install dependencies manually:
```bash
pip install torch numpy scipy scikit-learn soundfile librosa pyyaml pandas matplotlib tqdm
```

---

## 2. Prepare Test Data

### 2.1 Data Requirements

**Audio specifications:**
- **Format**: WAV (.wav)
- **Sampling rate**: Any (will be automatically resampled to 44.1 kHz)
- **Channels**: Mono or Stereo (will be automatically converted to mono)
- **Preprocessing**: Bandpass filtered (2-20 kHz) is recommended but not required
- **Segment length**: Should be 500ms clips

### 2.2 Directory Organization

Organize your test audio files in the following structure:

```
your_test_data/
├── positive/          # Audio segments containing dolphin clicks
│   ├── click_001.wav
│   ├── click_002.wav
│   └── ...
└── negative/          # Audio segments with noise or background
    ├── noise_001.wav
    ├── noise_002.wav
    └── ...
```

**Important notes:**
- Each WAV file should be a **500ms audio segment**
- Positive samples: segments known to contain dolphin clicks
- Negative samples: segments containing only noise or background sounds
- File names can be arbitrary

---

## 3. Run Evaluation

### 3.1 Basic Evaluation

Test with all available samples:

```bash
python main.py \
    --positive-dir your_test_data/positive \
    --negative-dir your_test_data/negative \
    --output-dir results
```

### 3.2 Custom Configuration

Use custom checkpoint and configuration files:

```bash
python main.py \
    --config configs/eval_wav.yaml \
    --checkpoint checkpoints/best.pt \
    --positive-dir your_test_data/positive \
    --negative-dir your_test_data/negative \
    --output-dir results/custom_test
```

---

## 4. Parameter Descriptions

### 4.1 Required Parameters

| Parameter          | Description                              | Example                      |
| ------------------ | ---------------------------------------- | ---------------------------- |
| `--positive-dir`   | Directory containing positive samples    | `data/positive`              |
| `--negative-dir`   | Directory containing negative samples    | `data/negative`              |

### 4.2 Optional Parameters

| Parameter          | Description                              | Default Value                |
| ------------------ | ---------------------------------------- | ---------------------------- |
| `--config`         | Path to configuration file               | `configs/eval_wav.yaml`      |
| `--checkpoint`     | Path to model checkpoint                 | `checkpoints/best.pt`        |
| `--output-dir`     | Output directory for results             | `results`                    |

### 4.3 Configuration File Settings

Edit `configs/eval_wav.yaml` to adjust evaluation settings:

```yaml
# Inference settings
inference:
  batch_size: 32          # Inference batch size (reduce if out of memory)
  device: cpu             # 'cpu' or 'cuda'
  sample_rate: 44100      # Audio sample rate (Hz)

# Classification threshold
thresholds:
  confidence_threshold: 0.5   # Binary classification threshold (0-1)
                              # Adjust higher for fewer false positives
                              # Adjust lower for fewer false negatives

# Output settings
output:
  save_predictions: true              # Save per-file predictions
  save_misclassified_files: true      # Save misclassified file list
  save_confusion_matrix: true         # Save confusion matrix plot
  save_roc_curve: true                # Save ROC curve
  save_pr_curve: true                 # Save Precision-Recall curve
  generate_detailed_report: true      # Generate detailed HTML report
```

---

## 5. Output Results

After evaluation, the output directory will contain:

```
results/
├── predictions.csv                 # Per-file predictions with confidence scores
├── misclassified.csv              # List of misclassified files
├── confusion_matrix.png           # Confusion matrix visualization
├── roc_curve.png                  # ROC curve
├── pr_curve.png                   # Precision-Recall curve
├── evaluation/                    # Detailed evaluation report
│   ├── metrics.json               # Metrics in JSON format
│   ├── metrics.txt                # Human-readable metrics
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── threshold_analysis.png     # Threshold vs. F1-score analysis
│   └── optimal_threshold.json     # Optimal threshold recommendation
└── DolphinClickDetection_YYYYMMDD_HHMMSS.log  # Execution log
```

### 5.1 Example: predictions.csv

| file_id      | true_label | predicted_label | confidence |
| ------------ | ---------- | --------------- | ---------- |
| click_001    | 1          | 1               | 0.9234     |
| click_002    | 1          | 1               | 0.8756     |
| noise_001    | 0          | 0               | 0.1234     |
| noise_002    | 0          | 1               | 0.6543     |

**Column descriptions:**
- `file_id`: Audio file name (without .wav extension)
- `true_label`: Ground truth label (1=click, 0=noise)
- `predicted_label`: Model prediction (1=click, 0=noise)
- `confidence`: Confidence score for positive class (0-1)

### 5.2 Example: misclassified.csv

Only contains files where `predicted_label ≠ true_label`, useful for error analysis.

### 5.3 Console Output

During evaluation, you'll see:

```
======================================================================
Dolphin Click Detector - Model Evaluation
======================================================================
Config file: configs/eval_wav.yaml
Model checkpoint: checkpoints/best.pt
Device: cpu
Batch size: 32
Sample rate: 44100 Hz

Loading model...
✓ Model loaded successfully

Loading test data...
Positive samples directory: your_test_data/positive
Negative samples directory: your_test_data/negative
Loading positive: 100%|████████████| 150/150 [00:05<00:00]
Loading negative: 100%|████████████| 150/150 [00:05<00:00]
✓ Positive samples: 150
✓ Negative samples: 150

Starting inference...
✓ Inference complete
Confidence threshold: 0.5

======================================================================
Evaluation Results
======================================================================
Accuracy:  0.9533
Precision: 0.9467
Recall:    0.9600
F1-Score:  0.9533
ROC AUC:   0.9876
PR AUC:    0.9845

Confusion Matrix:
  True Negative (TN): 142    False Positive (FP): 8
  False Negative (FN): 6     True Positive (TP): 144

✓ Predictions saved: results/predictions.csv
✓ Misclassified list saved: results/misclassified.csv
✓ Confusion matrix saved: results/confusion_matrix.png
✓ ROC curve saved: results/roc_curve.png
✓ PR curve saved: results/pr_curve.png
✓ Detailed report saved: results/evaluation

======================================================================
Evaluation complete!
All results saved to: results
======================================================================
```

---

## 6. Model Information

- **Architecture**: Lightweight 1D CNN with residual blocks
- **Input**: 500ms audio segments (22,050 samples @ 44.1 kHz)
- **Output**: Binary classification (dolphin click vs. noise)
- **Parameters**: ~65% fewer than full model for efficient inference
- **Training**: Trained on dolphin echolocation clicks with data augmentation

---

### Optional: Audio Preprocessing Script

A reference preprocessing script is provided in `utils/preprocess_audio.py`:
```bash
python utils/preprocess_audio.py \
    --input raw_audio.wav \
    --output-dir data/processed
```

**Note**: This is a reference implementation. You may need to adapt it to your specific data pipeline.