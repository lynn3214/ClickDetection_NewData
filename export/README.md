# Dolphin-Click Detector – Export Package

This `export/` folder contains the newly trained model, configuration, and minimal code needed to **run inference on new audio clips**. No training data is included.

---

## Folder Structure

```
export/
│
├── checkpoints/                            
│   └── best_model.pt                          # trained model weights
├── eval_wav_files.py                      # test script
├── environment.yaml                       # Conda environment configuration
├── README.md                              # usage instructions
│
├── models/                                # model definition
│   ├── __init__.py
│   └── cnn1d/
│       ├── __init__.py
│       ├── model.py
│       └── inference.py
│
├── training/                              # evaluation utilities
│   ├── __init__.py
│   ├── dataset/
│   │   ├── __init__.py
│   │   └── segments.py
│   └── eval/
│       ├── __init__.py
│       ├── metrics.py
│       └── report.py
│
└── utils/                                 # utility functions
    ├── __init__.py
    └── logging/
        ├── __init__.py
        └── logger.py
```

---

## 1. Environment Setup

If is necessary to create a fresh Conda environment:

```bash
conda env create -f environment.yaml
conda activate dolphin_click
```

*(Alternatively, install the packages listed in environment.yml using pip.)*

## 2. Prepare Test Data
Organize the test audio files in the following structure:
```
your_test_data/
├── positive/          # audio files with dolphin clicks
│   ├── rec001.wav
│   ├── rec002.wav
│   └── ...
└── negative/          # audio files with pure noise or background
    ├── noise001.wav
    ├── noise002.wav
    └── ...

```

Requirements:
- Format: WAV (.wav)
- Sampling rate: Any (automatically resampled to 44.1 kHz)
- Channel: Mono or Stereo (automatically converted to mono)
- Duration: No limit

## 3. Run Inference on Test Data
#### 3.1 Test with All Samples

```bash
python eval_wav_files.py \
    --checkpoint checkpoints/best_model.pt \
    --positive-dir your_test_data/positive \
    --negative-dir your_test_data/negative \
    --output-dir results/full_test

```

#### 3.2 Balance Test Set (1:1 Positive:Negative Ratio)
To ensure a balanced test set, use the `--balance` flag:

```bash
python eval_wav_files.py \
    --checkpoint checkpoints/best_model.pt \
    --positive-dir your_test_data/positive \
    --negative-dir your_test_data/negative \
    --output-dir results/balanced_test \
    --negative-sampling balance

```

#### 3.3 Custom Positive:Negative Ratio (e.g., 1:2)
To test with a custom ratio, use the `--pos-neg-ratio` flag with a ratio string (e.g., `1:2`):

```bash
python eval_wav_files.py \
    --checkpoint checkpoints/best_model.pt \
    --positive-dir your_test_data/positive \
    --negative-dir your_test_data/negative \
    --output-dir results/ratio_test \
    --negative-sampling ratio \
    --pos-neg-ratio 1:2
```

#### 3.4 Limit Negative Samples
To limit the number of negative samples, use the `--max-negative-samples` flag:
```
python eval_wav_files.py \
    --checkpoint checkpoints/best_model.pt \
    --positive-dir your_test_data/positive \
    --negative-dir your_test_data/negative \
    --output-dir results/limited_test \
    --negative-sampling max \
    --max-negative-samples 1000
```

## 4. Parameter Descriptions

#### 4.1 Required Parameters
| Parameter        | Description        | Example                          |
| ---------------- | ----------------- | ------------------------------ |
| `--checkpoint`   | directory containing model weights | `checkpoints/best_model.pt` |
| `--positive-dir` | directory containing positive samples | `data/positive`             |
| `--negative-dir` | directory containing negative samples | `data/negative`             |
| `--output-dir`   | output directory for results | `results/test_01`           |

#### 4.2 Optional Parameters
| Parameter             | Description               | Default Value |
| ---------------------- | ------------------------- | ------------- |
| `--file-threshold`     | threshold for file-level click detection | 0.5 |
| `--min-positive-ratio` | minimum ratio of positive windows for a file to be classified as "clicking" | 0.1 |
| `--device`             | device for inference (cpu or cuda) | cpu |

#### 4.3 Negative Sampling Parameters
| Parameter             | Description               | Default Value |
| ---------------------- | ------------------------- | ------------- |
| `--negative-sampling`    | negative sampling strategy | `none`(do not sample negative) / `balance`(balance positive and negative) / `ratio`(ratio sampling) / `max`(max sampling) |
| `--pos-neg-ratio`        | positive:negative sample ratio (only for ratio and max sampling) | `"1:2"`                                                 |
| `--max-negative-samples` | maximum number of negative samples (only for max sampling) | 1000                                                    |
| `--random-seed`          | random seed for negative sampling (for reproducibility) | 42                                                      |


## 4. Output Results
After testing, the output directory will contain the following files:
```
results/test_01/
├── file_level_results.csv          # every file's click probability
└── wav_file_evaluation/            # evaluation reports and charts
    ├── metrics.json
    ├── metrics.txt
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── pr_curve.png
    ├── threshold_analysis.png
    └── optimal_threshold.json
```

Example: file_level_results.csv
| file_name    | true_label | predicted_label | mean_prob | positive_ratio |
| ------------ | ---------- | --------------- | --------- | -------------- |
| rec001.wav   | 1          | 1               | 0.8234    | 0.35           |
| noise001.wav | 0          | 0               | 0.1234    | 0.02           |

Description:
- true_label: true label (1=clicking, 0=non-clicking)
- predicted_label: model's prediction (1=clicking, 0=non-clicking)
- mean_prob: average confidence score (closer to 1 means higher probability of clicking)
- positive_ratio: ratio of positive windows (above `min-positive-ratio` is classified as clicking)
