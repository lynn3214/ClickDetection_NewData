# 1. Issues from Last Report

In the previous report, I implemented a CNN model, but there were two main issues:

1.  **The model could not run on the other dataset.**
    * The primary reason was a messy code logic. Initially, the training set input was set as a single 120 ms click segment. Later, it was modified to a 200 ms click train composed of multiple single clicks, while the test set input was a full-length `.wav` file with variable duration. Additionally, the corresponding configuration file was not provided, resulting in a mismatch between the model input size and data format.

2.  **Clipping occurred when clicks and noise were combined.**
    * In the previous code, single clicks, click trains, and the combined outputs were all normalized separately, and noise files were also normalized during preprocessing. Since different normalization methods (RMS and peak normalization) were used at different stages, the resulting combined signals had abnormal amplitudes.

This week, the main focus was to fix these two issues and systematically improve the workflow.

---

# 2. Main Work This Week

## 2.1 Data Preprocessing Module

### (1) Audio Standardization
* Resample to 44.1 kHz
* Band-pass filter (2–20 kHz)
* Support multiple formats (`.wav`, `.mat`, `.pkl`)
* Processed data:
    * Training set: `data/train_resampled/`
    * Test set: `data/test_resampled/`
    * Noise set: `data/noise_resampled/`

### (2) Noise Segmentation
* Split long noise files into 500 ms segments
* Split training/testing set with 8:2 ratio
* Ensure file-level independence

---

## 2.2 Rule-based Detector

### (1) Feature Engineering
* Implemented features:
    * TKEO (Teager-Kaiser Energy Operator)
    * STE (Short-Time Energy)
    * HFC (High-Frequency Content)
    * Spectral Centroid
    * High/Low Frequency Energy Ratio
    * Envelope analysis (width, kurtosis)

### (2) Snapping Shrimp Filtering
* Transient analysis (peak sharpness, rise time)
* Energy concentration detection
* Dolphin click likelihood scoring

### (3) Batch Detection Functionality
* Recursive directory scanning
* Automatic extraction of fixed-length 500 ms segments
* Output CSV statistics and corresponding audio clips

---

## 2.3 Training Dataset Construction

### (1) Click Train Synthesis
* Design principle: simulate realistic dolphin echolocation behavior
* Randomly combine 2–5 clicks
* ICI range: 10–80 ms (biologically plausible)
* Training set: 8000 sequences
* Validation set: 2000 sequences

### (2) SNR Mixing
* Simulate low SNR in real marine environments:
    * Add continuous background noise (non-transient)
    * SNR range: −5 dB to 15 dB (random sampling)
    * RMS normalization ensures consistent power
* Key improvements:
    * Noise pool preloaded (100 files, RMS = 0.1)
    * Accurate SNR calculation based on RMS power
    * Prevent clipping (peak limit 0.95)

### (3) Data Augmentation
* Time shift: ±10 ms
* Amplitude scaling: 0.8–1.25×
* Apply with 80% probability

### (4) Negative Sample Processing
* Randomly extract 500 ms pure noise segments
* RMS normalization (consistent with positive samples)
* Training/validation noise sources independent (8:2 split)
* Final dataset size:
    * **Training set:**
        * Positive: 8000 (click trains)
        * Negative: 8000 (pure noise)
        * Total: 16000
    * **Validation set:**
        * Positive: 2000
        * Negative: 2000
        * Total: 4000
* Segment length: 500 ms (22050 samples @ 44.1 kHz)

---

## 2.4 CNN Model Architecture and Training

### (1) Model Architecture
* Lightweight 1D-CNN (LightweightClickClassifier)
* ~45K parameters (~65% reduction from original)
* Input: `[batch, 22050]` (500 ms)
* Output: `[batch, 2]` (click / noise)
* Network structure:
    ```
    Input (22050)
      ↓
    Conv1D (kernel=7, stride=2) → 16 channels. #增加卷积层数，每层提取不同特征，数量减少通道数4/8；多层卷积，最后一层的window size 0.05s 重新考虑卷积核大小
      ↓
    3× Residual Blocks (16 → 32 → 64 → 128)#减少残差快
      ↓
    Global Average Pooling
      ↓
    Dropout (0.3) #用在靠前的layer中，而不是用在最后
      ↓
    Linear → 2 classes
    ```

### (2) Training Configuration
### CNN Model Architecture
- Type: Lightweight 1D-CNN (`LightweightClickClassifier`)
- Input: 500 ms (22050 samples)
- Classes: 2 (click / noise)
- Base channels: 16
- Residual blocks: 3
- Dropout: 0.3
- Parameter reduction: ~65% compared to original model

### Training Configuration
- Batch size: 128
- Max epochs: 100
- Learning rate: 0.001
- Weight decay: 0.0001
- Early stopping patience: 10 epochs
- Loss: CrossEntropyLoss (with class weights)
- LR scheduler: ReduceLROnPlateau

### Training Dataset Construction
- Click Train enabled: Yes
- Train samples: 8000 positive sequences
- Validation samples: 2000 sequences
- Sequence length: 500 ms
- Clicks per sequence: 2–5
- ICI range: 10–80 ms
- Noise: 500 ms background noise mixed with each sequence
- Data augmentation:
  - Time shift: ±10 ms
  - Amplitude scaling: 0.8–1.25×
  - Apply probability: 80%
- Negative samples: RMS-normalized 500 ms pure noise

---

## 2.5 Test Set Construction and Evaluation

### (1) Data Sources
* Singapore sea recordings + Sentosa low SNR recordings (independent from training set)
* Total duration: ~X seconds
* Covers various noise environments

### (2) Detector Extraction
```sh
python main.py batch-detect \
  --input-dir data/test_resampled \
  --output-dir data/test_detection_results \
  --config configs/detection_enhanced.yaml \
  --save-audio \
  --segment-ms 500  # consistent with training set
```

### (3) Manual Verification
Use Audacity to visualize and check segments

Confirm detector output accuracy

Minimal false positives, no removal required

### (4) Model Evaluation
```
python main.py eval-wav \
  --checkpoint models/checkpoints/best_model.pt \
  --positive-dir data/test_positive_segments \
  --negative-dir data/noise_test_segs \
  --output-dir reports/eval_wav_results \
  --config configs/eval_wav.yaml
```

- Test set results    

\============================================================    
Model Evaluation Report - 500ms Segment Mode      
\============================================================

Model: models/v1.0/best_model.pt   
Positive Directory: data/test_detection_results/audio   
Negative Directory: data/noise_test_segs    
Confidence Threshold: 0.5   

Dataset Statistics:
  - Total Samples: 7348
  - Positive: 3358
  - Negative: 3990      
  
Evaluation Metrics:
  - Accuracy:  0.9784
  - Precision:  0.9825
  - Recall:  0.9699
  - F1 Score:  0.9762
  - ROC AUC: 0.9905
  - PR AUC:  0.9798
        
Confusion Matrix:
  - True Negative (TN): 3932
  - False Positive (FP): 58
  - False Negative (FN): 101
  - True Positive (TP): 3257

sklearn Classification Report:
```
               precision    recall  f1-score   support
    Negative       0.97      0.99      0.98      3990
    Positive       0.98      0.97      0.98      3358
    accuracy                           0.98      7348
   macro avg       0.98      0.98      0.98      7348
weighted avg       0.98      0.98      0.98      7348
```
## 2.6 Results & Analysis

* Evaluation metrics on Detector output:
    * Accuracy: 97.84%
    * Precision: 98.25%
    * Recall: 96.99%
    * F1 Score: 97.62%
    * ROC AUC: 99.05%
    * PR AUC: 97.98%

* Confusion matrix:
    ```
      TN: 3932   FP: 58  
      FN: 101    TP: 3257  
    ```
* Test set size: positive 3358, negative 3990, total 7348
* **Note:** Since the test set was pre-filtered by the Detector, there may be selection bias. Results reflect CNN classification performance only, not overall system recall.

---

# 3. Observed Issues and Limitations

* Test set was pre-filtered by Detector, introducing sample selection bias
* Extreme low SNR (< −5 dB) or overlapping clicks are not included
* Some negative samples overlap with training set sources, slightly overestimating performance

---

# 4. Next Steps

* **Evaluate Detector Recall**
    * Manually annotate 10 test recordings (~1 min audio)
    * Mark all identifiable dolphin clicks using Audacity
    * Save labels and compare with Detector output:
        ```sh
        python main.py detect \
          --input data/test_detector/*.wav \
          --config configs/detection.yaml
        ```
    * Calculate Detector accuracy and recall

* **Improve data diversity and robustness**
    * Add low SNR and overlapping click samples
    * Check consistency between training and test noise distributions



在训练集、测试集（sentosa）中的正样本中叠加负样本同来源的noise 0dB
分开测试集，圣淘沙的叠加noise，新加坡水域录音直接使用，因为已经叠加里背景噪音