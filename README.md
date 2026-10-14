# 🔍 Distorted Image Text Recognition — CRNN + CTC Loss

A deep learning pipeline built from scratch to recognize text sequences from heavily distorted grayscale images using a custom CRNN architecture trained with CTC Loss.

---

## 📌 Problem

Given grayscale images where text is hidden under heavy distortion — noise, blur, overlapping symbols, occlusion, and irregular spacing — the model must correctly predict the character sequence inside each image.

Evaluated using **Character Error Rate (CER)** — lower is better.

---

## 🏗️ Architecture
```
Input Image [1 × 32 × 128]
        ↓
CNN Block 1 — Conv(1→32) + BatchNorm + ReLU + MaxPool(2,2)   →  [32 × 16 × 64]
        ↓
CNN Block 2 — Conv(32→64) + BatchNorm + ReLU + MaxPool(2,2)  →  [64 × 8 × 32]
        ↓
CNN Block 3 — Conv(64→128) + BatchNorm + ReLU + MaxPool(2,1) →  [128 × 4 × 32]
        ↓
Reshape → Sequence of 32 feature vectors  [32 × 512]
        ↓
Linear Projection                          [32 × 256]
        ↓
Bidirectional LSTM — 2 layers, hidden=256  [32 × 512]
        ↓
Linear + Log Softmax                       [32 × vocab+1]
        ↓
CTC Loss (training)  /  Greedy Decoder (inference)
        ↓
Predicted Text
```

---

## ⚙️ Key Details

| Component | Details |
|---|---|
| CNN Backbone | 3 conv blocks, channels: 1 → 32 → 64 → 128 |
| Sequence Model | 2-layer BiLSTM, hidden size 256 |
| Loss Function | CTC Loss (blank index = 0, zero_infinity = True) |
| Optimizer | Adam (lr=0.001, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Preprocessing | CLAHE + Resize 128×32 + Normalize [0, 1] |
| Regularization | Dropout(0.3) + Gradient Clipping (max_norm=5.0) |
| Batch Size | 32 |
| Epochs | 20 |
| Evaluation | Character Error Rate (CER) |

---

## 🐛 Key Bug Fixed — Model Collapse

The model was predicting **"9X" for every single image** — including training images it had already seen. This is called model collapse.

**Root Cause:**
`input_lengths` passed to CTC Loss was hardcoded without verifying against the actual CNN output width. CTC computed wrong alignment probabilities which produced garbage gradients. The model found the path of least resistance — always predicting the same token.

**Fix:**
Derived T = 32 by tracing the CNN architecture manually:


Input width 128 → Pool(2,2) → 64 → Pool(2,2) → 32 → Pool(2,1) → 32


Also added BatchNorm after every conv layer, gradient clipping, and reduced batch size from 64 to 32 to stabilize CTC training.

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
```

Place `train_images/`, `test_images/`, and `train-labels.csv` in the root folder, then run all cells in `distorted_images_Recognition.ipynb`.

---

## 📁 Dataset

Dataset is part of the **CIG AI/ML Problem Statement** Open project and is not included in this repository.

---

## 🛠️ Tech Stack

`Python` &nbsp; `PyTorch` &nbsp; `OpenCV` &nbsp; `NumPy` &nbsp; `Pandas` &nbsp; `scikit-learn`

