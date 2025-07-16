# 🔊 Transformer-Based Audio Classification from Scratch | 1D CNN + Transformer Encoder

> **Name:** Sahil Sharma  
> **Program:** M.Sc - M.Tech (Data and Computational Sciences)  
> **Institute:** Indian Institute of Technology, Jodhpur  

---

## 📘 Overview

This project is a **comprehensive implementation of a Transformer-based deep learning model** for **multi-class classification of audio signals**. The goal was to design both a **1D CNN-based classifier** and a **Transformer Encoder with multi-head attention mechanism**, entirely from scratch, without relying on high-level libraries.

This assignment demonstrates:
- Deep understanding of **Transformer internals**
- Implementation of a **custom multi-head attention mechanism**
- Integration of CNN-based feature extraction and Transformer sequence modeling
- **End-to-end classification pipeline for audio waveform data**

---

## 🎯 Objectives

- Develop a **1D Convolutional Neural Network** (CNN) architecture to extract rich local features from audio signals.
- Build a **Transformer Encoder** from scratch, including:
  - Positional encodings
  - Multi-head attention layers
  - Feedforward layers
  - Layer normalization and residual connections
- Combine CNN + Transformer to perform robust **multi-class classification**.
- Train models using **k-fold cross-validation** and visualize results with **WandB**.
- Evaluate the models using standard metrics: **Accuracy, Confusion Matrix, F1 Scores, and ROC Curves**.

---

## 🎧 Dataset Description

- **Type:** Raw Audio Waveforms  
- **Total Samples:** 400  
- **Classes:** 10 (labeled from 0 to 9)  
- **Samples per Class:** 40 (balanced)  
- **Waveform Shape:** `[1, 220500]`  
- **Input Shape for Model:** `[batch_size, 1, 16000]` after preprocessing  

---

## 📊 Exploratory Data Analysis (EDA)

- Visualized **waveform samples** across classes to understand signal structure
- Analyzed **class distribution** (perfectly balanced)
- Validated **shape consistency** for model input
- Observed **temporal resolution and variation** across audio samples

---

## 🧱 Architecture 1 – 1D CNN Classifier (Baseline)

The first architecture was designed to perform classification using a **three-layer 1D CNN** with **ReLU activations** and a **final fully connected softmax output layer**.

### ✅ Key Features:

- Effective use of **convolutions** to extract local patterns
- **MaxPooling and AvgPooling** layers to downsample and retain signal essence
- Final **Linear layer + Softmax** to classify into 10 categories
- Trained with **Cross-Entropy Loss** and **Adam optimizer (lr = 0.001)**
- **100 epochs** of training with **k-fold validation (k=4)**

### 🧮 Layer Breakdown:

| Layer Type          | Configuration                                   |
|---------------------|--------------------------------------------------|
| Conv1               | In: 1 → Out: 32, kernel: 3×3, stride: 1, padding: 1 |
| Conv2               | In: 32 → Out: 64, same kernel/stride/padding     |
| Conv3               | In: 64 → Out: 128, same kernel/stride/padding    |
| Pooling             | MaxPool → MaxPool → AvgPool (2×2, stride=2)      |
| Fully Connected     | Flatten → Linear (64×4000 → 10), Softmax         |
| Parameters          | 2.59 Million Trainable Parameters                |

### 📈 Performance:

- **Training Accuracy:** 99.58%
- **Validation Accuracy:** 52.50%
- **Evaluation Tools:** Confusion Matrix, F1 Scores, ROC Curves

---

## 🚀 Architecture 2 – Transformer Encoder with Multi-Head Self-Attention

This is the core innovation of the project — a complete **Transformer Encoder architecture built from scratch**, fully modularized and integrated with the CNN backbone.

### ✅ Key Innovations:

- Implemented:
  - `MultiHeadSelfAttention` with Q-K-V computations
  - `PositionalEncoding` for sequence-awareness
  - `EncoderBlock` with residual connections
  - `TransformerClassifier` using CLS token logic
- Transformer input derived from CNN embeddings
- Used **CLS token representation** for final classification
- Applied **3 self-attention heads** in multi-head attention module
- **Fully connected classifier** on top of encoder output

### 🔧 Training Setup:

- CNN backbone used for feature extraction
- Embeddings passed into transformer
- **Positional Encoding** + **CLS token** prepended
- Trained for **100 epochs** with **Adam optimizer**
- Classification using final CLS embedding

### 🧩 Components Overview:

- `CLS Token` → Learnable embedding prepended to inputs
- `Positional Encoding` → Injects sequence order into embeddings
- `Multi-Head Attention` → 3 parallel attention heads
- `Encoder Layers` → Stacked self-attention + feedforward + residual
- `TransformerClassifier` → Extracts classification logits from CLS token

---

## 📊 Model Evaluation & Visualizations

| Metric                | CNN Baseline (Arch 1) |
|-----------------------|------------------------|
| Training Accuracy     | 99.58%                |
| Parameters            | 2.59M                 |
| ROC / F1 / CM Plotted | ✅                    |

| Metric                     | Transformer (Head = 1/2/4) |
|----------------------------|----------------------------|
| Components implemented     | ✅ Fully from scratch       |
| Transformer train accuracy | Functional (12–13%)        |
| Evaluation tools used      | ✅ ROC, F1, Confusion       |
| Model tuning complete?     | 🔄 Work in progress         |

🧠 **Note:** While the Transformer results are a work in progress, the architectural components and modular implementation form a **robust foundation for further research, fine-tuning, and experimentation**.

---

## 🧪 Comparative Summary

| Model        | Train Accuracy | Val Accuracy | Parameters |
|--------------|----------------|--------------|------------|
| CNN (Arch 1) | 99.58%         | 52.50%       | 2.59M      |
| Transformer (Heads=1/2/4) | ~13% | ~10% | 48.4M |

> The Transformer model, while complex, shows promise with future tuning and dataset scaling.

---

## 🛠️ Tools & Techniques Used

- **Framework:** PyTorch  
- **Training Monitor:** WandB  
- **Optimizer:** Adam  
- **Loss Function:** Cross Entropy  
- **Regularization:** Dropout, CLS token optimization  
- **Evaluation:** Accuracy, Confusion Matrix, F1 Scores, ROC-AUC  
- **Training Strategy:** 100 Epochs, k-Fold Cross-Validation  

---

## 🔬 Resources & References

- [Attention Is All You Need – Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762)  
- [PyTorch NLP Tutorial](https://pytorch.org/tutorials/beginner/nlp/pytorch_tutorial.html)  
- [PyTorch `nn` Docs](https://pytorch.org/docs/stable/nn.html)  
- [Kaggle – 1D CNN + Transformer](https://www.kaggle.com/code/buchan/transformer-network-with-1d-cnn-feature-extraction)  
- Course lecture slides provided by the instructor  

---

## 👨‍🎓 Author Info

```text
Name       : Sahil Sharma
Program    : M.Sc - M.Tech (Data and Computational Sciences)
Institute  : Indian Institute of Technology, Jodhpur
