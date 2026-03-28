# 🧠 CNN Calibration: When Confidence Meets Reality

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Authors:** María Muñoz Pérez, Isabel Carballo Rueda, MohammadErfan Jabbari  
**Course:** Deep Learning - CNN | October 2025

---

## Overview

A CNN that says it's 95% confident — but how often is it actually right? This project explores **calibration**: the alignment between a model's predicted confidence and its real-world accuracy.

Miscalibrated models are a hidden danger. In high-stakes domains like AI in medicine (my field), overconfident wrong predictions can be worse than uncertain ones. We study this problem on a binary CIFAR-10 task and apply **temperature scaling** as a lightweight fix.

---

## Research Questions

- Does model size drive overconfidence when data is limited?
- Can temperature scaling recover calibration without touching the weights?
- Are pretrained models better calibrated than models trained from scratch?

---

## 📁 Project Structure

```
CNN-Calibration/
├── README.md
├── CNN_Calibration_Project.ipynb      # Full implementation
├── CNN_Calibration_Project.html       # Notebook export
│
├── project_report.tex                 # LaTeX report source
├── project_report.pdf                 # 3-page academic report
├── references.bib
│
├── images/
│   ├── reliability_diagrams.png       # Before/after calibration
│   ├── temp_scaling_ece.png           # ECE vs temperature
│   └── temp_scaling_nll.png           # NLL vs temperature
│
├── data/                              # CIFAR-10 (auto-downloaded)
├── .gitignore
└── requirements.txt
```

---

## 🔬 Approach

### Task

Binary classification on CIFAR-10: **Bird** (class 2) vs **Cat** (class 3).  
Split: 8,000 train / 2,000 val / 2,000 test. Normalized to [-1, 1].

### Models Compared

| Model              | Parameters            | Training                |
| ------------------ | --------------------- | ----------------------- |
| LeNet-5            | 61,326                | From scratch, 20 epochs |
| Mini-LeNet         | 4,048                 | From scratch, 20 epochs |
| MobileNet-V3-Small | 1.5M (2K trainable)   | Fine-tuned, 80 epochs   |
| ResNet-18          | 11.2M (1K trainable)  | Fine-tuned, 80 epochs   |
| VGG-16             | 134.3M (8K trainable) | Fine-tuned, 80 epochs   |

### Calibration Metrics

**Expected Calibration Error (ECE)** — partitions predictions into 15 confidence bins and measures the weighted gap between confidence and accuracy. Lower = better.

```
ECE = Σ (|B_m|/n) × |acc(B_m) - conf(B_m)|
```

**Temperature Scaling** — a single scalar `T` applied to the logits before softmax. No retraining needed.

```
p_i = exp(z_i / T) / Σ exp(z_j / T)
```

`T > 1` softens predictions, `T < 1` sharpens them. The optimal `T` is found by minimizing NLL on the validation set.

---

## 📊 Results

### Reliability Diagrams

![Reliability Diagrams](images/reliability_diagrams.png)

Before calibration, LeNet-5 is severely overconfident — predictions pile up in high-confidence bins even when accuracy doesn't match. After temperature scaling (T=3.71), the bars align closely with the diagonal. Mini-LeNet was already more conservative and needed little correction (T=1.23).

### ECE vs Temperature

<p align="center">
  <img src="images/temp_scaling_ece.png" width="400"/>
</p>

### Summary Table

| Model      | Generalization Gap | ECE (before) | ECE (after) | Optimal T |
| ---------- | ------------------ | ------------ | ----------- | --------- |
| LeNet-5    | 17.7%              | 0.1614       | 0.0243      | 3.71      |
| Mini-LeNet | 4.3%               | 0.0371       | 0.0329      | 1.23      |
| VGG-16     | 0.9%               | 0.0168       | 0.0168      | 1.00      |

**Main findings:**

1. High capacity + limited data → overfitting → overconfidence
2. Temperature scaling is effective even when miscalibration is severe
3. Pretrained models generalize and calibrate better out of the box

---

## 🔗 Resources

- [Project Report (PDF)](project_report.pdf)
- [Guo et al. 2017](http://proceedings.mlr.press/v70/guo17a.html)

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.
