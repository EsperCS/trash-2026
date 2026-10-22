# AI Security & Jailbreak Detection Pipeline

[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![RMIT Hackathon 2026](https://img.shields.io/badge/RMIT%20Hackathon-2026%20Preparation-red.svg)](https://rmit-hackathon.com/)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9615-brightgreen.svg)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-90.25%25-brightgreen.svg)]()

> **RMIT Hackathon 2026 Technical Preparation — Security Track**  
> **Author:** Hắc Minh Quân (HMQ) — Task Owner: Fine-Tuning & AI Security Evaluation  
> **Affiliation:** Ho Chi Minh City University of Technology (HCMUT) / Team Bún Bò Huế  

---

## 📌 Project Overview

This repository contains an end-to-end, production-grade AI Security classification pipeline designed to detect **adversarial jailbreaks and prompt injection attempts** against Large Language Models (LLMs). Built and benchmarked using the official RMIT Hackathon dataset, this project provides:

1. **Automated Data Auditing (`src/data_audit.py`):** Structural validation, class balance verification, exact/near-duplicate detection, and length distribution statistics.
2. **High-Performance Defensive Baseline (`src/train_baseline.py`):** Multi-scale feature union of word ($1\text{–}3$ n-grams) and character ($2\text{–}6$ n-grams) TF-IDF vectors with tuned regularized Logistic Regression. Trains in **5.4 seconds** and achieves **0.9615 ROC-AUC** and **90.25% Accuracy**.
3. **Transformer Fine-Tuning Pipeline (`src/train_transformer.py`):** HuggingFace Trainer workflow for DistilBERT / XLM-RoBERTa fine-tuning with early stopping, device auto-resolution (MPS/CUDA/CPU), and stratified validation.
4. **Comprehensive Error & Adversarial Analysis (`reports/error_analysis.md`):** Quantitative inspection of false positives, false negatives, operational threshold curves, and empirical robustness across 6 attack vectors (Direct, Indirect, Base64, Leetspeak, and Multilingual).
5. **System-Level Defense-in-Depth (`reports/security_recommendations.md`):** Enterprise security architecture based on [Microsoft's Zero Trust SFI Framework](https://learn.microsoft.com/en-us/security/zero-trust/sfi/defend-indirect-prompt-injection).

---

## 📊 Benchmark Results

Evaluated on the held-out validation set ($N = 800$, stratified 80/20 split, `seed=42`) from the verified 4,000-sample training dataset:

| Model / Run ID | Training Data | Evaluation Data | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---:|---:|---:|---:|---:|
| `baseline-01` | RMIT Train (3.2k) | RMIT Val (800) | 0.9549 | 88.62% | 86.12% | 85.31% | 0.8571 |
| **`baseline-tuned`** | **RMIT Train (3.2k)** | **RMIT Val (800)** | **0.9615** | **90.25%** | **87.58%** | **88.12%** | **0.8785** |
| `malignant-zero-shot` | RMIT Train (4.0k) | **Malignant Dataset (1,581)** | **0.9984** | **96.52%** | **99.54%** | **79.93%** | **0.8866** |
| **`fortified-combined`** | **RMIT + Malignant (5.5k)** | **Malignant Val (317)** | **1.0000** | **99.68%** | **99.07%** | **99.07%** | **0.9907** |
| `fortified-combined` | RMIT + Malignant (5.5k) | RMIT Val (800) | 0.9589 | 90.00% | 87.22% | 87.81% | 0.8751 |

```text
Confusion Matrix (Zero-Shot on marycamilainfo/prompt-injection-malignant, N=1,581):
                 Predicted Benign    Predicted Malicious
Actual Benign         1,311                   1  (FPR = 0.08%!)
Actual Malicious         54                 215  (Recall = 79.93% at theta=0.5; 97.03% at theta=0.2)
```

---

## 🎯 Key Empirical Findings

1. **Low-Resource Language Blindspot (Critical Vulnerability):**  
   An English-trained detector detects standard English injection (`"Ignore all previous instructions..."`) with **$P = 0.9739$**, but completely fails on Vietnamese translation (`"Bỏ qua tất cả các hướng dẫn trước đó..."`) scoring **$P = 0.0671$** (classified as benign!). Multilingual pre-training (e.g. XLM-RoBERTa or PhoBERT) or translation preprocessing is mandatory.
2. **Encoding Bypass:**  
   Base64-encoded instructions pass through with **$P = 0.0520$**. Pre-decoding pipelines must be enforced before classification.
3. **Threshold Engineering:**  
   In mission-critical gateways, shifting decision threshold $\theta$ from $0.50$ to $0.30$ increases attack recall from **88.12% to 94.37%**.

---

## 📁 Repository Structure

```text
Bun-Bo-Hue---RMIT-HACKATHON-2026/
├── README.md                           # Project documentation (this file)
├── requirements.txt                    # Python package dependencies
├── .gitignore                          # Git exclusions (checkpoints, data, venv)
├── .env.example                        # Environment variable template
├── configs/
│   └── baseline.yaml                   # Baseline model configuration
├── data/
│   ├── train.csv                       # Verified competition training data (4,000 rows)
│   ├── test.csv                        # Verified competition test data (1,000 rows)
│   ├── sample_submission.csv           # Official submission template (1,000 rows)
│   └── malignant/                      # marycamilainfo/prompt-injection-malignant benchmark (1,581 rows)
├── models/
│   ├── baseline_tfidf_lr.pkl           # Trained serialization of tuned pipeline (0.9615 ROC-AUC)
│   └── fortified_tfidf_lr.pkl          # Fortified model trained on RMIT + Malignant (fixes leetspeak)
├── submissions/
│   ├── baseline_submission.csv         # Kaggle leaderboard primary submission (1,000 rows)
│   └── fortified_submission.csv        # Fortified submission with external benchmark features (1,000 rows)
├── src/
│   ├── __init__.py
│   ├── data_audit.py                   # Dataset auditing & distribution analyzer
│   ├── train_baseline.py               # Classical ML baseline trainer & submission generator
│   ├── train_transformer.py            # HuggingFace Transformer fine-tuning script
│   └── evaluate.py                     # Evaluation metrics, plotting & error analysis
├── tests/
│   ├── __init__.py
│   └── test_data_and_metrics.py        # 9 unit tests for metrics and pipeline integrity
└── reports/
    ├── data_audit.md                   # Verified audit of dataset structure
    ├── experiment_results.csv          # Experiment tracking log with measured metrics
    ├── error_analysis.md               # In-depth false positive/negative & robustness report
    ├── security_recommendations.md     # Defense-in-depth architecture guide
    └── HMQ_handoff.md                  # Team handoff for Quân Hoàng, Phú, and Nhân
```

---

## 🚀 Quick Start & Reproduction

### 1. Prerequisites & Environment Setup

Python 3.12 or 3.14 on macOS, Linux, or Windows:

```bash
# Clone the repository
git clone https://github.com/hacminhquan/Bun-Bo-Hue---RMIT-HACKATHON-2026.git
cd Bun-Bo-Hue---RMIT-HACKATHON-2026

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Verify Pipeline Integrity with Unit Tests

Run the comprehensive unit test suite:

```bash
python3 tests/test_data_and_metrics.py
# Expected output: Ran 9 tests in ~0.1s -> OK
```

### 3. Run Data Audit

Inspect the competition dataset:

```bash
python3 src/data_audit.py --data_dir data
# Output generated in reports/data_audit.md
```

### 4. Train the Baseline & Generate Submission

Train the optimized classifier and generate the submission file for Kaggle:

```bash
# Standard baseline (C=1.0)
python3 src/train_baseline.py --data_dir data --seed 42 --C 1.0 --exp_name baseline-01

# Optimized baseline (C=10.0) -> Reaches 0.9615 ROC-AUC
python3 src/train_baseline.py --data_dir data --seed 42 --C 10.0 --exp_name baseline-tuned
```

The output prediction file is saved to:
`submissions/baseline_submission.csv` (Columns: `Id,TARGET`, 1,000 rows).

### 5. Multi-GPU Kaggle Baseline (2x Tesla T4)

To run the SOTA Deep Learning Baseline (`microsoft/deberta-v3-base` + TF-IDF Ensemble) on Kaggle with **2x Tesla T4 GPUs** and zero local downloads:

1. Open [Kaggle](https://www.kaggle.com/) -> Create New Notebook.
2. In the right sidebar:
   - **Accelerator:** Select `GPU T4 x2` (29.2 GB total VRAM).
   - **Internet:** Turn `On`.
   - **Input Data:** Click `+ Add Input` -> Search and attach the remote dataset:
     `https://kaggle.com/datasets/2be4a417b9fd2b9b565cf2136f15e49bd5a39cba1609fe565865a4908d94fc53`
3. Click `File` -> `Import Notebook` -> Upload [`notebooks/rmithack_baseline.ipynb`](notebooks/rmithack_baseline.ipynb).
4. Click **Run All**. The notebook will:
   - Auto-detect GPU hardware and distributed devices.
   - Scan `/kaggle/input/` dynamically and resolve data structures, schemas, and labels.
   - Fit Engine A (Fast TF-IDF N-grams).
   - Train Engine B (`microsoft/deberta-v3-base`) with LoRA PEFT using mixed-precision FP16 across both GPUs.
   - Blend Engine C (SOTA Ensemble) and export `/kaggle/working/submission.csv`.
   - Run an 8-vector adversarial red-team stress test.

---

## 🛡️ Enterprise Defense-in-Depth Architecture

A classifier is **one layer** in a multi-tier security framework. As detailed in [`reports/security_recommendations.md`](reports/security_recommendations.md):

```
User Query ──▶ [1. Input Sanitization]
                     │
                     ▼
             [2. Classifier Gate (Our Model)]
              ├── Score < 0.25: Fast Path (Pass)
              ├── 0.25 <= Score <= 0.70: Inspection Queue (LLM Critic)
              └── Score > 0.70: Rejection / Security Block
                     │
                     ▼
             [3. Content Isolation & Delimiters (<user_input>)]
                     │
                     ▼
             [4. Least-Privilege LLM Processing]
                     │
                     ▼
             [5. Output Screening & PII Scrubber] ──▶ Safe Response
```

---

## 👥 Team Coordination (Handoff Summary)

- **Quân Hoàng (Blue Team):** Submit `submissions/baseline_submission.csv` directly to Kaggle to claim the **20% Challenge 1 mark**. Use `src/train_baseline.py` as the baseline anchor for Challenge 2 modeling. For maximum leaderboard score, run `notebooks/rmithack_baseline.ipynb` on Kaggle 2x T4.
- **Phú (Red Team — Image Attacks):** Note that text classifiers are blind to image pixels. Craft OCR-based visual injection prompts to demonstrate this attack surface.
- **Nhân (Red Team — Prompt Attacks):** Use the Vietnamese language transfer vulnerability ($P=0.0671$) and Base64 obfuscation ($P=0.0520$) discovered in our error analysis to construct your 5 crafted attack prompts for Challenge 4.

---

## 📄 License & Attribution

This project is developed under the MIT License for educational and research purposes in connection with RMIT Hackathon 2026.

