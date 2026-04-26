# 🧬 CRISPR Off-Target Prediction System

A machine learning system that predicts whether a DNA sequence (off-target site) is **SAFE** or poses a risk of **unintended gene editing** by CRISPR-Cas9.

---

## 📁 Project Structure

```
crispr_project/
├── app.py              ← Streamlit web application
├── train_model.py      ← ML training pipeline
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
├── CIRCLE_seq.csv      ← Primary training dataset (place here)
├── GUIDE-Seq.csv       ← Validation dataset (place here)
│
└── (generated after training)
    ├── rf_model.joblib ← Trained Random Forest model
    ├── lr_model.joblib ← Trained Logistic Regression model
    ├── scaler.joblib   ← Feature scaler for LR
    └── metrics.json    ← Evaluation metrics
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place datasets
Put `CIRCLE_seq.csv` and `GUIDE-Seq.csv` in the same folder as the scripts.

### 3. Train the models
```bash
python train_model.py
```

Or specify custom paths:
```bash
python train_model.py --circle path/to/CIRCLE_seq.csv --guide path/to/GUIDE-Seq.csv
```

### 4. Launch the Streamlit app
```bash
streamlit run app.py
```

---

## 🔬 How It Works

### ML Pipeline
```
Raw CSV datasets
      ↓
Data cleaning (remove gap chars, filter short seqs)
      ↓
Class balancing (downsample 10:1 neg:pos ratio)
      ↓
Feature engineering (9 numerical features per pair)
      ↓
80/20 train/test split (stratified)
      ↓
Random Forest + Logistic Regression training
      ↓
Evaluation (Accuracy, Confusion Matrix, ROC-AUC)
      ↓
Model serialisation (joblib)
      ↓
Streamlit prediction interface
```

### Features Extracted
| Feature | Description |
|---------|-------------|
| `mismatch_count` | Total mismatches between sgRNA and target |
| `similarity_score` | Proportion of matching bases |
| `seed_mismatch` | Mismatches in PAM-proximal 12 bases |
| `max_consecutive_mismatch` | Longest mismatch run |
| `gc_content_sgrna` | GC fraction of the guide RNA |
| `gc_content_off` | GC fraction of the off-target sequence |
| `length_diff` | Absolute difference in sequence lengths |
| `distal_mismatch` | Mismatches in first 8 (PAM-distal) bases |
| `first_half_mismatch` | Mismatches in first half of alignment |

### Risk Thresholds
| Probability | Risk Level | Recommendation |
|-------------|-----------|----------------|
| < 35% | 🟢 LOW | ✅ SAFE |
| 35–65% | 🟡 MEDIUM | ⚠️ CAUTION |
| > 65% | 🔴 HIGH | 🚨 NOT SAFE |

---

## 📊 Model Performance

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Random Forest | **90.87%** | **0.9505** |
| Logistic Regression | 80.05% | 0.9020 |

Trained on CIRCLE-seq (584,949 pairs) + GUIDE-seq (213,933 pairs).

---

## 🧬 Datasets

- **CIRCLE-seq** — Comprehensive *in vitro* off-target detection; 7,371 off-target sites.
- **GUIDE-seq** — Genome-wide, unbiased DSB identification; 50 validated off-target sites.

---

## ⚠️ Disclaimer

This tool is for **research and educational purposes only**. It should not be used as the sole basis for clinical or therapeutic decisions. Always validate computationally predicted off-targets experimentally.

---

## 🛠️ Tech Stack

- Python 3.x
- scikit-learn (Random Forest, Logistic Regression)
- pandas / numpy (data processing)
- Streamlit (web interface)
- matplotlib / seaborn (visualization)
- joblib (model serialisation)
