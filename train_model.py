"""
train_model.py — CRISPR Off-Target Prediction System
=====================================================
Run this script once to train and save the ML models.
Usage:
    python train_model.py
    python train_model.py --circle path/to/CIRCLE_seq.csv --guide path/to/GUIDE-Seq.csv
"""

import argparse
import json
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# ─── Configuration ────────────────────────────────────────────────────────────
DEFAULT_CIRCLE = "CIRCLE_seq.csv"
DEFAULT_GUIDE  = "GUIDE-Seq.csv"
OUTPUT_DIR     = os.path.dirname(os.path.abspath(__file__))


# ─── Feature Engineering ─────────────────────────────────────────────────────

def clean_seq(s) -> str:
    """Remove all non-ATGC characters and uppercase.

    FIX #5: accepts any type (not just str) — cast via str() first, matching
    the defensive pattern needed when reading from DataFrames where NaN or
    numeric values may sneak through despite upstream .dropna().
    """
    return re.sub(r'[^ATGCatgc]', '', str(s)).upper()


def extract_features(sgrna: str, off_seq: str) -> dict:
    """
    Convert a (sgRNA, off-target) pair into numerical features for ML.

    Features:
    1.  mismatch_count          — total number of base mismatches
    2.  similarity_score        — proportion of matching bases
                                  NOTE: this is a linear transform of mismatch_count
                                  (1 - mismatch_count/length) and therefore adds no
                                  independent information when sequence lengths are
                                  equal. It is retained here for backward compatibility
                                  with saved models. Consider replacing with edit
                                  distance (including indels) in a future retrain.
    3.  seed_mismatch           — mismatches in PAM-proximal 12 bases (seed region)
    4.  max_consecutive_mismatch — longest consecutive run of mismatches
    5.  gc_content_sgrna        — GC fraction of the sgRNA
    6.  gc_content_off          — GC fraction of the off-target sequence
    7.  length_diff             — absolute difference in sequence lengths
    8.  distal_mismatch         — mismatches in the first 8 (PAM-distal) bases
    9.  first_half_mismatch     — mismatches in the first half of the alignment
    """
    length = min(len(sgrna), len(off_seq))
    s1, s2 = sgrna[:length], off_seq[:length]

    mismatches        = [i for i in range(length) if s1[i] != s2[i]]
    mismatch_count    = len(mismatches)
    similarity        = 1 - mismatch_count / length if length > 0 else 0

    seed_len          = min(12, length)
    seed_mismatch     = sum(a != b for a, b in zip(s1[-seed_len:], s2[-seed_len:]))

    max_consec = curr = 0
    for i in range(length):
        if s1[i] != s2[i]:
            curr += 1
            max_consec = max(max_consec, curr)
        else:
            curr = 0

    gc_sgrna = (sgrna.count('G') + sgrna.count('C')) / len(sgrna) if sgrna else 0
    gc_off   = (off_seq.count('G') + off_seq.count('C')) / len(off_seq) if off_seq else 0
    len_diff = abs(len(sgrna) - len(off_seq))

    distal_len      = min(8, length)
    distal_mismatch = sum(a != b for a, b in zip(s1[:distal_len], s2[:distal_len]))
    half            = length // 2
    first_half_mm   = sum(s1[i] != s2[i] for i in range(half)) if half > 0 else 0

    return {
        'mismatch_count':           mismatch_count,
        'similarity_score':         similarity,
        'seed_mismatch':            seed_mismatch,
        'max_consecutive_mismatch': max_consec,
        'gc_content_sgrna':         gc_sgrna,
        'gc_content_off':           gc_off,
        'length_diff':              len_diff,
        'distal_mismatch':          distal_mismatch,
        'first_half_mismatch':      first_half_mm,
    }


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_circle_seq(path: str) -> pd.DataFrame:
    print(f"  Loading CIRCLE-seq: {path}")
    df = pd.read_csv(path)
    df = df[['sgRNA_seq', 'off_seq', 'label']].dropna()
    df['sgRNA_clean'] = df['sgRNA_seq'].apply(clean_seq)
    df['off_clean']   = df['off_seq'].apply(clean_seq)
    df = df[(df['sgRNA_clean'].str.len() >= 15) & (df['off_clean'].str.len() >= 15)]
    df['label']       = df['label'].astype(int)
    print(f"    → {len(df):,} rows | labels: {df['label'].value_counts().to_dict()}")
    return df[['sgRNA_clean', 'off_clean', 'label']]


def load_guide_seq(path: str) -> pd.DataFrame:
    print(f"  Loading GUIDE-seq: {path}")
    df = pd.read_csv(path)
    df = df.rename(columns={'DNA': 'off_seq', 'crRNA': 'sgRNA_seq'})
    df = df[['sgRNA_seq', 'off_seq', 'label']].dropna()
    df['sgRNA_clean'] = df['sgRNA_seq'].apply(clean_seq)
    df['off_clean']   = df['off_seq'].apply(clean_seq)
    df = df[(df['sgRNA_clean'].str.len() >= 15) & (df['off_clean'].str.len() >= 15)]
    df['label']       = df['label'].astype(int)
    print(f"    → {len(df):,} rows | labels: {df['label'].value_counts().to_dict()}")
    return df[['sgRNA_clean', 'off_clean', 'label']]


# ─── Main Training Pipeline ───────────────────────────────────────────────────

def train(circle_path: str, guide_path: str):
    print("\n" + "="*60)
    print("  CRISPR Off-Target Prediction — Model Training")
    print("="*60)

    # ── Step 1: Load data ────────────────────────────────────────────────────
    print("\n[1/5] Loading datasets...")
    frames = []
    if os.path.exists(circle_path):
        frames.append(load_circle_seq(circle_path))
    else:
        print(f"  ⚠️  CIRCLE-seq not found at {circle_path}, skipping.")

    if os.path.exists(guide_path):
        frames.append(load_guide_seq(guide_path))
    else:
        print(f"  ⚠️  GUIDE-seq not found at {guide_path}, skipping.")

    if not frames:
        print("  ❌  No datasets found. Exiting.")
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    print(f"\n  Combined: {len(df):,} rows total")
    print(f"  Positive (label=1): {(df['label']==1).sum():,}")
    print(f"  Negative (label=0): {(df['label']==0).sum():,}")

    # ── Step 2: Balance dataset ──────────────────────────────────────────────
    print("\n[2/5] Balancing dataset (10:1 negative:positive ratio)...")
    pos = df[df['label'] == 1]
    neg = df[df['label'] == 0]

    # FIX #4: use replace=False and cap n_samples at len(neg) to ensure true
    # downsampling. The original code passed n_samples=len(pos)*10 without a
    # cap, so when len(neg) < len(pos)*10 resample() would silently oversample
    # (bootstrap with replacement), which is the opposite of the intent.
    target_neg = min(len(pos) * 10, len(neg))
    neg_dn = resample(neg, n_samples=target_neg, replace=False, random_state=42)

    df_bal = pd.concat([pos, neg_dn]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  After balancing: {len(df_bal):,} rows")
    print(f"  Positive: {(df_bal['label']==1).sum():,} | Negative: {(df_bal['label']==0).sum():,}")

    # ── Step 3: Feature extraction ───────────────────────────────────────────
    print("\n[3/5] Extracting features...")
    features = df_bal.apply(
        lambda row: extract_features(row['sgRNA_clean'], row['off_clean']), axis=1
    )
    X = pd.DataFrame(features.tolist())
    y = df_bal['label'].values
    print(f"  Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"  Features: {X.columns.tolist()}")

    # ── Step 4: Train/test split ─────────────────────────────────────────────
    print("\n[4/5] Splitting data (80/20) and training models...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── Random Forest ────────────────────────────────────────────────────────
    print("\n  → Training Random Forest (100 trees, depth=12)...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_acc  = accuracy_score(y_test, rf_pred)
    rf_auc  = roc_auc_score(y_test, rf_prob)
    rf_cm   = confusion_matrix(y_test, rf_pred)
    print(f"     Accuracy : {rf_acc:.4f} ({rf_acc*100:.2f}%)")
    print(f"     ROC-AUC  : {rf_auc:.4f}")
    print(f"     Confusion Matrix:\n{rf_cm}")

    # ── Logistic Regression ──────────────────────────────────────────────────
    print("\n  → Training Logistic Regression...")
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    lr = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
    lr.fit(X_train_s, y_train)
    lr_pred = lr.predict(X_test_s)
    lr_prob = lr.predict_proba(X_test_s)[:, 1]
    lr_acc  = accuracy_score(y_test, lr_pred)
    lr_auc  = roc_auc_score(y_test, lr_prob)
    lr_cm   = confusion_matrix(y_test, lr_pred)
    print(f"     Accuracy : {lr_acc:.4f} ({lr_acc*100:.2f}%)")
    print(f"     ROC-AUC  : {lr_auc:.4f}")
    print(f"     Confusion Matrix:\n{lr_cm}")

    # ── Step 5: Save artifacts ───────────────────────────────────────────────
    print(f"\n[5/5] Saving model artifacts to {OUTPUT_DIR}/...")
    joblib.dump(rf,     os.path.join(OUTPUT_DIR, 'rf_model.joblib'))
    joblib.dump(lr,     os.path.join(OUTPUT_DIR, 'lr_model.joblib'))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.joblib'))

    metrics = {
        'rf_accuracy':   rf_acc,
        'rf_auc':        rf_auc,
        'rf_cm':         rf_cm.tolist(),
        'lr_accuracy':   lr_acc,
        'lr_auc':        lr_auc,
        'lr_cm':         lr_cm.tolist(),
        'feature_names': X.columns.tolist(),  # used by app.py for order validation
        'n_train':       len(X_train),
        'n_test':        len(X_test),
        'n_positive':    int((y == 1).sum()),
        'n_negative':    int((y == 0).sum()),
    }
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n  ✅ Saved: rf_model.joblib")
    print("  ✅ Saved: lr_model.joblib")
    print("  ✅ Saved: scaler.joblib")
    print("  ✅ Saved: metrics.json")
    print("\n" + "="*60)
    print("  Training complete! Run `streamlit run app.py` to launch.")
    print("="*60 + "\n")


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CRISPR off-target prediction models")
    parser.add_argument("--circle", default=DEFAULT_CIRCLE, help="Path to CIRCLE_seq.csv")
    parser.add_argument("--guide",  default=DEFAULT_GUIDE,  help="Path to GUIDE-Seq.csv")
    args = parser.parse_args()
    train(args.circle, args.guide)
