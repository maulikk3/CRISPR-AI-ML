"""
CRISPR Off-Target Prediction System
=====================================
Streamlit app for predicting whether a DNA off-target site is SAFE or NOT SAFE.
Uses Random Forest + Logistic Regression trained on CIRCLE-seq + GUIDE-seq datasets.
"""

import streamlit as st
import numpy as np
import pandas as pd
import re
import joblib
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CRISPR Off-Target Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
  }

  /* Dark sci-fi theme */
  .stApp {
    background: #0a0e1a;
    color: #e2e8f0;
  }

  /* Header banner */
  .hero-banner {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 50%, #0d1b2a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 70% 50%, rgba(56,189,248,0.08) 0%, transparent 60%);
    pointer-events: none;
  }
  .hero-title {
    font-size: 2.1rem;
    font-weight: 700;
    color: #f0f9ff;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
  }
  .hero-subtitle {
    font-size: 0.95rem;
    color: #7dd3fc;
    margin: 0;
    font-weight: 400;
  }
  .dna-icon {
    font-size: 3rem;
    position: absolute;
    right: 2.5rem;
    top: 50%;
    transform: translateY(-50%);
    opacity: 0.3;
  }

  /* Cards */
  .card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
  }
  .card-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #38bdf8;
    margin-bottom: 1rem;
  }

  /* Risk badges */
  .risk-low {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    color: #6ee7b7;
    padding: 1rem 1.5rem;
    border-radius: 10px;
    font-size: 1.5rem;
    font-weight: 700;
    text-align: center;
  }
  .risk-medium {
    background: linear-gradient(135deg, #78350f, #92400e);
    border: 1px solid #f59e0b;
    color: #fcd34d;
    padding: 1rem 1.5rem;
    border-radius: 10px;
    font-size: 1.5rem;
    font-weight: 700;
    text-align: center;
  }
  .risk-high {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444;
    color: #fca5a5;
    padding: 1rem 1.5rem;
    border-radius: 10px;
    font-size: 1.5rem;
    font-weight: 700;
    text-align: center;
  }

  /* Sequence display */
  .seq-display {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.1em;
    line-height: 2;
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    overflow-x: auto;
    white-space: nowrap;
  }

  /* Metric pills */
  .metric-pill {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.82rem;
    color: #94a3b8;
    margin: 0.2rem;
  }
  .metric-value {
    color: #38bdf8;
    font-weight: 600;
  }

  /* Input styling override */
  .stTextInput > div > div > input {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
    background: #0f172a !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
  }
  .stTextInput > div > div > input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.15) !important;
  }

  /* Button */
  .stButton > button {
    background: linear-gradient(135deg, #0369a1, #0284c7) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #0284c7, #0ea5e9) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(56,189,248,0.3) !important;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e293b !important;
  }

  /* Info boxes */
  .info-box {
    background: #0c1929;
    border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.87rem;
    color: #93c5fd;
    line-height: 1.5;
  }

  /* Section headers */
  .section-header {
    font-size: 1rem;
    font-weight: 600;
    color: #cbd5e1;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
  }

  /* Mismatch position chip */
  .base-match {
    display: inline-block;
    color: #34d399;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    margin: 1px;
    padding: 2px 4px;
    border-radius: 3px;
    background: rgba(52,211,153,0.08);
  }
  /* FIX #7: differentiated mismatch classes for sgRNA ref vs target alt */
  .base-mismatch {
    display: inline-block;
    color: #f87171;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    margin: 1px;
    padding: 2px 4px;
    border-radius: 3px;
    background: rgba(248,113,113,0.15);
    border-bottom: 2px solid #ef4444;
  }
  .base-ref {
    display: inline-block;
    color: #fbbf24;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    margin: 1px;
    padding: 2px 4px;
    border-radius: 3px;
    background: rgba(251,191,36,0.12);
    border-bottom: 2px solid #f59e0b;
  }

  /* Accuracy badge */
  .accuracy-badge {
    background: #0c1929;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    text-align: center;
    margin: 0.4rem 0;
  }
  .accuracy-number {
    font-size: 1.4rem;
    font-weight: 700;
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
  }
  .accuracy-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    display: block;
  }

  /* Warning box */
  .stAlert { border-radius: 8px !important; }

  /* Progress bar */
  .stProgress > div > div { border-radius: 4px; }

  /* Tab style */
  .stTabs [data-baseweb="tab-list"] {
    background: #111827 !important;
    border-radius: 8px !important;
    padding: 4px !important;
    gap: 4px !important;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 6px !important;
    color: #64748b !important;
    font-weight: 500 !important;
  }
  .stTabs [aria-selected="true"] {
    background: #1e3a5f !important;
    color: #38bdf8 !important;
  }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ─────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_models():
    """Load pre-trained models and metrics."""
    rf   = joblib.load(BASE_DIR / "rf_model.joblib")
    lr   = joblib.load(BASE_DIR / "lr_model.joblib")
    sc   = joblib.load(BASE_DIR / "scaler.joblib")
    with open(BASE_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return rf, lr, sc, metrics


# FIX #5: match train_model.py — cast to str before regex to handle any input type
def clean_seq(s) -> str:
    return re.sub(r'[^ATGCatgc]', '', str(s)).upper()


def validate_seq(s: str, name: str):
    """Return (cleaned, error_msg).

    Accepts sgRNA (typically 20 nt) and off-target sequences (up to 23+ nt with PAM).
    Hard cap raised to 50 to accommodate longer genomic windows without silent truncation.
    """
    cleaned = clean_seq(s)
    if len(cleaned) < 15:
        return None, f"{name}: must be at least 15 valid bases (A/T/G/C). Got {len(cleaned)}."
    if len(cleaned) > 50:  # FIX #9: raised from 30 to 50 to cover PAM + flanking bases
        return None, f"{name}: too long (max 50 bases). Got {len(cleaned)}."
    return cleaned, None


def extract_features(sgrna: str, off_seq: str) -> dict:
    length = min(len(sgrna), len(off_seq))
    s1, s2 = sgrna[:length], off_seq[:length]

    mismatches = [i for i in range(length) if s1[i] != s2[i]]
    mismatch_count = len(mismatches)
    similarity     = 1 - mismatch_count / length if length > 0 else 0

    seed_len      = min(12, length)
    seed_mismatch = sum(a != b for a, b in zip(s1[-seed_len:], s2[-seed_len:]))

    max_consec = curr = 0
    for i in range(length):
        if s1[i] != s2[i]:
            curr += 1; max_consec = max(max_consec, curr)
        else:
            curr = 0

    gc_sgrna = (sgrna.count('G') + sgrna.count('C')) / len(sgrna) if sgrna else 0
    gc_off   = (off_seq.count('G') + off_seq.count('C')) / len(off_seq) if off_seq else 0
    len_diff = abs(len(sgrna) - len(off_seq))
    distal_len = min(8, length)
    distal_mismatch = sum(a != b for a, b in zip(s1[:distal_len], s2[:distal_len]))
    half = length // 2
    first_half_mm = sum(s1[i] != s2[i] for i in range(half)) if half > 0 else 0

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


def get_mismatch_positions(sgrna: str, off_seq: str) -> list:
    length = min(len(sgrna), len(off_seq))
    return [i for i in range(length) if sgrna[i] != off_seq[i]]


def risk_level(prob: float) -> tuple:
    if prob < 0.35:
        return "LOW", "✅ SAFE", "#10b981", "risk-low"
    elif prob < 0.65:
        return "MEDIUM", "⚠️ CAUTION", "#f59e0b", "risk-medium"
    else:
        return "HIGH", "🚨 NOT SAFE", "#ef4444", "risk-high"


# FIX #7: differentiate sgRNA reference bases (amber) from target mismatches (red)
def render_alignment(sgrna: str, off_seq: str) -> str:
    length = min(len(sgrna), len(off_seq))
    s1, s2 = sgrna[:length], off_seq[:length]
    html_s1, html_s2 = [], []
    for a, b in zip(s1, s2):
        if a == b:
            html_s1.append(f'<span class="base-match">{a}</span>')
            html_s2.append(f'<span class="base-match">{b}</span>')
        else:
            html_s1.append(f'<span class="base-ref">{a}</span>')       # sgRNA ref — amber
            html_s2.append(f'<span class="base-mismatch">{b}</span>')  # target alt — red
    return (
        f"<div style='margin-bottom:0.3rem'>"
        f"<span style='color:#64748b;font-size:0.75rem;margin-right:0.5rem'>sgRNA&nbsp;&nbsp;</span>"
        f"{''.join(html_s1)}</div>"
        f"<div>"
        f"<span style='color:#64748b;font-size:0.75rem;margin-right:0.5rem'>Target&nbsp;</span>"
        f"{''.join(html_s2)}</div>"
    )


def plot_confusion_matrix(cm, title, ax):
    labels = ["Safe (0)", "Off-target (1)"]
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=0.5, linecolor='#1e293b',
        annot_kws={"size": 11, "weight": "bold"},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=11, color='#e2e8f0', pad=10)
    ax.set_xlabel("Predicted", color='#94a3b8', fontsize=9)
    ax.set_ylabel("Actual", color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#94a3b8', labelsize=8)


# ─── Load Models ─────────────────────────────────────────────────────────────
# FIX #3: initialize model_error before the try block so it's always defined
model_error = None
try:
    rf_model, lr_model, scaler, metrics = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    model_error = str(e)

# ─── Session State Defaults ───────────────────────────────────────────────────
# FIX #1: Initialize widget keys in session_state so example buttons can write to them
if "sgrna_input" not in st.session_state:
    st.session_state["sgrna_input"] = ""
if "off_input" not in st.session_state:
    st.session_state["off_input"] = ""

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧬 CRISPR Predictor")
    st.markdown("---")

    if models_loaded:
        st.markdown("**Model Performance**")
        st.markdown(f"""
        <div class="accuracy-badge">
          <span class="accuracy-number">{metrics['rf_accuracy']*100:.1f}%</span>
          <span class="accuracy-label">Random Forest Accuracy</span>
        </div>
        <div class="accuracy-badge">
          <span class="accuracy-number">{metrics['rf_auc']:.3f}</span>
          <span class="accuracy-label">RF ROC-AUC Score</span>
        </div>
        <div class="accuracy-badge">
          <span class="accuracy-number">{metrics['lr_accuracy']*100:.1f}%</span>
          <span class="accuracy-label">Logistic Reg. Accuracy</span>
        </div>
        <div class="accuracy-badge">
          <span class="accuracy-number">{metrics['lr_auc']:.3f}</span>
          <span class="accuracy-label">LR ROC-AUC Score</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("⚠️ Models not loaded")

    st.markdown("---")
    st.markdown("**Model Selection**")
    model_choice = st.radio("Active prediction model:", ["Random Forest", "Logistic Regression"], index=0)

    st.markdown("---")
    st.markdown("**Training Data**")
    st.markdown("""
    <div class="info-box">
      📊 <strong>CIRCLE-seq</strong><br>584,949 sequence pairs<br>7,371 off-target sites
    </div>
    <div class="info-box">
      📊 <strong>GUIDE-seq</strong><br>213,933 sequence pairs<br>50 off-target sites
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Risk Thresholds**")
    st.markdown("""
    <div style='font-size:0.82rem;color:#64748b;line-height:2'>
    🟢 &lt; 35% → LOW (SAFE)<br>
    🟡 35–65% → MEDIUM (CAUTION)<br>
    🔴 &gt; 65% → HIGH (NOT SAFE)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Built with 🧬 scikit-learn + Streamlit")

# ─── Hero Banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <span class="dna-icon">🧬</span>
  <p class="hero-title">CRISPR Off-Target Prediction System</p>
  <p class="hero-subtitle">
    Machine-learning powered analysis of CRISPR-Cas9 guide RNA off-target risk
    &nbsp;·&nbsp; Trained on CIRCLE-seq + GUIDE-seq datasets
    &nbsp;·&nbsp; Random Forest + Logistic Regression
  </p>
</div>
""", unsafe_allow_html=True)

if not models_loaded:
    st.error(f"❌ Could not load models: {model_error}")
    st.info("Run `train_model.py` first to generate the model files.")
    st.stop()

# ─── Main Tabs ───────────────────────────────────────────────────────────────
tab_predict, tab_model, tab_about = st.tabs(["🔬 Prediction", "📊 Model Analytics", "ℹ️ About"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    col_input, col_output = st.columns([1, 1], gap="large")

    # ── Input Panel ──────────────────────────────────────────────────────────
    with col_input:
        st.markdown('<div class="card-title">Input Sequences</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
          Enter sequences using only the bases <strong>A, T, G, C</strong> (15–50 characters each).
          Non-DNA characters will be automatically stripped.
        </div>
        """, unsafe_allow_html=True)

        # FIX #1: bind widgets to session_state keys so example buttons can
        # update them by writing to st.session_state before st.rerun()
        sgrna_input = st.text_input(
            "sgRNA Sequence (Guide RNA)",
            key="sgrna_input",
            placeholder="e.g. GAGTCCGAGCAGAAGAAGAA",
            help="The guide RNA sequence used by CRISPR-Cas9 to locate its target.",
        )

        offtarget_input = st.text_input(
            "DNA Target Sequence (Off-target site)",
            key="off_input",
            placeholder="e.g. GAAGTCCGAGGAGAGGAAGAAAGG",
            help="The DNA genomic sequence to evaluate for off-target risk.",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 Predict Off-Target Risk", use_container_width=True)

        # ── Example Sequences ────────────────────────────────────────────────
        st.markdown('<p class="section-header">Quick Examples</p>', unsafe_allow_html=True)
        examples = {
            "High Risk (few mismatches)": ("GGGAAAGACCCAGCATCCGT", "GGGAAAGACCCAGCATCCGT"),
            "Medium Risk":               ("GAGTCCGAGCAGAAGAAGAA", "GAAGTCCGAGGAGAGGAAGA"),
            "Low Risk (many mismatches)": ("GGGAAAGACCCAGCATCCGT", "CAAGGCATGATCAGCTTTAA"),
        }
        for label, (sg, off) in examples.items():
            if st.button(f"Load: {label}", key=label, use_container_width=True):
                # FIX #1: write directly to session_state keys that the
                # text_input widgets are bound to — values update on rerun
                st.session_state["sgrna_input"] = sg
                st.session_state["off_input"]   = off
                st.rerun()

    # ── Output Panel ─────────────────────────────────────────────────────────
    with col_output:
        st.markdown('<div class="card-title">Prediction Result</div>', unsafe_allow_html=True)

        # FIX #2: removed dead "last_sgrna" session_state branch that was
        # never populated; condition simplified to just the button press
        if predict_btn:
            if not sgrna_input.strip() or not offtarget_input.strip():
                st.warning("⚠️ Please enter both sequences.")
            else:
                sgrna_clean, err1 = validate_seq(sgrna_input, "sgRNA")
                off_clean,   err2 = validate_seq(offtarget_input, "DNA Target")

                if err1:
                    st.error(f"❌ {err1}")
                elif err2:
                    st.error(f"❌ {err2}")
                else:
                    # Feature extraction
                    feats   = extract_features(sgrna_clean, off_clean)
                    X_input = pd.DataFrame([feats])

                    # FIX #6: enforce feature order matches what the model was trained on
                    expected_features = metrics.get("feature_names", [])
                    if expected_features:
                        if list(X_input.columns) != expected_features:
                            st.error(
                                "⚠️ Feature mismatch between app and saved model. "
                                "Please retrain the model with the current `train_model.py`."
                            )
                            st.stop()
                        X_input = X_input[expected_features]  # enforce column order

                    # Predict with chosen model
                    if model_choice == "Random Forest":
                        prob = rf_model.predict_proba(X_input)[0][1]
                        active_model_name = "Random Forest"
                    else:
                        X_scaled = scaler.transform(X_input)
                        prob = lr_model.predict_proba(X_scaled)[0][1]
                        active_model_name = "Logistic Regression"

                    level, recommendation, color, css_class = risk_level(prob)
                    mm_positions = get_mismatch_positions(sgrna_clean, off_clean)

                    # ── Risk Badge ───────────────────────────────────────────
                    st.markdown(f"""
                    <div class="{css_class}">
                      {recommendation}<br>
                      <span style="font-size:1rem;font-weight:500;opacity:0.85">
                        Risk Level: {level} &nbsp;|&nbsp; Score: {prob:.1%}
                      </span>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ── Risk Score Bar ───────────────────────────────────────
                    st.markdown("**Risk Probability**")
                    st.progress(prob)  # FIX #10: removed redundant min(prob, 1.0)
                    col_a, col_b, col_c = st.columns(3)
                    col_a.markdown("<div style='color:#10b981;font-size:0.8rem'>0% — SAFE</div>", unsafe_allow_html=True)
                    col_b.markdown("<div style='color:#f59e0b;font-size:0.8rem;text-align:center'>50%</div>", unsafe_allow_html=True)
                    col_c.markdown("<div style='color:#ef4444;font-size:0.8rem;text-align:right'>100% — RISK</div>", unsafe_allow_html=True)

                    # ── Key Metrics ──────────────────────────────────────────
                    st.markdown('<p class="section-header">Sequence Analysis</p>', unsafe_allow_html=True)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Mismatches", feats['mismatch_count'])
                    m2.metric("Similarity", f"{feats['similarity_score']:.1%}")
                    m3.metric("Seed MM", feats['seed_mismatch'])
                    m4, m5, m6 = st.columns(3)
                    m4.metric("Max Consec. MM", feats['max_consecutive_mismatch'])
                    m5.metric("GC (sgRNA)", f"{feats['gc_content_sgrna']:.1%}")
                    m6.metric("GC (Target)", f"{feats['gc_content_off']:.1%}")

                    # ── Sequence Alignment ───────────────────────────────────
                    st.markdown('<p class="section-header">Sequence Alignment</p>', unsafe_allow_html=True)
                    align_html = render_alignment(sgrna_clean, off_clean)
                    st.markdown(
                        f'<div class="seq-display">{align_html}</div>',
                        unsafe_allow_html=True
                    )

                    if mm_positions:
                        pos_str = ", ".join([str(i+1) for i in mm_positions])
                        st.markdown(f"""
                        <div class="info-box">
                          🔴 Mismatch positions (1-based): <strong>{pos_str}</strong><br>
                          <span style='color:#fbbf24'>Amber bases</span> = sgRNA reference &nbsp;&nbsp;
                          <span style='color:#f87171'>Red bases</span> = target mismatch &nbsp;&nbsp;
                          <span style='color:#34d399'>Green bases</span> = match
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="info-box" style='border-left-color:#ef4444'>
                          ⚠️ <strong>Perfect match detected</strong> — sequences are identical.
                          This represents maximum off-target risk.
                        </div>
                        """, unsafe_allow_html=True)

                    # ── Explanation ──────────────────────────────────────────
                    st.markdown('<p class="section-header">What does this mean?</p>', unsafe_allow_html=True)

                    explanations = []
                    if feats['similarity_score'] > 0.8:
                        explanations.append("🔴 <strong>Very high similarity</strong> between sgRNA and target greatly increases off-target editing risk.")
                    elif feats['similarity_score'] > 0.6:
                        explanations.append("🟡 <strong>Moderate similarity</strong> — CRISPR-Cas9 may still bind this site with reduced efficiency.")
                    else:
                        explanations.append("🟢 <strong>Low similarity</strong> — Cas9 is unlikely to efficiently bind this target site.")

                    if feats['seed_mismatch'] == 0:
                        explanations.append("🔴 <strong>Seed region (PAM-proximal)</strong> is a perfect match — this is the most critical zone for off-target binding.")
                    elif feats['seed_mismatch'] <= 2:
                        explanations.append("🟡 <strong>Seed region</strong> has few mismatches — off-target binding remains possible.")
                    else:
                        explanations.append("🟢 <strong>Seed region</strong> has significant mismatches — Cas9 binding is substantially impaired.")

                    if feats['mismatch_count'] <= 3:
                        explanations.append(f"⚠️ Only <strong>{feats['mismatch_count']} mismatch(es)</strong> found — research shows Cas9 can tolerate up to 5 mismatches.")
                    else:
                        explanations.append(f"ℹ️ <strong>{feats['mismatch_count']} mismatches</strong> reduce binding probability, though seed-region mismatches matter most.")

                    for exp in explanations:
                        st.markdown(f'<div class="info-box">{exp}</div>', unsafe_allow_html=True)

                    st.markdown(f"""
                    <div style='margin-top:0.8rem;font-size:0.78rem;color:#475569;text-align:right'>
                      Prediction by: {active_model_name} · Confidence: {prob:.3f}
                    </div>
                    """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="
              display: flex; flex-direction: column; align-items: center;
              justify-content: center; height: 400px;
              color: #334155; text-align: center;
            ">
              <div style="font-size: 3.5rem; margin-bottom: 1rem; opacity: 0.4">🔬</div>
              <div style="font-size: 1rem; font-weight: 500; color:#475569">Enter sequences and click Predict</div>
              <div style="font-size: 0.83rem; color: #334155; margin-top: 0.5rem">
                Results will appear here including risk level,<br>alignment visualization, and explanations.
              </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown('<div class="card-title">Model Performance Analytics</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # ── Confusion Matrices ───────────────────────────────────────────────────
    with col1:
        st.markdown("#### Confusion Matrices")
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        fig.patch.set_facecolor('#111827')
        for ax in axes:
            ax.set_facecolor('#0f172a')

        plot_confusion_matrix(metrics['rf_cm'], "Random Forest", axes[0])
        plot_confusion_matrix(metrics['lr_cm'], "Logistic Regression", axes[1])
        plt.tight_layout(pad=1.5)
        st.pyplot(fig)
        plt.close()

    # ── Metrics Comparison ───────────────────────────────────────────────────
    with col2:
        st.markdown("#### Model Comparison")
        comparison_data = {
            "Metric": ["Accuracy", "ROC-AUC Score", "Training Data"],
            "Random Forest": [
                f"{metrics['rf_accuracy']*100:.2f}%",
                f"{metrics['rf_auc']:.4f}",
                "CIRCLE-seq + GUIDE-seq",
            ],
            "Logistic Regression": [
                f"{metrics['lr_accuracy']*100:.2f}%",
                f"{metrics['lr_auc']:.4f}",
                "CIRCLE-seq + GUIDE-seq",
            ],
        }
        df_comp = pd.DataFrame(comparison_data)
        st.dataframe(df_comp.set_index("Metric"), use_container_width=True)

        # Bar chart
        fig2, ax2 = plt.subplots(figsize=(5, 2.8))
        fig2.patch.set_facecolor('#111827')
        ax2.set_facecolor('#0f172a')
        models_names = ["Random Forest", "Logistic Reg."]
        accuracies   = [metrics['rf_accuracy'], metrics['lr_accuracy']]
        aucs         = [metrics['rf_auc'], metrics['lr_auc']]
        x = np.arange(2)
        bars1 = ax2.bar(x - 0.2, accuracies, 0.35, label='Accuracy', color='#0284c7', alpha=0.85)
        bars2 = ax2.bar(x + 0.2, aucs, 0.35, label='ROC-AUC', color='#7c3aed', alpha=0.85)
        ax2.set_xticks(x); ax2.set_xticklabels(models_names, color='#94a3b8', fontsize=8)
        ax2.set_ylim(0, 1.1)
        ax2.tick_params(colors='#64748b', labelsize=8)
        ax2.spines[['top','right']].set_visible(False)
        ax2.spines[['left','bottom']].set_color('#1e293b')
        ax2.set_facecolor('#0f172a')
        ax2.legend(fontsize=8, labelcolor='#94a3b8', facecolor='#111827', edgecolor='#1e293b')
        for bar in bars1: ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                                    f'{bar.get_height():.3f}', ha='center', va='bottom',
                                    color='#38bdf8', fontsize=7)
        for bar in bars2: ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                                    f'{bar.get_height():.3f}', ha='center', va='bottom',
                                    color='#a78bfa', fontsize=7)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.markdown("---")

    # ── Feature Importance ───────────────────────────────────────────────────
    st.markdown("#### Feature Importance (Random Forest)")
    feature_names_display = {
        'mismatch_count':           'Total Mismatch Count',
        'similarity_score':         'Similarity Score',
        'seed_mismatch':            'Seed Region Mismatch',
        'max_consecutive_mismatch': 'Max Consecutive Mismatches',
        'gc_content_sgrna':         'GC Content (sgRNA)',
        'gc_content_off':           'GC Content (Off-target)',
        'length_diff':              'Sequence Length Difference',
        'distal_mismatch':          'Distal Region Mismatch',
        'first_half_mismatch':      'First-Half Mismatch Count',
    }
    importances = rf_model.feature_importances_
    feat_names  = list(feature_names_display.values())
    sorted_idx  = np.argsort(importances)[::-1]

    fig3, ax3 = plt.subplots(figsize=(10, 3.5))
    fig3.patch.set_facecolor('#111827')
    ax3.set_facecolor('#0f172a')
    colors = ['#0284c7' if i < 3 else '#1e3a5f' for i in range(len(feat_names))]
    bars = ax3.bar(range(len(feat_names)),
                   importances[sorted_idx],
                   color=[colors[i] for i in range(len(feat_names))],
                   edgecolor='#0f172a', linewidth=0.5)
    ax3.set_xticks(range(len(feat_names)))
    ax3.set_xticklabels([feat_names[i] for i in sorted_idx],
                        rotation=30, ha='right', color='#94a3b8', fontsize=8)
    ax3.tick_params(axis='y', colors='#64748b', labelsize=8)
    ax3.spines[['top','right']].set_visible(False)
    ax3.spines[['left','bottom']].set_color('#1e293b')
    ax3.set_ylabel("Importance", color='#64748b', fontsize=9)
    ax3.set_title("Random Forest Feature Importances", color='#e2e8f0', fontsize=10)
    patch = mpatches.Patch(color='#0284c7', label='Top 3 features')
    ax3.legend(handles=[patch], fontsize=8, labelcolor='#94a3b8',
               facecolor='#111827', edgecolor='#1e293b')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.markdown("---")

    # ── Dataset Summary ──────────────────────────────────────────────────────
    st.markdown("#### Training Dataset Summary")
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    col_d1.metric("CIRCLE-seq Rows",    "584,949")
    col_d2.metric("GUIDE-seq Rows",     "213,933")
    col_d3.metric("Off-target (label=1)", "7,421")
    col_d4.metric("Train/Test Split",   "80% / 20%")

    st.markdown("""
    <div class="info-box">
      ⚖️ <strong>Class Imbalance Handling</strong> — The dataset is heavily imbalanced (≈100:1 safe:risk ratio).
      The negative class was downsampled to a 10:1 ratio and <code>class_weight='balanced'</code> was applied
      to ensure the model doesn't simply predict "safe" for all inputs.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown('<div class="card-title">About This System</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("""
        ### What is CRISPR Off-Target Prediction?

        **CRISPR-Cas9** is a revolutionary gene-editing technology that uses a short guide RNA (sgRNA)
        to direct the Cas9 protein to cut specific DNA sequences. However, Cas9 can sometimes bind and
        cut DNA at unintended locations — these are called **off-target sites**.

        Off-target edits are a major safety concern in clinical applications, as they can:
        - Disrupt functional genes
        - Activate oncogenes (cancer-driving genes)
        - Cause chromosomal rearrangements

        This system uses **machine learning** to predict whether a given DNA sequence is at risk of
        being inadvertently edited by a specific sgRNA.

        ---

        ### How It Works

        **1. Feature Engineering** — Raw DNA sequences are converted into numerical features:

        | Feature | Description |
        |---------|-------------|
        | Mismatch Count | Number of base differences between sgRNA and target |
        | Similarity Score | Proportion of matching positions |
        | Seed Region Mismatch | Mismatches in PAM-proximal 12 bases (most critical) |
        | Max Consecutive Mismatch | Longest run of consecutive mismatches |
        | GC Content | Fraction of G+C bases in each sequence |
        | Distal Region Mismatch | Mismatches in PAM-distal 8 bases |
        | First-Half Mismatch | Mismatches in the first half of the alignment |

        **2. Model Training** — A Random Forest classifier (100 trees, depth-12) and Logistic Regression
        were trained on the combined CIRCLE-seq + GUIDE-seq datasets.

        **3. Prediction** — Given a new sgRNA + target pair, features are extracted and the model
        returns a probability score. Scores above 65% are flagged as HIGH risk (NOT SAFE).

        ---

        ### Datasets Used

        - **CIRCLE-seq** — Comprehensive *in vitro* detection of CRISPR-Cas9 off-targets using
          circularization. Provides genome-wide off-target detection with high sensitivity.

        - **GUIDE-seq** — Genome-wide, unbiased identification of DSBs enabled by sequencing.
          A highly sensitive method using double-stranded oligonucleotide (dsODN) tags.
        """)

    with col_r:
        st.markdown("""
        ### Key Concepts

        **Seed Region**
        The ~12 bases immediately adjacent to the PAM (Protospacer Adjacent Motif) are the
        "seed region." Mismatches here are most disruptive to Cas9 binding.

        **PAM Sequence**
        A short DNA motif (NGG for SpCas9) required for Cas9 to recognize and cut DNA.
        The target must have a PAM adjacent to the protospacer.

        **Mismatch Tolerance**
        SpCas9 can tolerate up to 3–5 mismatches, especially in the PAM-distal region.
        This is why off-target prediction is non-trivial.

        ---

        ### Limitations

        - Predictions are probabilistic, not deterministic
        - Model trained on *in vitro* data (CIRCLE-seq), which may differ from *in vivo* editing
        - Epigenetic factors (chromatin accessibility) are not captured
        - Always validate computationally predicted off-targets experimentally

        ---

        ### Tech Stack

        - **Python** 3.x
        - **scikit-learn** — Random Forest, Logistic Regression
        - **pandas / numpy** — Data processing
        - **Streamlit** — Web interface
        - **matplotlib / seaborn** — Visualization
        """)

        st.markdown("""
        <div class="info-box" style='margin-top:1rem'>
          ⚠️ <strong>Disclaimer</strong><br>
          This tool is intended for <em>research and educational purposes only</em>.
          It should not be used as the sole basis for clinical or therapeutic decisions.
          Always consult domain experts and perform experimental validation.
        </div>
        """, unsafe_allow_html=True)
