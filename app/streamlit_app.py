"""PhishGuard - home page.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402

st.set_page_config(
    page_title="PhishGuard - AI Phishing Email Detector",
    page_icon="🛡️",
    layout="wide",
)
st.markdown(
    f"<style>{(ROOT / 'app' / 'assets' / 'style.css').read_text()}</style>",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def dataset_facts():
    df = pd.read_csv(cfg.RAW_CSV)
    df = df.dropna(subset=["Email Text"]).drop_duplicates("Email Text")
    counts = df["Email Type"].value_counts()
    return {
        "total": len(df),
        "safe": int(counts.get("Safe Email", 0)),
        "phish": int(counts.get("Phishing Email", 0)),
    }


@st.cache_data(show_spinner=False)
def best_model_facts():
    import json

    metrics = json.loads((cfg.MODEL_DIR / "metrics.json").read_text())
    cv = json.loads((cfg.MODEL_DIR / "cv_results.json").read_text())
    best = max(metrics, key=lambda k: metrics[k]["f1"])
    return metrics[best], cv.get(best, {}), best


facts = dataset_facts()

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ PhishGuard</h1>
        <div class="tagline">
        An AI-driven phishing email detector built with classical NLP and
        machine learning. Paste any email below the <b>Analyze Email</b> tab
        and get a verdict, a 0-100 risk score, highlighted bait words,
        per-link safety checks and a plain-English explanation of
        <i>why</i> the call was made.
        </div>
        <div class="badge-row">
            <span>18,600+ training emails</span>
            <span class="teal">TF-IDF + Word2Vec + metadata</span>
            <span class="blue">4 models compared</span>
            <span>Explainable verdicts</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Emails in dataset", f"{facts['total']:,}")
m2.metric("Phishing samples", f"{facts['phish']:,}")
m3.metric("Safe samples", f"{facts['safe']:,}")

score, cv_info, best_name = best_model_facts()
pretty = {
    "logistic_regression": "Logistic Regression",
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest",
    "neural_network": "Neural Network",
}.get(best_name, best_name)
m4.metric("Best model F1", f"{score['f1']:.3f}", delta=pretty)

st.markdown("### How it works")
c1, c2, c3 = st.columns(3)
c1.markdown(
    """
    <div class="card"><h4>1 · Read & clean</h4>
    HTML tags, tracking links and noise are stripped. The remaining text is
    tokenised and lemmatised, while URLs, sender domains and writing-style
    statistics are pulled out as metadata.</div>
    """, unsafe_allow_html=True,
)
c2.markdown(
    """
    <div class="card"><h4>2 · Score</h4>
    A tuned classifier outputs a phishing probability. Transparent
    heuristics (link traps, urgency phrases, typosquatting) add points on
    top to form a 0-100 risk score you can audit by hand.</div>
    """, unsafe_allow_html=True,
)
c3.markdown(
    """
    <div class="card"><h4>3 · Explain</h4>
    Bait words are highlighted in the original text, every link is safety
    checked, the sender domain gets a reputation verdict, and the full
    reasoning chain is listed - no black box.</div>
    """, unsafe_allow_html=True,
)

st.markdown("### Explore")
st.markdown(
    """
    Use the sidebar to jump between sections:

    - 🔍 **Analyze Email** - paste a mail and get the full forensic breakdown
    - 📊 **Dataset Statistics** - class balance, word clouds, keyword traps
    - 🤖 **Model Comparison** - metrics, ROC curves, timings, feature importance
    - ℹ️ **About** - architecture, methodology and honest limitations
    """
)

# small count-up animation on the hero metrics (purely cosmetic)
components.html(
    """
    <script>
    const targets = document.parentWindow ? null : null;
    function animate(el, end){
        let start = 0; const step = Math.max(1, Math.floor(end/45));
        const t = setInterval(() => {
            start += step;
            if (start >= end){ start = end; clearInterval(t); }
            el.innerText = start.toLocaleString();
        }, 22);
    }
    window.addEventListener('load', () => {
        document.querySelectorAll('[data-testid="stMetricValue"]').forEach(el => {
            const raw = el.innerText.replace(/[^0-9]/g,'');
            const n = parseInt(raw || '0');
            if (n > 100) animate(el, n);
        });
    });
    </script>
    """,
    height=0,
)
