"""Page: side-by-side comparison of all trained models."""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config as cfg

st.set_page_config(page_title="Model Comparison", page_icon="🤖", layout="wide")

st.markdown("## 🤖 Model Performance Comparison")

viz_dir = cfg.VIZ_DIR

# ----- load metrics -----
@st.cache_data(show_spinner=False)
def load_metrics():
    with open(cfg.MODEL_DIR / "metrics.json") as fh:
        return json.load(fh)

@st.cache_data(show_spinner=False)
def load_cv():
    with open(cfg.MODEL_DIR / "cv_results.json") as fh:
        return json.load(fh)

@st.cache_data(show_spinner=False)
def load_timings():
    with open(cfg.MODEL_DIR / "timings.json") as fh:
        return json.load(fh)

metrics = load_metrics()
cv = load_cv()
timings = load_timings()

# ----- build comparison table -----
rows = []
pretty_names = {
    "logistic_regression": "Logistic Regression",
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest",
    "neural_network": "Neural Network",
    "word2vec_lr": "Word2Vec + LR",
}
for name, m in metrics.items():
    cv_info = cv.get(name, {})
    tim = timings.get(name, {})
    rows.append({
        "Model": pretty_names.get(name, name),
        "Accuracy": f"{m['accuracy']:.4f}",
        "Precision": f"{m['precision']:.4f}",
        "Recall": f"{m['recall']:.4f}",
        "F1 Score": f"{m['f1']:.4f}",
        "ROC AUC": f"{m['roc_auc']:.4f}",
        "CV F1 (mean±std)": (f"{cv_info.get('mean', '-'):.4f} ± "
                             f"{cv_info.get('std', '-'):.4f}"),
        "Train time (s)": f"{tim.get('train_seconds', '-'):.2f}",
    })

table = pd.DataFrame(rows)
st.dataframe(table, use_container_width=True, hide_index=True)

# ----- highlight best model -----
best = max(metrics, key=lambda k: metrics[k]["f1"])
best_pretty = pretty_names.get(best, best)
st.success(f"**Best performing model:** {best_pretty} "
           f"(F1 = {metrics[best]['f1']:.4f}, Accuracy = {metrics[best]['accuracy']:.4f})")

# ----- visualisations -----
st.markdown("### Visualisations")

col1, col2 = st.columns(2)

metric_bars = viz_dir / "metric_bars.png"
roc_path = viz_dir / "roc_curves.png"
cm_lr = viz_dir / "cm_logistic_regression.png"
feat_imp = viz_dir / "feature_importance.png"
timings_png = viz_dir / "timings.png"
cv_box = viz_dir / "cv_boxplot.png"
pr_path = viz_dir / "pr_curves.png"

with col1:
    if metric_bars.exists():
        st.image(str(metric_bars), caption="Model Metrics Comparison",
                 use_container_width=True)
    if roc_path.exists():
        st.image(str(roc_path), caption="ROC Curves", use_container_width=True)
    if cv_box.exists():
        st.image(str(cv_box), caption="Cross-Validation F1 Scores",
                 use_container_width=True)

with col2:
    if pr_path.exists():
        st.image(str(pr_path), caption="Precision-Recall Curves",
                 use_container_width=True)
    if timings_png.exists():
        st.image(str(timings_png), caption="Training & Inference Timings",
                 use_container_width=True)
    if feat_imp.exists():
        st.image(str(feat_imp), caption="Feature Importance (Logistic Regression)",
                 use_container_width=True)

st.markdown("### Confusion Matrices")
cm_cols = st.columns(4)
for i, name in enumerate(["logistic_regression", "naive_bayes", "random_forest", "neural_network"]):
    cm_path = viz_dir / f"cm_{name}.png"
    if cm_path.exists():
        cm_cols[i].image(str(cm_path), caption=pretty_names.get(name, name),
                         use_container_width=True)

# ----- feature importance deep dive -----
st.markdown("### Top TF-IDF Tokens for Logistic Regression")
lr_tokens = viz_dir / "lr_top_tokens.png"
if lr_tokens.exists():
    st.image(str(lr_tokens), caption="Top 18 tokens with highest logistic regression coefficients",
             use_container_width=True)

# ----- count vs tfidf comparison -----
st.markdown("### CountVectorizer vs TF-IDF")
count_vs = viz_dir / "count_vs_tfidf.png"
if count_vs.exists():
    st.image(str(count_vs), use_container_width=True)

# ----- Word2Vec t-SNE -----
st.markdown("### Word2Vec Embedding Space (t-SNE)")
w2v_tsne = viz_dir / "w2v_tsne.png"
if w2v_tsne.exists():
    st.image(str(w2v_tsne), caption="t-SNE projection of Word2Vec word vectors",
             use_container_width=True)

# ----- metadata correlation -----
st.markdown("### Metadata Feature Correlation")
meta_corr = viz_dir / "meta_correlation.png"
if meta_corr.exists():
    st.image(str(meta_corr), use_container_width=True)
