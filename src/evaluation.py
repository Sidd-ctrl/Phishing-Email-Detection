"""Evaluation + every plot used in the notebook, report and slides.

Runs entirely off the artifacts written by train_models, so it can be
re-executed any number of times without retraining anything.
"""

from __future__ import annotations

import json
import pickle

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import load_npz
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from . import config as cfg
from . import data_loader

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.dpi": cfg.FIG_DPI,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#444",
    "savefig.bbox": "tight",
})

PRETTY = {
    "logistic_regression": "Logistic Regression",
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest",
    "neural_network": "Neural Network",
    "word2vec_lr": "Word2Vec + LR",
}
MODEL_ORDER = ["logistic_regression", "naive_bayes", "random_forest",
               "neural_network"]


def _load_predictions():
    return dict(np.load(cfg.MODEL_DIR / "test_predictions.npz"))


def _load_metrics():
    with open(cfg.MODEL_DIR / "metrics.json") as fh:
        return json.load(fh)


# ============================================================ dataset plots

def plot_class_distribution():
    df = pd.read_csv(cfg.RAW_CSV).dropna(subset=["Email Text"])
    counts = df["Email Type"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 4.8))
    bars = ax.bar(counts.index, counts.values,
                  color=[cfg.PALETTE["safe"], cfg.PALETTE["phishing"]],
                  edgecolor="black", width=0.55)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 150,
                f"{val:,}\n({val / counts.sum():.1%})",
                ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_title("Class distribution in the raw dataset")
    ax.set_ylabel("Number of emails")
    ax.set_xlabel("")
    sns.despine()
    fig.savefig(cfg.VIZ_DIR / "class_distribution.png")
    plt.close(fig)


def plot_length_distribution():
    with open(cfg.CLEAN_CSV.with_suffix(".pkl"), "rb") as fh:
        clean = pickle.load(fh)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label, color in zip(("Safe Email", "Phishing Email"),
                            (cfg.PALETTE["safe"], cfg.PALETTE["phishing"])):
        lengths = clean.loc[clean["Email Type"] == label, "raw_length"]
        lengths = lengths[lengths < 6000]          # trim the long tail
        sns.kdeplot(lengths, ax=ax, label=label, color=color,
                    linewidth=2.2, fill=True, alpha=0.15)
    ax.set_title("Email length (characters) by class")
    ax.set_xlabel("Character count (clipped at 6000)")
    ax.legend()
    sns.despine()
    fig.savefig(cfg.VIZ_DIR / "length_distribution.png")
    plt.close(fig)


def plot_wordclouds():
    from wordcloud import WordCloud

    with open(cfg.CLEAN_CSV.with_suffix(".pkl"), "rb") as fh:
        clean = pickle.load(fh)

    for label, fname, color in (
        ("Phishing Email", "wordcloud_phishing.png", "Reds"),
        ("Safe Email", "wordcloud_safe.png", "Greens"),
    ):
        text = " ".join(clean.loc[clean["Email Type"] == label, "clean_text"])
        wc = WordCloud(width=1400, height=700, colormap=color,
                       background_color="white", max_words=120).generate(text)
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"Most frequent tokens - {label.lower()}s")
        ax.axis("off")
        fig.savefig(cfg.VIZ_DIR / fname)
        plt.close(fig)


def plot_keyword_frequency():
    """For each suspicious keyword, how much more often it appears in
    phishing mail than safe mail."""
    with open(cfg.CLEAN_CSV.with_suffix(".pkl"), "rb") as fh:
        clean = pickle.load(fh)

    phish = clean.loc[clean["Email Type"] == "Phishing Email", "raw_text"].str.lower()
    safe = clean.loc[clean["Email Type"] == "Safe Email", "raw_text"].str.lower()
    n_p, n_s = len(phish), len(safe)

    rows = []
    for kw in cfg.SUSPICIOUS_KEYWORDS:
        p_rate = phish.str.contains(rf"\b{kw}\b", regex=True).mean()
        s_rate = safe.str.contains(rf"\b{kw}\b", regex=True).mean()
        ratio = p_rate / max(s_rate, 1e-4)
        rows.append({"keyword": kw, "phish_pct": p_rate * 100,
                     "safe_pct": s_rate * 100, "ratio": ratio})
    kw_df = pd.DataFrame(rows).sort_values("ratio", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(kw_df))[::-1]
    ax.barh(y + 0.18, kw_df["phish_pct"], height=0.36,
            color=cfg.PALETTE["phishing"], label="% of phishing emails")
    ax.barh(y - 0.18, kw_df["safe_pct"], height=0.36,
            color=cfg.PALETTE["safe"], label="% of safe emails")
    ax.set_yticks(y)
    ax.set_yticklabels(kw_df["keyword"])
    ax.set_xlabel("% of emails containing the word")
    ax.set_title("Suspicious words: phishing vs safe occurrence")
    ax.legend(loc="lower right", fontsize=12)
    sns.despine()
    fig.savefig(cfg.VIZ_DIR / "keyword_frequency.png")
    plt.close(fig)


def plot_meta_correlation():
    meta = pd.read_csv(cfg.FEATURES_CSV).drop(columns=["label"])
    corr = meta.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.3, cbar_kws={"shrink": 0.8},
                ax=ax, annot=False)
    ax.set_title("Correlation between handcrafted features (training set)")
    fig.savefig(cfg.VIZ_DIR / "meta_correlation.png")
    plt.close(fig)


def plot_embedding_tsne():
    """2-D t-SNE of the Word2Vec email vectors - the two classes should
    visibly separate if the embeddings carry signal."""
    from sklearn.manifold import TSNE

    dense = np.load(cfg.PROCESSED_DIR / "matrices" / "dense.npz")
    W = np.vstack([dense["W_train"], dense["W_test"]])
    y = np.concatenate([dense["y_train"], dense["y_test"]])
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    idx = rng.choice(len(W), size=3000, replace=False)

    tsne = TSNE(n_components=2, perplexity=30, random_state=cfg.RANDOM_STATE,
                learning_rate="auto", init="pca")
    emb = tsne.fit_transform(W[idx])

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for lab, color, name in ((0, cfg.PALETTE["safe"], "Safe"),
                             (1, cfg.PALETTE["phishing"], "Phishing")):
        mask = y[idx] == lab
        ax.scatter(emb[mask, 0], emb[mask, 1], s=9, color=color,
                   label=name, alpha=0.65, edgecolors="none")
    ax.set_title("t-SNE of Word2Vec email embeddings (3,000 sample)")
    ax.legend(markerscale=3)
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(left=True, bottom=True)
    fig.savefig(cfg.VIZ_DIR / "w2v_tsne.png")
    plt.close(fig)


# ============================================================ model plots

def plot_confusion_matrices():
    preds = _load_predictions()
    y_test = preds["y_test"]
    for name in MODEL_ORDER:
        cm = confusion_matrix(y_test, preds[f"{name}_pred"])
        fig, ax = plt.subplots(figsize=(5.6, 4.8))
        disp = ConfusionMatrixDisplay(cm, display_labels=["Safe", "Phishing"])
        disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
        ax.set_title(f"Confusion matrix - {PRETTY[name]}")
        fig.savefig(cfg.VIZ_DIR / f"cm_{name}.png")
        plt.close(fig)


def plot_roc_curves():
    preds = _load_predictions()
    y_test = preds["y_test"]
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    for name, color in zip(MODEL_ORDER, cfg.PALETTE["models"]):
        fpr, tpr, _ = roc_curve(y_test, preds[f"{name}_proba"])
        with open(cfg.MODEL_DIR / "metrics.json") as fh:
            auc = json.load(fh)[name]["roc_auc"]
        ax.plot(fpr, tpr, color=color, linewidth=2.2,
                label=f"{PRETTY[name]} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the holdout test set")
    ax.legend(loc="lower right", fontsize=12)
    sns.despine()
    fig.savefig(cfg.VIZ_DIR / "roc_curves.png")
    plt.close(fig)


def plot_pr_curves():
    preds = _load_predictions()
    y_test = preds["y_test"]
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    for name, color in zip(MODEL_ORDER, cfg.PALETTE["models"]):
        prec, rec, _ = precision_recall_curve(y_test, preds[f"{name}_proba"])
        ax.plot(rec, prec, color=color, linewidth=2.2, label=PRETTY[name])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (positive class = phishing)")
    ax.legend(loc="lower left", fontsize=12)
    sns.despine()
    fig.savefig(cfg.VIZ_DIR / "pr_curves.png")
    plt.close(fig)


def plot_metric_bars():
    metrics = _load_metrics()
    rows = []
    for name in MODEL_ORDER:
        for metric in ("accuracy", "precision", "recall", "f1"):
            rows.append({"model": PRETTY[name], "metric": metric,
                         "value": metrics[name][metric]})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 5.4))
    sns.barplot(data=df, x="model", y="value", hue="metric", ax=ax,
                palette=cfg.PALETTE["models"], edgecolor="black",
                linewidth=0.6, order=[PRETTY[m] for m in MODEL_ORDER])
    ax.set_ylim(0.85, 1.01)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_title("Holdout test scores by model")
    ax.legend(title="Metric", ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    ax.tick_params(axis="x", labelsize=12)
    sns.despine()
    fig.savefig(cfg.VIZ_DIR / "metric_bars.png")
    plt.close(fig)


def plot_timings():
    with open(cfg.MODEL_DIR / "timings.json") as fh:
        timings = json.load(fh)

    names = [PRETTY[m] for m in MODEL_ORDER]
    train_t = [timings[m]["train_seconds"] for m in MODEL_ORDER]
    pred_t = [timings[m]["predict_ms_per_email"] for m in MODEL_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    bars = axes[0].bar(names, train_t, color=cfg.PALETTE["models"],
                       edgecolor="black")
    axes[0].set_title("Total tuning + training time (s)")
    axes[0].set_ylabel("Seconds")
    for bar, val in zip(bars, train_t):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val, f"{val:.0f}s",
                     ha="center", va="bottom", fontsize=11)
    axes[0].tick_params(axis="x", labelrotation=18, labelsize=11)

    bars = axes[1].bar(names, pred_t, color=cfg.PALETTE["models"],
                       edgecolor="black")
    axes[1].set_title("Prediction latency (ms per email)")
    axes[1].set_ylabel("Milliseconds")
    for bar, val in zip(bars, pred_t):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}",
                     ha="center", va="bottom", fontsize=11)
    axes[1].tick_params(axis="x", labelrotation=18, labelsize=11)

    sns.despine(fig=fig)
    fig.tight_layout()
    fig.savefig(cfg.VIZ_DIR / "timings.png")
    plt.close(fig)


def plot_feature_importance():
    rf = joblib.load(cfg.MODEL_DIR / "random_forest.joblib")
    with open(cfg.MODEL_DIR / "feature_bundle.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    names = list(bundle.tfidf.get_feature_names_out()) + list(bundle.meta_columns)

    imp = pd.Series(rf.feature_importances_, index=names)
    top = imp.sort_values(ascending=False).head(18)

    fig, ax = plt.subplots(figsize=(9, 7))
    top = top.sort_values(ascending=True)
    colors = [cfg.PALETTE["accent"] if n in bundle.meta_columns
              else "#8d99ae" for n in top.index]
    ax.barh(range(len(top)), top.values, color=colors, edgecolor="black")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=11)
    ax.set_xlabel("Gini importance")
    ax.set_title("Top 18 features - Random Forest")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=cfg.PALETTE["accent"], label="handcrafted"),
                       Patch(color="#8d99ae", label="TF-IDF token")],
              loc="lower right")
    sns.despine()
    fig.savefig(cfg.VIZ_DIR / "feature_importance.png")
    plt.close(fig)


def plot_lr_top_tokens():
    lr = joblib.load(cfg.MODEL_DIR / "logistic_regression.joblib")
    with open(cfg.MODEL_DIR / "feature_bundle.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    names = list(bundle.tfidf.get_feature_names_out())
    coef = lr.coef_[0][: len(names)]

    top_phish = pd.Series(coef, index=names).sort_values(ascending=False).head(12)
    top_safe = pd.Series(coef, index=names).sort_values().head(12)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=False)
    axes[0].barh(top_phish.index[::-1], top_phish.values[::-1],
                 color=cfg.PALETTE["phishing"], edgecolor="black")
    axes[0].set_title("Tokens pushing towards PHISHING")
    axes[1].barh(top_safe.index[::-1], top_safe.values[::-1],
                 color=cfg.PALETTE["safe"], edgecolor="black")
    axes[1].set_title("Tokens pushing towards SAFE")
    for ax in axes:
        ax.set_xlabel("Logistic regression coefficient")
    sns.despine(fig=fig)
    fig.tight_layout()
    fig.savefig(cfg.VIZ_DIR / "lr_top_tokens.png")
    plt.close(fig)


def plot_cv_boxplot():
    with open(cfg.MODEL_DIR / "cv_results.json") as fh:
        cv = json.load(fh)
    data = {PRETTY[name]: cv[name]["folds_f1"] for name in MODEL_ORDER}

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bp = ax.boxplot(data.values(), labels=data.keys(), patch_artist=True,
                    widths=0.55)
    for patch, color in zip(bp["boxes"], cfg.PALETTE["models"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_ylabel("F1 score per fold")
    ax.set_title(f"{cfg.CV_FOLDS}-fold cross-validation (training split)")
    ax.tick_params(axis="x", labelrotation=12)
    sns.despine()
    fig.savefig(cfg.VIZ_DIR / "cv_boxplot.png")
    plt.close(fig)


def plot_count_vs_tfidf():
    """Quick head-to-head: same LR, count features vs TF-IDF features."""
    from sklearn.linear_model import LogisticRegression

    X_c_train = load_npz(cfg.PROCESSED_DIR / "matrices" / "X_count_train.npz")
    X_c_test = load_npz(cfg.PROCESSED_DIR / "matrices" / "X_count_test.npz")
    X_t_train = load_npz(cfg.PROCESSED_DIR / "matrices" / "X_tfidf_train.npz")
    X_t_test = load_npz(cfg.PROCESSED_DIR / "matrices" / "X_tfidf_test.npz")
    dense = np.load(cfg.PROCESSED_DIR / "matrices" / "dense.npz")
    y_train, y_test = dense["y_train"], dense["y_test"]

    scores = {}
    for label, X_tr, X_te in (("CountVectorizer", X_c_train, X_c_test),
                              ("TF-IDF", X_t_train, X_t_test)):
        lr = LogisticRegression(max_iter=1500, random_state=cfg.RANDOM_STATE)
        lr.fit(X_tr, y_train)
        scores[label] = lr.score(X_te, y_test)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    bars = ax.bar(scores.keys(), scores.values(),
                  color=[cfg.PALETTE["accent"], cfg.PALETTE["safe"]],
                  edgecolor="black", width=0.5)
    ax.set_ylim(0.9, 1.0)
    for bar, val in zip(bars, scores.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
                f"{val:.4f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Test accuracy (same Logistic Regression)")
    ax.set_title("Count features vs TF-IDF weighting")
    sns.despine()
    fig.savefig(cfg.VIZ_DIR / "count_vs_tfidf.png")
    plt.close(fig)
    return scores


# ============================================================ architecture

def plot_architecture():
    """Hand-drawn style system diagram used in the report and slides."""
    fig, ax = plt.subplots(figsize=(13, 6.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=11):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color,
                                   edgecolor="black", linewidth=1.4,
                                   zorder=3, alpha=0.95))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", zorder=4, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#333",
                                    lw=1.8), zorder=2)

    box(0.3, 2.9, 2.0, 1.2, "Raw emails\n(Kaggle dataset)", "#ffd6a5")
    box(2.9, 2.9, 2.1, 1.2, "Preprocessing\nHTML • URLs • tokens\nlemmatization", "#fdffb6")
    box(5.6, 5.0, 2.2, 1.0, "TF-IDF + metadata", "#caffbf")
    box(5.6, 3.5, 2.2, 1.0, "Word2Vec averages", "#9bf6ff")
    box(5.6, 2.0, 2.2, 1.0, "SVD dense vectors", "#a0c4ff")
    box(8.4, 2.9, 2.1, 1.6, "Model zoo\nLR • NB • RF • MLP\n(5-fold CV)", "#bdb2ff")
    box(11.0, 2.9, 1.8, 1.2, "Evaluation\n& metrics", "#ffc6ff")
    box(8.4, 0.4, 2.1, 1.2, "Streamlit app\nrisk score + XAI", "#ffadad")

    arrow(2.3, 3.5, 2.9, 3.5)
    arrow(5.0, 3.9, 5.6, 5.3)
    arrow(5.0, 3.5, 5.6, 4.0)
    arrow(5.0, 3.1, 5.6, 2.5)
    arrow(7.8, 5.4, 8.5, 4.3)
    arrow(7.8, 4.0, 8.4, 3.8)
    arrow(7.8, 2.5, 8.5, 3.2)
    arrow(10.5, 3.7, 11.0, 3.6)
    arrow(9.45, 2.9, 9.45, 1.6)

    ax.set_title("System architecture - AI-driven phishing email detection",
                 fontsize=15, pad=12)
    fig.savefig(cfg.VIZ_DIR / "architecture.png")
    plt.close(fig)


# ============================================================ entrypoint

def run_all():
    cfg.VIZ_DIR.mkdir(parents=True, exist_ok=True)
    plot_class_distribution()
    plot_length_distribution()
    plot_wordclouds()
    plot_keyword_frequency()
    plot_meta_correlation()
    plot_embedding_tsne()
    plot_confusion_matrices()
    plot_roc_curves()
    plot_pr_curves()
    plot_metric_bars()
    plot_timings()
    plot_feature_importance()
    plot_lr_top_tokens()
    plot_cv_boxplot()
    count_vs = plot_count_vs_tfidf()
    plot_architecture()
    print("[eval] all figures written to", cfg.VIZ_DIR)

    # comparison table as CSV
    metrics = _load_metrics()
    with open(cfg.MODEL_DIR / "timings.json") as fh:
        timings = json.load(fh)
    with open(cfg.MODEL_DIR / "cv_results.json") as fh:
        cv = json.load(fh)

    rows = []
    for name in MODEL_ORDER:
        rows.append({
            "Model": PRETTY[name],
            "Accuracy": metrics[name]["accuracy"],
            "Precision": metrics[name]["precision"],
            "Recall": metrics[name]["recall"],
            "F1": metrics[name]["f1"],
            "ROC AUC": metrics[name]["roc_auc"],
            "CV F1 mean±std": f"{cv[name]['mean']} ± {cv[name]['std']}",
            "Train time (s)": timings[name]["train_seconds"],
            "Predict (ms/email)": timings[name]["predict_ms_per_email"],
        })
    table = pd.DataFrame(rows)
    table.to_csv(cfg.MODEL_DIR / "comparison_table.csv", index=False)
    print(table.to_string(index=False))
    print("[eval] count vs tfidf accuracy:", count_vs)


if __name__ == "__main__":
    run_all()
