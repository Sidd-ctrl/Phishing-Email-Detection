"""Page: dataset exploration - class balance, word clouds, patterns."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from wordcloud import WordCloud

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config as cfg

st.set_page_config(page_title="Dataset Statistics", page_icon="📊", layout="wide")

st.markdown("## 📊 Dataset Exploration")

viz_dir = cfg.VIZ_DIR
viz_dir.mkdir(parents=True, exist_ok=True)

# ----- load data -----
@st.cache_data(show_spinner=False)
def load_clean():
    import pickle
    pkl_path = cfg.CLEAN_CSV.with_suffix(".pkl")
    if pkl_path.exists():
        with open(pkl_path, "rb") as fh:
            return pickle.load(fh)
    # fallback to CSV
    csv_path = cfg.CLEAN_CSV
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None

df = load_clean()

if df is None:
    st.error("Cleaned data not found. Run the preprocessing pipeline first.")
    st.stop()

# ----- class distribution -----
st.markdown("### Class Distribution")
c1, c2 = st.columns(2)

safe_count = int((df["label"] == 0).sum())
phish_count = int((df["label"] == 1).sum())
total = safe_count + phish_count

c1.metric("Safe Emails", f"{safe_count:,}", f"{safe_count/total:.1%}")
c1.metric("Phishing Emails", f"{phish_count:,}", f"{phish_count/total:.1%}")

with c2:
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Safe Email", "Phishing Email"]
    sizes = [safe_count, phish_count]
    colors = ["#2a9d8f", "#e63946"]
    explode = (0, 0.05)
    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct="%1.1f%%", startangle=90, shadow=False)
    ax.axis("equal")
    st.pyplot(fig)
    plt.close()

# ----- word clouds -----
st.markdown("### Word Clouds")
wc1, wc2 = st.columns(2)

# check if pre-generated images exist
wc_phish = viz_dir / "wordcloud_phishing.png"
wc_safe = viz_dir / "wordcloud_safe.png"

if wc_phish.exists() and wc_safe.exists():
    wc1.image(str(wc_phish), caption="Phishing Emails", use_container_width=True)
    wc2.image(str(wc_safe), caption="Safe Emails", use_container_width=True)
else:
    # generate on the fly
    with st.spinner("Generating word clouds ..."):
        phish_text = " ".join(df[df["label"] == 1]["clean_text"].dropna().tolist())
        safe_text = " ".join(df[df["label"] == 0]["clean_text"].dropna().tolist())

        wc = WordCloud(width=800, height=400, background_color="white",
                       max_words=120, colormap="Reds")
        wc.generate(phish_text[:500000] if len(phish_text) > 500000 else phish_text)
        wc1.pyplot(wc.figure)
        plt.close()

        wc2 = WordCloud(width=800, height=400, background_color="white",
                        max_words=120, colormap="Greens")
        wc2.generate(safe_text[:500000] if len(safe_text) > 500000 else safe_text)
        wc2.pyplot(wc.figure)
        plt.close()

# ----- length distribution -----
st.markdown("### Email Length Distribution")
lenfig_path = viz_dir / "length_distribution.png"
if lenfig_path.exists():
    st.image(str(lenfig_path), use_container_width=True)
else:
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, df["raw_length"].quantile(0.98), 60)
    ax.hist(df[df["label"] == 0]["raw_length"], bins=bins, alpha=0.6,
            label="Safe", color="#2a9d8f")
    ax.hist(df[df["label"] == 1]["raw_length"], bins=bins, alpha=0.6,
            label="Phishing", color="#e63946")
    ax.set_xlabel("Character count")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
    plt.close()

# ----- keyword frequency comparison -----
st.markdown("### Suspicious Keyword Frequency")
kf_path = viz_dir / "keyword_frequency.png"
if kf_path.exists():
    st.image(str(kf_path), use_container_width=True)
else:
    st.info("Keyword frequency chart not pre-generated. Run the pipeline with evaluate stage.")

# ----- sender domains -----
st.markdown("### Sender Domain Coverage")
domains = df["sender_domain"].value_counts()
known = domains.drop("unknown", errors="ignore").sum()
unknown = domains.get("unknown", 0)
st.metric("Emails with identifiable sender domain",
          f"{known:,}", f"{known/total:.1%}")
st.metric("Emails with unknown sender",
          f"{unknown:,}", f"{unknown/total:.1%}")
if len(domains) > 1:
    st.markdown("**Top 10 sender domains:**")
    top_domains = domains.head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    top_domains.plot(kind="barh", ax=ax, color="#457b9d")
    ax.set_xlabel("Count")
    ax.invert_yaxis()
    st.pyplot(fig)
    plt.close()
