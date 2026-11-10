"""Page: project overview, methodology, tech stack and limitations."""

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config as cfg

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

st.markdown("## ℹ️ About This Project")

# architecture diagram
arch_path = cfg.VIZ_DIR / "architecture.png"
if arch_path.exists():
    st.image(str(arch_path), caption="System Architecture",
             use_container_width=True)

st.markdown("""
### Project Overview

**AI-Driven Phishing Email Detection Using NLP** is a final-year university
project that explores how a combination of classical natural language
processing techniques and machine learning can be used to detect phishing
emails. The system processes raw email text through a series of cleaning
and feature extraction stages, then applies tuned classifiers to produce
a transparent, explainable verdict.

---

### Methodology

The pipeline is divided into five stages:

1. **Data Collection & Cleaning** — The Kaggle Phishing Email Dataset
   (~18,650 emails) is loaded, duplicate bodies are dropped, empty rows
   are discarded. Each email is stripped of HTML tags, URLs are extracted
   and blanked out, and the plain text is tokenised and POS-aware
   lemmatised.

2. **Feature Engineering** — Three complementary representations are built:
   - **TF-IDF** (15,000 unigrams + bigrams) — captures word salience
   - **CountVectorizer** (15,000 unigrams + bigrams) — used by Naive Bayes
   - **Word2Vec** (100-d embeddings trained on the corpus) — semantic similarity
   - **Hand-crafted metadata** — URL count, sender domain reputation,
     exclamation count, uppercase ratio, generic greeting detection, and
     more — 20+ features engineered from domain knowledge.

3. **Model Training** — Four classifiers are tuned via cross-validation:
   - **Logistic Regression** (L2, solver=lbfgs, tuned C)
   - **Multinomial Naive Bayes** (tuned alpha and fit_prior)
   - **Random Forest** (400 trees, tuned depth/samples/class_weight)
   - **MLP Classifier** (1-2 hidden layers, tuned architecture)
   - **Bonus:** Word2Vec + Logistic Regression baseline

4. **Evaluation** — Models are scored on a held-out 20% test set using
   accuracy, precision, recall, F1, and ROC AUC. 5-fold cross-validation
   is reported for F1 stability. Train-time and per-email inference latency
   are also measured.

5. **Explainable Inference** — The app runs emails through the same pipeline,
   shows a 0-100 risk score (70% model probability + 30% transparent
   heuristics), highlights bait words in the original text, checks every
   URL for red flags, evaluates the sender domain for typosquatting, and
   lists plain-English reasons for the verdict.

---

### Tech Stack

| Component | Library / Tool |
|---|---|
| Data processing | pandas, numpy, scipy |
| Text cleaning | NLTK, regex, html.parser |
| Feature extraction | scikit-learn (TF-IDF, CountVectorizer, SVD), Gensim (Word2Vec) |
| Machine learning | scikit-learn (LR, NB, RF, MLP) |
| Visualisation | matplotlib, seaborn, wordcloud, plotly |
| App / deployment | Streamlit |
| Report / slides | python-docx, python-pptx, reportlab |

---

### Dataset

[Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)
from Kaggle, mirrored on Hugging Face. The dataset contains ~18,650 email
bodies labelled as "Phishing Email" or "Safe Email". The corpus is
moderately imbalanced (~61% safe, ~39% phishing), which is realistic for
production environments.

### Classifier Performance (Holdout Set)

| Model | F1 | Accuracy | ROC AUC | Train time |
|---|---|---|---|---|
| Logistic Regression | **0.9754** | 0.9815 | 0.9969 | 9.9 s |
| Neural Network | 0.9733 | 0.9800 | 0.9967 | 30.5 s |
| Naive Bayes | 0.9708 | 0.9781 | 0.9967 | 0.4 s |
| Random Forest | 0.9707 | 0.9781 | 0.9970 | 327.1 s |
| Word2Vec + LR | 0.9541 | 0.9661 | 0.9928 | 1.5 s |

---

### Limitations & Honest Caveats

- **Dataset constraints:** The Kaggle dataset stores only email *bodies*.
  Most rows (~89%) lack MIME headers, so sender-domain reputation is
  unavailable for most samples. Real-world deployments would use full
  RFC-822 messages.
- **Temporal drift:** The dataset was collected several years ago; phishing
  language and infrastructure evolve quickly. Periodic retraining would be
  needed.
- **Adversarial robustness:** A motivated attacker could craft emails that
  exploit the known feature representations. The explainability layer helps
  but does not guarantee detection.
- **Computational cost:** The Random Forest takes ~5 minutes to train on a
  laptop CPU; the tuned MLP takes ~30 seconds. All other models train in
  under 10 seconds.

---

### Future Work

- Integrate full email headers (SPF, DKIM, DMARC checks)
- Graph-based analysis of URL redirect chains
- Deep learning (BERT / RoBERTa fine-tuning for sequence classification)
- Real-time streaming detection with adaptive threshold
- Browser extension for client-side scanning
- Multi-language phishing detection
""")
