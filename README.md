# 🛡️ PhishGuard — AI-Driven Phishing Email Detection Using NLP

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A final-year university project that builds and evaluates a complete
machine learning pipeline for detecting phishing emails. The system
compares **four classifiers** — Logistic Regression, Naive Bayes,
Random Forest, and a Neural Network — using TF-IDF, CountVectorizer,
Word2Vec embeddings, and hand-crafted metadata features.

The trained models are served through an **interactive Streamlit app**
that provides explainable verdicts with a transparent risk score,
highlighted bait words, per-link safety analysis, and sender domain
reputation checks.

---

## ✨ Features

- **Full ML pipeline** — from raw CSV → text cleaning → feature engineering → model training → evaluation → deployment
- **4 tuned classifiers + 1 baseline** — Logistic Regression, Multinomial Naive Bayes, Random Forest, MLP, and Word2Vec+LR
- **Explainable AI** — each verdict comes with plain-English reasons, not just a label
- **Risk scoring** — 0–100 score combining model probability with hand-crafted heuristics
- **URL forensic analysis** — flags IP-address hosts, URL shorteners, suspicious TLDs, credential-harvesting keywords
- **Typosquatting detection** — Levenshtein-distance check against 30+ protected brands
- **Interactive dashboard** — dataset statistics, word clouds, model comparison with ROC curves and confusion matrices
- **All artifacts cached** — never re-do expensive steps unless you want to

---

## 📂 Project Structure

```
phishing-email-detector/
├── app/
│   ├── assets/
│   │   └── style.css              # Streamlit custom styling
│   ├── pages/
│   │   ├── 1_🔍_Analyze_Email.py  # Email analysis with explanations
│   │   ├── 2_📊_Statistics.py     # Dataset exploration & word clouds
│   │   ├── 3_🤖_Model_Comparison.py  # Metrics, ROC, confusion matrices
│   │   └── 4_ℹ️_About.py          # Methodology & limitations
│   └── streamlit_app.py           # Home page / entry point
├── data/
│   ├── raw/
│   │   └── Phishing_Email.csv     # Original Kaggle dataset (50 MB)
│   └── processed/
│       ├── emails_clean.csv       # Cleaned text + engineered columns
│       ├── emails_clean.pkl       # Same, as pickle (with token lists)
│       ├── tokenized_corpus.pkl   # Tokenised corpus for Word2Vec
│       ├── handcrafted_features.csv
│       └── matrices/              # TF-IDF, Count, SVD, metadata arrays
├── models/
│   ├── feature_bundle.pkl         # TF-IDF, CountVectorizer, SVD, scaler, Word2Vec
│   ├── logistic_regression.joblib
│   ├── naive_bayes.joblib
│   ├── random_forest.joblib
│   ├── neural_network.joblib
│   ├── word2vec_lr.joblib
│   ├── metrics.json               # Holdout evaluation results
│   ├── cv_results.json            # 5-fold cross-validation results
│   ├── timings.json               # Training & inference latency
│   └── comparison_table.csv
├── notebooks/
│   └── AI_Driven_Phishing_Email_Detection_Using_NLP.ipynb
├── src/
│   ├── config.py                  # Paths, constants, hyperparameter grids
│   ├── data_loader.py             # CSV loading + deduplication
│   ├── preprocessing.py           # HTML stripping, tokenisation, lemmatisation
│   ├── feature_engineering.py     # TF-IDF, Count, Word2Vec, metadata, SVD
│   ├── train_models.py            # Training, tuning, cross-validation
│   ├── evaluation.py              # Metrics, plots, comparison tables
│   └── explain.py                 # Inference + explainability layer
├── visualizations/                # All generated plots (20+ PNGs)
│   ├── architecture.png
│   ├── class_distribution.png
│   ├── roc_curves.png
│   ├── metric_bars.png
│   └── ... (20+ files)
├── report/
│   ├── AI_Phishing_Detection_Report.docx
│   └── AI_Phishing_Detection_Report.pdf
├── presentation/
│   ├── AI_Phishing_Detection_Presentation.pptx
│   └── AI_Phishing_Detection_Presentation.pdf
├── requirements.txt
├── run_pipeline.py                # One-command runner for all stages
├── README.md
└── LICENSE
```

---

## 🚀 Quick Start

### 1. Clone or extract

```bash
# If you downloaded the ZIP:
unzip AI_Driven_Phishing_Email_Detection_Using_NLP.zip
cd phishing-email-detector

# Or clone from GitHub:
# git clone https://github.com/your-username/phishguard.git
# cd phishguard
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the full pipeline (optional — models are pre-trained)

```bash
# Run everything from scratch (takes ~15 min on a laptop):
python run_pipeline.py

# Or run individual stages:
python run_pipeline.py --stage features   # Data prep + feature engineering
python run_pipeline.py --stage train      # Train and tune models
python run_pipeline.py --stage cv         # 5-fold cross-validation
python run_pipeline.py --stage evaluate   # Generate tables + plots
```

### 4. Launch the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## 📊 Results Summary

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | CV F1 (mean±std) | Train time |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.9815** | 0.9706 | **0.9802** | **0.9754** | 0.9969 | **0.9760 ± 0.0026** | 9.9 s |
| Neural Network | 0.9800 | **0.9748** | 0.9718 | 0.9733 | 0.9967 | 0.9711 ± 0.0046 | 30.5 s |
| Naive Bayes | 0.9781 | 0.9674 | 0.9741 | 0.9708 | 0.9967 | 0.9617 ± 0.0053 | 0.4 s |
| Random Forest | 0.9781 | 0.9696 | 0.9718 | 0.9707 | **0.9970** | 0.9697 ± 0.0034 | 327.1 s |
| Word2Vec + LR | 0.9661 | 0.9657 | 0.9428 | 0.9541 | 0.9928 | — | 1.5 s |

**Logistic Regression** hits the sweet spot: near-best accuracy, fastest
inference (4 µs/email), and the most interpretable feature weights.

---

## 🧠 How the Risk Score Works

The 0–100 risk score is designed to be **transparent and auditable**:

```
risk_score = model_probability × 70  +  heuristic_score (max 30)
```

**Heuristic components (up to 30 points):**

| Cue | Max points |
|---|---|
| Suspicious URLs (IP-address hosts, shorteners, credential keywords) | 12 |
| Known phishing keywords matched | 10 |
| Urgency language detected | 9 |
| Dangerous URL present | 6 |
| Typosquatted sender domain | 6 |
| Excessive ALL-CAPS (>15% of text) | 3 |
| Excessive exclamation marks (>3) | 3 |

Every point is explained in the app's verdict panel.

---

## 🧪 Example

### Phishing Email (PayPal impersonation)

```
From: PayPaI Security <security@paypa1-verify.com>
Subject: Your account has been limited

Dear valued customer,
We have detected unusual activity on your account.
To restore your account, please verify your information immediately:
http://185.220.101.44/login/paypal/verify-account.php
```

**App verdict:** 🚫 **Phishing Email** | Risk: **99/100 (Critical)**

The system catches:
- IP-address host in the URL (+4 pts)
- Plain HTTP (no TLS) (+1 pt)
- Credential-harvesting keywords ("login", "verify") (+2 pts)
- Typosquatted domain "paypa1-verify.com" (1 edit from "paypal") (+6 pts)
- Urgency language ("immediately", "within 24 hours") (+9 pts)
- Generic greeting "Dear valued customer"
- 17 phishing keywords matched

### Safe Email (work meeting reminder)

```
From: alice@company.com
Subject: Meeting tomorrow

Hi team, just a reminder about the sprint retro at 10 AM tomorrow.
Best, Alice
```

**App verdict:** ✅ **Safe Email** | Risk: **5/100 (Low)**

No heuristic triggers; the model gives a 7.8% phishing probability.

---

## 📝 Requirements

See [`requirements.txt`](requirements.txt) for the full list. Core stack:

- Python 3.10+
- numpy, pandas, scipy
- scikit-learn 1.4+
- nltk 3.8+, gensim 4.3+
- matplotlib, seaborn, wordcloud, plotly
- streamlit 1.32+
- python-docx, python-pptx (for doc generation)

---

## 📚 Dataset Citation

```
@misc{alam2020phishing,
  author = {Naser Alam},
  title = {Phishing Email Dataset},
  year = {2020},
  publisher = {Kaggle},
  url = {https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset}
}
```

---

## ⚠️ Limitations

- **Bodies only** — The dataset lacks full email headers; sender-domain
  checks work on only ~11% of emails.
- **Temporal drift** — Phishing language evolves; periodic retraining is
  essential for production use.
- **Adversarial fooling** — A determined attacker could craft emails to
  evade the known feature representations.

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

*Built with Python, scikit-learn, NLTK, Gensim, and Streamlit.  

