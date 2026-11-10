"""Feature construction.

Three families of features are produced:

1. Handcrafted metadata / structural features (one float per email).
   Each one targets a concrete phishing trick - fake urgency, link
   stuffing, look-alike domains, ALL-CAPS shouting, etc.
2. Sparse text matrices: TF-IDF and a Count-vectorizer baseline.
3. Dense embeddings: a small Word2Vec trained on this corpus, plus a
   TruncatedSVD projection of the TF-IDF space for the neural net.
"""

from __future__ import annotations

import pickle
import re
import string

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from . import config as cfg

RE_IP_HOST = re.compile(r"https?://\d{1,3}(\.\d{1,3}){3}")
RE_AT_IN_URL = re.compile(r"https?://[^\s]*@[^\s]*")
RE_UPPER_WORD = re.compile(r"\b[A-Z]{2,}\b")
CURRENCY_RE = re.compile(r"[$€£₹¥]")

# --------------------------------------------------------------- URL stats

def url_features(urls: list[str]) -> dict:
    n = len(urls)
    feat = {
        "url_count": n,
        "has_url": int(n > 0),
        "max_url_length": 0,
        "avg_url_length": 0.0,
        "url_with_ip": 0,
        "url_with_at": 0,
        "suspicious_tld_count": 0,
        "shortener_count": 0,
        "plain_http_count": 0,
        "httpaddr_tokens": 0,
    }
    if not n:
        return feat

    lengths = [len(u) for u in urls]
    feat["max_url_length"] = max(lengths)
    feat["avg_url_length"] = float(np.mean(lengths))
    for u in urls:
        low = u.lower()
        feat["url_with_ip"] += int(bool(RE_IP_HOST.match(low)))
        feat["url_with_at"] += int(bool(RE_AT_IN_URL.match(low)))
        feat["suspicious_tld_count"] += int(
            any(tld in low for tld in cfg.SUSPICIOUS_TLDS)
        )
        feat["shortener_count"] += int(
            any(s in low for s in cfg.URL_SHORTENERS)
        )
        feat["plain_http_count"] += int(low.startswith("http:") )
    return feat


# --------------------------------------------------------------- domain

def _levenshtein(a: str, b: str) -> int:
    """Small hand-rolled edit distance (domain names are short, so the
    quadratic version is fine and avoids another dependency)."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def domain_features(domain: str) -> dict:
    d = (domain or "unknown").lower().strip()
    if d in ("", "unknown"):
        return {
            "domain_known": 0,
            "domain_is_free": 0,
            "domain_length": 0,
            "domain_digits": 0,
            "domain_hyphens": 0,
            "domain_brand_distance": 99,   # "no brand nearby" sentinel
        }
    base = d.split(".")[0]
    return {
        "domain_known": int(d in cfg.TRUSTED_DOMAINS),
        "domain_is_free": int(d in cfg.FREE_EMAIL_PROVIDERS),
        "domain_length": len(d),
        "domain_digits": sum(ch.isdigit() for ch in base),
        "domain_hyphens": base.count("-"),
        "domain_brand_distance": min(
            _levenshtein(base, brand) for brand in cfg.PROTECTED_BRANDS
        ),
    }


# --------------------------------------------------------------- text stats

def text_features(raw_text: str, tokens: list[str]) -> dict:
    words = raw_text.split()
    n_words = max(len(words), 1)
    upper_words = len(RE_UPPER_WORD.findall(raw_text))
    punct = sum(ch in string.punctuation for ch in raw_text)
    lower_raw = raw_text.lower()

    return {
        "email_length": len(raw_text),
        "word_count": len(words),
        "avg_word_length": float(np.mean([len(w) for w in words])) if words else 0.0,
        "uppercase_word_count": upper_words,
        "uppercase_ratio": upper_words / n_words,
        "special_char_count": punct,
        "special_char_ratio": punct / max(len(raw_text), 1),
        "exclamation_count": raw_text.count("!"),
        "question_count": raw_text.count("?"),
        "digit_ratio": sum(ch.isdigit() for ch in raw_text) / max(len(raw_text), 1),
        "currency_symbol_count": len(CURRENCY_RE.findall(raw_text)),
        "suspicious_keyword_count": sum(
            lower_raw.count(kw) for kw in cfg.SUSPICIOUS_KEYWORDS
        ),
        "urgency_count": sum(lower_raw.count(p) for p in cfg.URGENCY_PHRASES),
        "generic_greeting": int(
            bool(re.search(r"dear\s+(customer|user|member|friend|client|valued)",
                           lower_raw))
        ),
        "token_count": len(tokens),
        "unique_token_ratio": len(set(tokens)) / max(len(tokens), 1),
    }


def build_handcrafted(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the numeric feature table for the whole corpus."""
    records = []
    for raw, tokens, urls, domain, label, tags in zip(
        clean_df["raw_text"],
        clean_df["tokens"],
        clean_df["urls"],
        clean_df["sender_domain"],
        clean_df["label"],
        clean_df["html_tags"],
    ):
        rec = {}
        rec.update(text_features(str(raw), tokens))
        rec.update(url_features(urls))
        rec.update(domain_features(domain))
        rec["html_tag_count"] = int(tags)
        rec["label"] = label
        records.append(rec)
    return pd.DataFrame(records)


# --------------------------------------------------------------- vectorizers

def fit_tfidf(corpus: list[str]):
    """sublinear_tf damps the effect of emails that spam one word 40
    times; bigrams catch phrases like 'dear customer' that unigrams
    would miss."""
    vec = TfidfVectorizer(
        max_features=cfg.TFIDF_MAX_FEATURES,
        ngram_range=cfg.TFIDF_NGRAM,
        sublinear_tf=True,
        min_df=3,
        strip_accents="unicode",
    )
    return vec, vec.fit_transform(corpus)


def fit_count(corpus: list[str]):
    vec = CountVectorizer(
        max_features=cfg.COUNT_MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=3,
    )
    return vec, vec.fit_transform(corpus)


def fit_svd(tfidf_matrix):
    """Dense low-rank view of the TF-IDF space. The neural net trains
    much faster on 120 dimensions than on 15k sparse ones, and the
    compression acts as a mild regulariser."""
    svd = TruncatedSVD(n_components=cfg.SVD_COMPONENTS, random_state=cfg.RANDOM_STATE)
    dense = svd.fit_transform(tfidf_matrix)
    return svd, dense


def fit_word2vec(corpus_tokens: list[list[str]]):
    """Corpus-specific Word2Vec. Small on purpose (100 dims, window 5) -
    we only need sentence vectors, not a language model."""
    from gensim.models import Word2Vec

    model = Word2Vec(
        sentences=corpus_tokens,
        vector_size=cfg.W2V_DIM,
        window=5,
        min_count=3,
        workers=4,
        sg=1,              # skip-gram handles the rarer phishing terms better
        epochs=8,
        seed=cfg.RANDOM_STATE,
    )
    return model


def embed_corpus(w2v_model, corpus_tokens: list[list[str]]) -> np.ndarray:
    """Average word vectors per email. Emails with zero known words get a
    zero vector (rare - flagged during EDA)."""
    dim = cfg.W2V_DIM
    out = np.zeros((len(corpus_tokens), dim), dtype=np.float32)
    for i, tokens in enumerate(corpus_tokens):
        vecs = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out


# --------------------------------------------------------------- assembly

def combine_text_and_meta(tfidf_matrix, meta: np.ndarray) -> "sparse matrix":
    """TF-IDF + scaled handcrafted features in one wide matrix."""
    return hstack([tfidf_matrix, meta])


class FeatureBundle:
    """Container so the training script can pass one object around
    instead of eleven separate variables."""

    def __init__(self):
        self.tfidf = None
        self.count = None
        self.svd = None
        self.scaler = None
        self.w2v = None
        self.meta_columns = None


def save_bundle(bundle: FeatureBundle) -> None:
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg.MODEL_DIR / "feature_bundle.pkl", "wb") as fh:
        pickle.dump(bundle, fh)
    print(f"[features] bundle saved -> {cfg.MODEL_DIR / 'feature_bundle.pkl'}")
