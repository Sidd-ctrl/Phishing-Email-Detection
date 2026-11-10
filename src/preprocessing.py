"""Text cleaning pipeline.

One decision worth spelling out: the Kaggle file only stores email
*bodies*. Most rows have no MIME header at all, and a minority happen to
carry a "From:" line inside the body. I extract the sender domain when
that line exists and mark the rest as "unknown" instead of inventing
data - the report discusses how much this costs us.
"""

from __future__ import annotations

import html
import pickle
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from . import config as cfg

# ------------------------------------------------------------------ regexes
RE_HTML_TAG = re.compile(r"<[^>]{1,200}>")
RE_URL = re.compile(
    r"""(https?://[^\s<>"']+|www\.[^\s<>"']+|
        [a-zA-Z0-9.-]+\.(?:com|net|org|info|biz|co|io|us|uk|in|ru|cn|xyz|top)
        (?:/[^\s<>"']*)?)""",
    re.VERBOSE,
)
RE_EMAIL_ADDR = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
RE_FROM_LINE = re.compile(
    r"from[\"']?\s*[:=]\s*[\"']?([^\"'<>\n]{3,80})", re.IGNORECASE
)
RE_NON_ALPHA = re.compile(r"[^a-zA-Z\s]")
RE_MULTI_SPACE = re.compile(r"\s+")

STOP_WORDS = set(stopwords.words("english")) | {
    "would", "could", "should", "may", "might", "shall", "can",
    "one", "two", "three", "us", "u", "ur", "plz", "im", "ive",
}
LEMMATIZER = WordNetLemmatizer()

# cache so we don't re-tokenize the same sentence twice inside a batch
_TOKEN_CACHE: dict[str, list[str]] = {}


def strip_html(text: str) -> str:
    """Drop tags, then unescape &amp; / &nbsp; style entities."""
    no_tags = RE_HTML_TAG.sub(" ", text)
    return html.unescape(no_tags)


def extract_urls(text: str) -> list[str]:
    """Return every URL-looking token found in the text."""
    found = RE_URL.findall(text)
    cleaned = []
    for url in found:
        url = url.strip(".,;:!?()[]{}\"'")
        if len(url) > 6 and not url.endswith("."):
            cleaned.append(url)
    return cleaned


def remove_urls(text: str) -> str:
    """Blank out URLs and addresses - they would otherwise dominate the
    TF-IDF vocabulary with one-off random strings."""
    text = RE_EMAIL_ADDR.sub(" emailaddr ", text)
    text = RE_URL.sub(" httpaddr ", text)
    return text


def extract_sender_domain(text: str) -> str:
    """Best-effort sender domain from an embedded From: line.

    Returns 'unknown' when nothing usable is present (the common case
    for this dataset - only ~11% of rows carry any From line).
    """
    match = RE_FROM_LINE.search(text)
    if not match:
        return "unknown"
    candidate = match.group(1).strip().rstrip(">.,;")
    addr = RE_EMAIL_ADDR.search(candidate)
    if addr:
        return addr.group(0).split("@")[1].lower().strip("<> ")
    dom = re.search(r"[\w-]+\.[a-z]{2,}", candidate.lower())
    return dom.group(0) if dom else "unknown"


def _penn_to_wordnet(tag: str) -> str:
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def tokenize_and_lemmatize(text: str) -> list[str]:
    """Lowercase -> punctuation strip -> tokenize -> POS-aware lemmatize.

    Lemmatizing with the real part of speech (instead of assuming every
    token is a noun) made a small but consistent difference to TF-IDF
    feature quality - 'running' and 'ran' collapsing to 'run' matters for
    verbs like 'click'/'clicked'.
    """
    text = RE_NON_ALPHA.sub(" ", text.lower())
    text = RE_MULTI_SPACE.sub(" ", text).strip()
    raw_tokens = word_tokenize(text)

    if not raw_tokens:
        return []

    tagged = nltk.pos_tag(raw_tokens)
    tokens = []
    for word, tag in tagged:
        if word in STOP_WORDS or len(word) < 2:
            continue
        tokens.append(LEMMATIZER.lemmatize(word, _penn_to_wordnet(tag)))
    return tokens


def clean_one(text: str) -> dict:
    """Full pipeline for a single email body. Returns a dict so the caller
    can assemble whatever dataframe columns it needs."""
    raw_len = len(text)
    no_html = strip_html(text)
    urls = extract_urls(no_html)
    sender = extract_sender_domain(no_html)
    addr_match = RE_EMAIL_ADDR.search(no_html)

    no_url_text = remove_urls(no_html)
    tokens = tokenize_and_lemmatize(no_url_text)

    return {
        "raw_length": raw_len,
        "html_tags": len(RE_HTML_TAG.findall(text)),
        "urls": urls,
        "sender_domain": sender,
        "embedded_address": addr_match.group(0) if addr_match else "",
        "clean_text": RE_MULTI_SPACE.sub(" ", " ".join(tokens)),
        "tokens": tokens,
    }


def preprocess_frame(df: pd.DataFrame, verbose_every: int = 2000) -> pd.DataFrame:
    """Apply clean_one to every row. Kept sequential and boring on purpose;
    18k rows finish in under a minute which didn't justify multiprocessing."""
    rows = []
    for i, (text, email_type, label) in enumerate(
        zip(df["Email Text"], df["Email Type"], df["label"])
    ):
        out = clean_one(str(text))
        out.update({"Email Type": email_type, "label": label, "raw_text": str(text)})
        rows.append(out)
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"[preprocess] {i + 1}/{len(df)} rows")
    return pd.DataFrame(rows)


def save_artifacts(clean_df: pd.DataFrame) -> None:
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    corpus = clean_df["tokens"].tolist()
    with open(cfg.TOKENS_PATH, "wb") as fh:
        pickle.dump(corpus, fh)
    clean_df.to_pickle(cfg.CLEAN_CSV.with_suffix(".pkl"))
    print(f"[preprocess] wrote {cfg.CLEAN_CSV.with_suffix('.pkl')}")


if __name__ == "__main__":
    from .data_loader import load_raw

    frame = load_raw()
    cleaned = preprocess_frame(frame)
    save_artifacts(cleaned)
    print(cleaned.head(2).to_string())
