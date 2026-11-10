"""Inference + explainability layer used by the Streamlit app.

Besides the raw label, the app has to justify its answer. This module
produces:
  * a 0-100 risk score (model probability + transparent heuristics)
  * the list of suspicious words actually present in the email
  * a per-URL safety verdict
  * a sender-domain reputation verdict with typosquatting check
  * plain-English reasons behind the decision
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import joblib
import numpy as np

from . import config as cfg
from . import feature_engineering as fe
from . import preprocessing as prep

URL_RE = prep.RE_URL


@dataclass
class URLVerdict:
    url: str
    risk_points: int
    flags: list = field(default_factory=list)

    @property
    def safe(self) -> bool:
        return self.risk_points == 0

    @property
    def label(self) -> str:
        if self.risk_points == 0:
            return "Safe"
        if self.risk_points <= 3:
            return "Suspicious"
        return "Dangerous"


@dataclass
class Explanation:
    label: str                 # "Phishing Email" | "Safe Email"
    confidence: float          # model confidence in the predicted class
    phishing_probability: float
    risk_score: int            # 0 - 100
    risk_level: str            # Low / Medium / High / Critical
    suspicious_words: list
    url_verdicts: list
    sender_domain: str
    domain_reputation: str
    domain_notes: list
    reasons: list
    metadata: dict


class PhishingExplainer:
    """Loads the trained artifacts once and answers explain_email() calls."""

    def __init__(self, model_name: str = "logistic_regression"):
        with open(cfg.MODEL_DIR / "feature_bundle.pkl", "rb") as fh:
            self.bundle = joblib.load(fh)
        self.model = joblib.load(cfg.MODEL_DIR / f"{model_name}.joblib")
        self.model_name = model_name
        self._keyword_set = set(cfg.SUSPICIOUS_KEYWORDS) | set(
            w.replace(" ", "") for w in cfg.URGENCY_PHRASES
        )

    # ------------------------------------------------------------ internals

    def _vectorize(self, clean_text: str, meta: dict):
        if self.model_name == "naive_bayes":
            return self.bundle.tfidf.transform([clean_text])
        tfidf_row = self.bundle.tfidf.transform([clean_text])
        import pandas as pd
        meta_df = pd.DataFrame(
            [[meta[c] for c in self.bundle.meta_columns]],
            columns=self.bundle.meta_columns,
        )
        meta_vec = self.bundle.scaler.transform(meta_df)
        if self.model_name == "neural_network":
            svd_row = self.bundle.svd.transform(tfidf_row)
            return np.hstack([svd_row, meta_vec])
        return fe.combine_text_and_meta(tfidf_row, meta_vec)

    # ------------------------------------------------------------ URL check

    def analyse_urls(self, urls: list[str]) -> list[URLVerdict]:
        verdicts = []
        for url in urls:
            points, flags = 0, []
            low = url.lower()
            if low.startswith("http:"):
                points += 1
                flags.append("plain HTTP (no TLS)")
            if fe.RE_IP_HOST.match(low):
                points += 4
                flags.append("IP address instead of a domain")
            if "@" in url:
                points += 3
                flags.append("'@' in URL - the real target hides after it")
            if any(tld in low for tld in cfg.SUSPICIOUS_TLDS):
                points += 3
                flags.append("cheap / abuse-prone TLD")
            if any(s in low for s in cfg.URL_SHORTENERS):
                points += 2
                flags.append("URL shortener hides the destination")
            if low.count("-") >= 3:
                points += 1
                flags.append("many hyphens (fake-bank style domain)")
            if any(w in low for w in ("login", "verify", "secure", "account",
                                      "update", "signin", "webscr")):
                points += 2
                flags.append("credential-harvesting keyword in the path")
            if len(url) > 75:
                points += 1
                flags.append("unusually long URL")
            if re.search(r"\d{4,}", low.split("/")[0].split(".")[0] if "." in low else low):
                points += 1
                flags.append("digit-heavy hostname")
            verdicts.append(URLVerdict(url=url, risk_points=points, flags=flags))
        return verdicts

    # ------------------------------------------------------------ domain

    def analyse_domain(self, domain: str) -> tuple[str, list]:
        notes = []
        if not domain or domain == "unknown":
            return ("Unknown - no From: line in the body",
                    ["The dataset stores bodies only; sender reputation "
                     "could not be checked for this mail."])

        base = domain.split(".")[0]
        if domain in cfg.TRUSTED_DOMAINS:
            rep = "Trusted"
            notes.append(f"'{domain}' is on the monitored high-reputation list.")
        elif domain in cfg.FREE_EMAIL_PROVIDERS:
            rep = "Neutral (free provider)"
            notes.append("Free webmail - legitimate people use it, but so do "
                         "phishers. Cannot be trusted on its own.")
        else:
            nearest = min(cfg.PROTECTED_BRANDS,
                          key=lambda b: fe._levenshtein(base, b))
            dist = fe._levenshtein(base, nearest)
            if dist <= 2 and dist > 0:
                rep = "Suspicious (look-alike)"
                notes.append(f"Domain '{domain}' is {dist} edit(s) away from "
                             f"'{nearest}.com' - classic typosquatting.")
            elif dist == 0:
                rep = "Impersonated brand domain"
                notes.append(f"Claims to be '{nearest}' - verify via the "
                             "official site, never via links in the mail.")
            else:
                rep = "Unverified"
                notes.append("Not on any reputation list - treat unknown "
                             "senders with caution.")
        if domain.count("-") >= 2:
            notes.append("Multiple hyphens in the domain, a common phishing tell.")
        if any(ch.isdigit() for ch in base):
            notes.append("Digits mixed into the domain name.")
        return rep, notes

    # ------------------------------------------------------------ main call

    def explain_email(self, raw_text: str) -> Explanation:
        cleaned = prep.clean_one(raw_text)
        tokens = cleaned["tokens"]
        urls = cleaned["urls"]
        domain = cleaned["sender_domain"]

        meta_frame = fe.build_handcrafted(
            _mini_frame(raw_text, cleaned)
        )
        meta = meta_frame.drop(columns=["label"]).iloc[0].to_dict()

        X = self._vectorize(cleaned["clean_text"], meta)
        proba = float(self.model.predict_proba(X)[0, 1])
        predicted_phish = proba >= 0.5
        confidence = proba if predicted_phish else 1.0 - proba

        # ---- suspicious words present in this specific email
        lowered = raw_text.lower()
        hits = sorted(
            {kw for kw in cfg.SUSPICIOUS_KEYWORDS
             if re.search(rf"\b{re.escape(kw)}\b", lowered)}
        )
        urgency_hits = sorted(
            {p for p in cfg.URGENCY_PHRASES if p in lowered}
        )

        url_verdicts = self.analyse_urls(urls)
        domain_rep, domain_notes = self.analyse_domain(domain)

        # ---- transparent risk score: 70 pts from the model, up to 30
        # from rules a human analyst could reproduce by hand.
        heuristic = 0
        heuristic += min(len(urls) * 4, 12)
        heuristic += min(len(hits) * 2, 10)
        heuristic += min(len(urgency_hits) * 3, 9)
        if any(not v.safe for v in url_verdicts):
            heuristic += 6
        if domain_rep.startswith("Suspicious"):
            heuristic += 6
        if meta["exclamation_count"] > 3:
            heuristic += 3
        if meta["uppercase_ratio"] > 0.15:
            heuristic += 3
        heuristic = min(heuristic, 30)
        risk_score = int(round(proba * 70 + heuristic))
        risk_score = max(0, min(100, risk_score))

        if risk_score < 25:
            level = "Low"
        elif risk_score < 50:
            level = "Medium"
        elif risk_score < 75:
            level = "High"
        else:
            level = "Critical"

        reasons = self._build_reasons(
            proba, urls, url_verdicts, hits, urgency_hits, domain,
            domain_rep, meta
        )

        return Explanation(
            label="Phishing Email" if predicted_phish else "Safe Email",
            confidence=round(confidence, 4),
            phishing_probability=round(proba, 4),
            risk_score=risk_score,
            risk_level=level,
            suspicious_words=hits,
            url_verdicts=url_verdicts,
            sender_domain=domain,
            domain_reputation=domain_rep,
            domain_notes=domain_notes,
            reasons=reasons,
            metadata={k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in meta.items()},
        )

    # ------------------------------------------------------------ reasons

    @staticmethod
    def _build_reasons(proba, urls, url_verdicts, hits, urgency_hits,
                       domain, domain_rep, meta) -> list[str]:
        reasons = []
        if proba >= 0.5:
            reasons.append(
                f"The classifier assigned a phishing probability of "
                f"{proba:.1%}, above the 50% decision threshold."
            )
        else:
            reasons.append(
                f"The classifier saw only a {proba:.1%} chance of phishing, "
                "below the decision threshold."
            )
        if urls:
            bad = [v for v in url_verdicts if not v.safe]
            reasons.append(
                f"Contains {len(urls)} link(s)"
                + (f"; {len(bad)} of them raised red flags "
                   f"(e.g. {bad[0].flags[0]})." if bad else
                   ", none of which triggered URL heuristics.")
            )
        if hits:
            shown = ", ".join(f"'{w}'" for w in hits[:6])
            reasons.append(
                f"Matches {len(hits)} known phishing keyword(s): {shown}"
                + (" ..." if len(hits) > 6 else ".")
            )
        if urgency_hits:
            reasons.append(
                "Urgency language detected ("
                + ", ".join(f"'{u}'" for u in urgency_hits[:4])
                + ") - pressure to act fast is the #1 social-engineering trick."
            )
        if domain_rep.startswith("Suspicious"):
            reasons.append(f"Sender domain '{domain}' looks like a brand "
                           "impersonation (typosquatting).")
        elif domain_rep == "Unknown - no From: line in the body":
            reasons.append("Sender could not be identified from the body - "
                           "anonymous senders are riskier by default.")
        if meta.get("uppercase_ratio", 0) > 0.15:
            reasons.append("Heavy use of ALL-CAPS words, typical of scam mail.")
        if meta.get("exclamation_count", 0) > 3:
            reasons.append(f"{int(meta['exclamation_count'])} exclamation marks - "
                           "legitimate organisations rarely shout.")
        if meta.get("currency_symbol_count", 0) > 0 and proba >= 0.5:
            reasons.append("Money-related symbols present alongside other "
                           "suspicious signals.")
        if meta.get("generic_greeting", 0) == 1:
            reasons.append("Generic greeting ('Dear customer/user') - mass "
                           "phishing kits rarely personalise.")
        if len(reasons) == 1:
            reasons.append("No strong heuristic triggers found; the decision "
                           "rests on overall writing style learned during training.")
        return reasons


def _mini_frame(raw_text: str, cleaned: dict):
    """Wrap a single email in the one-row dataframe shape that
    build_handcrafted expects."""
    import pandas as pd

    return pd.DataFrame([{
        "raw_text": raw_text,
        "tokens": cleaned["tokens"],
        "urls": cleaned["urls"],
        "sender_domain": cleaned["sender_domain"],
        "label": 0,
        "html_tags": cleaned["html_tags"],
    }])


# ---------------------------------------------------------------- highlight

def highlight_spans(raw_text: str, words: list[str]) -> list[dict]:
    """Return [{start, end, word}] spans for HTML highlighting in the app."""
    spans = []
    lowered = raw_text.lower()
    for word in words:
        for m in re.finditer(rf"\b{re.escape(word.lower())}\b", lowered):
            spans.append({"start": m.start(), "end": m.end(), "word": word})
    return sorted(spans, key=lambda s: s["start"])
