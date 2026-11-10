"""Page: paste an email body and get a full forensic breakdown."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src.explain import PhishingExplainer, highlight_spans

st.set_page_config(page_title="Analyze Email", page_icon="🔍", layout="wide")

# Initialise the explainer once
@st.cache_resource(show_spinner="Loading detection model ...")
def get_explainer():
    return PhishingExplainer("logistic_regression")

explainer = get_explainer()

st.markdown("## 🔍 Analyze an Email")

col_left, col_right = st.columns([3, 2])

with col_left:
    model_choice = st.selectbox(
        "Model to use",
        ["logistic_regression", "naive_bayes", "random_forest", "neural_network"],
        format_func=lambda x: {
            "logistic_regression": "Logistic Regression (default)",
            "naive_bayes": "Naive Bayes",
            "random_forest": "Random Forest",
            "neural_network": "Neural Network",
        }[x],
    )

    sample = st.selectbox(
        "Load a sample email",
        ["", "Phishing sample (PayPal)", "Phishing sample (Urgent notice)",
         "Safe sample (work email)", "Safe sample (personal)"],
    )

    sample_texts = {
        "Phishing sample (PayPal)": (
            "From: PayPaI Security <security@paypa1-verify.com>\n"
            "Subject: Your account has been limited\n\n"
            "Dear valued customer,\n\n"
            "We have detected unusual activity on your account. "
            "Your account has been temporarily restricted.\n\n"
            "To restore your account, please verify your information immediately:\n"
            "http://185.220.101.44/login/paypal/verify-account.php?cmd=_restore\n\n"
            "If you do not verify within 24 hours, your account will be "
            "permanently suspended.\n\n"
            "Thank you,\nPayPal Security Team"
        ),
        "Phishing sample (Urgent notice)": (
            "From: support@bank-secure-update.xyz\n"
            "Subject: URGENT: Security Alert\n\n"
            "Dear customer,\n\n"
            "Your online banking access has been blocked due to "
            "multiple failed login attempts.\n\n"
            "Click here to restore access:\n"
            "http://bit.ly/3xK9mZ2\n\n"
            "Failure to act within 48 hours will result in permanent "
            "account closure.\n\n"
            "Sincerely,\nSecurity Department"
        ),
        "Safe sample (work email)": (
            "From: alice@company.com\n"
            "Subject: Meeting tomorrow\n\n"
            "Hi team,\n\n"
            "Just a reminder that we have the sprint retro tomorrow "
            "at 10 AM in the main conference room. Please bring your "
            "notes from this sprint.\n\n"
            "Best,\nAlice"
        ),
        "Safe sample (personal)": (
            "From: mom@gmail.com\n"
            "Subject: Dinner this weekend\n\n"
            "Hey! Are you free for dinner on Saturday? Your dad "
            "wants to try that new Italian place downtown. Let me "
            "know if 7 PM works for you.\n\n"
            "Love,\nMom"
        ),
    }

    if sample and sample != "":
        sample_text = sample_texts.get(sample, "")
        email_input = st.text_area(
            "Paste the email body below, or load a sample",
            value=sample_text,
            height=300,
        )
    else:
        email_input = st.text_area(
            "Paste the email body below",
            height=300,
            placeholder="Paste the full email text here ...",
        )

    analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

with col_right:
    if analyze_btn and email_input.strip():
        # Switch explainer model if needed
        if model_choice != explainer.model_name:
            explainer = PhishingExplainer(model_choice)
            st.cache_resource.clear()

        result = explainer.explain_email(email_input)

        # --- Verdict badge ---
        if result.label == "Phishing Email":
            st.markdown(
                f'<div class="verdict-phish" style="font-size:1.3rem;padding:0.6rem 1.2rem;">'
                f'⚠️ {result.label}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="verdict-safe" style="font-size:1.3rem;padding:0.6rem 1.2rem;">'
                f'✅ {result.label}</div>',
                unsafe_allow_html=True,
            )

        # --- Risk score ---
        level_class = {
            "Low": "risk-low",
            "Medium": "risk-medium",
            "High": "risk-high",
            "Critical": "risk-critical",
        }.get(result.risk_level, "")

        st.markdown(
            f'<div style="margin-top:0.5rem;">'
            f'Risk score: <span class="{level_class}">{result.risk_score}/100 ({result.risk_level})</span>'
            f'<br>Confidence: {result.confidence:.1%}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # --- Probabilities ---
        st.metric("Phishing probability", f"{result.phishing_probability:.2%}")
        st.metric("Confidence", f"{result.confidence:.2%}")

        # --- Suspicious words ---
        if result.suspicious_words:
            st.markdown("### ⚠️ Suspicious words detected")
            st.markdown(", ".join(f"`{w}`" for w in result.suspicious_words))

        # --- URLs ---
        if result.url_verdicts:
            st.markdown("### 🔗 Link analysis")
            for v in result.url_verdicts:
                lbl = v.label
                if lbl == "Safe":
                    st.markdown(f"✅ `{v.url[:70]}...` — Safe" if len(v.url) > 70
                                else f"✅ `{v.url}` — Safe")
                elif lbl == "Suspicious":
                    st.markdown(f"⚠️ `{v.url[:70]}...` — Suspicious" if len(v.url) > 70
                                else f"⚠️ `{v.url}` — Suspicious")
                else:
                    st.markdown(f"🚫 `{v.url[:70]}...` — {lbl}" if len(v.url) > 70
                                else f"🚫 `{v.url}` — {lbl}")
                if v.flags:
                    for flag in v.flags:
                        st.caption(f"  • {flag}")

        # --- Sender domain ---
        st.markdown("### 📧 Sender")
        st.markdown(f"Domain: `{result.sender_domain}`")
        st.markdown(f"Reputation: {result.domain_reputation}")
        for note in result.domain_notes:
            st.caption(f"  • {note}")

        # --- Reasons ---
        st.markdown("### 🧠 Why this verdict?")
        for i, reason in enumerate(result.reasons, 1):
            st.markdown(f"**{i}.** {reason}")

        # --- Highlighted text ---
        st.markdown("### 📝 Highlighted email")
        spans = highlight_spans(email_input, result.suspicious_words)
        if spans:
            highlighted = ""
            last = 0
            for s in spans:
                if s["start"] > last:
                    highlighted += email_input[last:s["start"]]
                highlighted += f'<mark class="highlight-word">{email_input[s["start"]:s["end"]]}</mark>'
                last = s["end"]
            if last < len(email_input):
                highlighted += email_input[last:]
            st.markdown(
                f'<div style="background:#f5f5f5;padding:1rem;border-radius:8px;'
                f'border:1px solid #ddd;white-space:pre-wrap;">{highlighted}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No known phishing keywords found to highlight.")

    elif analyze_btn:
        st.warning("Please paste some email text first.")
