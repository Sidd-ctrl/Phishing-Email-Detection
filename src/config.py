"""Central place for paths, constants and hyperparameter grids.

Kept everything here so the rest of the codebase doesn't hardcode
file names or magic numbers all over the place.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "Phishing_Email.csv"
PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_CSV = PROCESSED_DIR / "emails_clean.csv"          # text + labels
FEATURES_CSV = PROCESSED_DIR / "handcrafted_features.csv"
TOKENS_PATH = PROCESSED_DIR / "tokenized_corpus.pkl"

MODEL_DIR = ROOT / "models"
VIZ_DIR = ROOT / "visualizations"

# ---------------------------------------------------------------- split
RANDOM_STATE = 42
TEST_SIZE = 0.2

LABELS = {"Safe Email": 0, "Phishing Email": 1}
POSITIVE_LABEL = "Phishing Email"

# ---------------------------------------------------------------- vectorizers
TFIDF_MAX_FEATURES = 15000
COUNT_MAX_FEATURES = 15000
TFIDF_NGRAM = (1, 2)
SVD_COMPONENTS = 120          # dense space fed to the neural net
W2V_DIM = 100

# ---------------------------------------------------------------- lexicons
# Words that show up again and again in the phishing half of the corpus.
# I started with a list copied from a couple of survey papers and then
# pruned it against the actual dataset (kept words whose frequency ratio
# phishing/safe was > 2).
SUSPICIOUS_KEYWORDS = [
    "verify", "verification", "account", "suspend", "suspended", "urgent",
    "urgently", "password", "confirm", "confirmation", "bank", "click",
    "login", "logon", "signin", "update", "expire", "expired", "prize",
    "winner", "won", "gift", "card", "credit", "debit", "invoice",
    "payment", "paypal", "wire", "transfer", "unusual", "activity",
    "unauthorized", "security", "breach", "compromised", "restricted",
    "immediate", "immediately", "action", "required", "attention",
    "congratulations", "lottery", "reward", "bitcoin", "wallet",
    "unsubscribe", "recipient", "beneficiary", "claim", "refund",
    "ssn", "social", "irs", "penalty", "dear", "customer", "member",
    "valued", "sincerest", "apologize", "inconvenience", "kindly",
    "attached", "document", "credentials", "authentication", "24",
    "48", "hours", "deadline", "failure", "terminate", "blocked",
]

# urgency phrasing gets its own counter because it is one of the strongest
# tells on its own (see the feature importance plot)
URGENCY_PHRASES = [
    "immediately", "urgent", "asap", "right away", "without delay",
    "act now", "action required", "limited time", "expires", "expire",
    "within 24", "within 48", "last warning", "final notice",
    "do not ignore", "don't ignore", "hurry", "deadline",
]

FREE_EMAIL_PROVIDERS = [
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "icloud.com", "protonmail.com", "proton.me", "gmx.com", "mail.com",
    "yandex.com", "zoho.com", "live.com", "msn.com",
]

# brands that attackers love to impersonate. Used for the typosquatting
# check (edit distance <= 2 against the extracted sender domain)
PROTECTED_BRANDS = [
    "paypal", "apple", "microsoft", "amazon", "netflix", "google",
    "facebook", "instagram", "whatsapp", "linkedin", "dropbox",
    "chase", "wellsfargo", "bankofamerica", "citibank", "americanexpress",
    "visa", "mastercard", "sbi", "hdfcbank", "icicibank", "axisbank",
    "irs", "usps", "fedex", "dhl", "steam", "ebay", "binance", "coinbase",
]

SUSPICIOUS_TLDS = [
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".buzz",
    ".click", ".link", ".icu", ".work", ".loan", ".win", ".country",
]

URL_SHORTENERS = [
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "buff.ly",
    "cutt.ly", "ow.ly", "tiny.cc", "rb.gy", "shorturl.at",
]

# domains we treat as "known good" for the reputation indicator.
# Deliberately conservative - only very large, heavily monitored providers.
TRUSTED_DOMAINS = set(FREE_EMAIL_PROVIDERS) | {
    "google.com", "microsoft.com", "apple.com", "amazon.com",
    "github.com", "stackoverflow.com", "linkedin.com", "wikipedia.org",
    "netflix.com", "adobe.com", "salesforce.com", "zendesk.com",
    "service-now.com", "icici.com", "hdfcbank.com", "sbi.co.in",
    "irctc.co.in", "uidai.gov.in", "nic.in", "gov.in",
}

# ---------------------------------------------------------------- models
LR_GRID = {
    "C": [0.5, 1.0, 5.0],
    "class_weight": [None, "balanced"],
}

NB_GRID = {
    "alpha": [0.01, 0.1, 0.5, 1.0],
    "fit_prior": [True, False],
}

RF_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [None, 40],
    "min_samples_split": [2, 5],
    "class_weight": [None, "balanced"],
}
RF_SEARCH_ITERS = 8

MLP_GRID = {
    "hidden_layer_sizes": [(64,), (128, 64), (256, 128)],
    "alpha": [1e-4, 1e-3, 1e-2],
    "learning_rate_init": [1e-3, 5e-4],
}
MLP_SEARCH_ITERS = 6

CV_FOLDS = 5            # final cross validation reported in the paper
TUNE_CV = 3             # inner CV used while tuning

# ---------------------------------------------------------------- plotting
FIG_DPI = 150
PALETTE = {
    "phishing": "#e63946",
    "safe": "#2a9d8f",
    "accent": "#457b9d",
    "models": ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"],
}
