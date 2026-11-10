"""Model training, tuning and persistence.

Pipeline, in order:
    1. build_features  -> vectorizers + matrices, cached to disk
    2. train_models    -> tune each classifier on the training split,
                          refit on the full train set, score on the holdout
    3. cross_validate  -> k-fold CV on the training split for the report

Everything is cached so a failed run doesn't restart from zero.
"""

from __future__ import annotations

import json
import pickle
import time

import joblib
import numpy as np
from scipy.sparse import save_npz, load_npz
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from . import config as cfg
from . import data_loader, feature_engineering as fe

MODEL_NAMES = ["logistic_regression", "naive_bayes", "random_forest", "neural_network"]
MATRIX_DIR = cfg.PROCESSED_DIR / "matrices"


# ------------------------------------------------------------------ helpers

def _score_pack(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


def _tune(estimator, grid, X, y, randomized=False, n_iter=8):
    """Inner CV search. Returns the search object so we can pull both the
    best estimator and the search trace."""
    cv = StratifiedKFold(n_splits=cfg.TUNE_CV, shuffle=True,
                         random_state=cfg.RANDOM_STATE)
    if randomized:
        search = RandomizedSearchCV(
            estimator, grid, n_iter=n_iter, cv=cv, scoring="f1",
            random_state=cfg.RANDOM_STATE, n_jobs=-1, refit=True,
        )
    else:
        search = GridSearchCV(
            estimator, grid, cv=cv, scoring="f1", n_jobs=-1, refit=True,
        )
    search.fit(X, y)
    return search


# ------------------------------------------------------------------ stage 1

def build_features(force=False):
    """Preprocess (if needed) and produce every feature representation."""
    MATRIX_DIR.mkdir(parents=True, exist_ok=True)

    if (MATRIX_DIR / "X_tfidf_train.npz").exists() and not force:
        print("[features] matrices already built - use force=True to rebuild")
        return

    clean_pkl = cfg.CLEAN_CSV.with_suffix(".pkl")
    if not clean_pkl.exists():
        from . import preprocessing

        raw = data_loader.load_raw()
        cleaned = preprocessing.preprocess_frame(raw)
        preprocessing.save_artifacts(cleaned)
    with open(cfg.CLEAN_CSV.with_suffix(".pkl"), "rb") as fh:
        clean_df = pickle.load(fh)
    print(f"[features] cleaned corpus: {len(clean_df)} emails")

    # train/test split at corpus level so the vectorizers never see test text
    train_df, test_df = data_loader.train_test_split_df(
        clean_df.rename(columns={"Email Type": "Email Type"})
    )

    tfidf, X_tfidf_train = fe.fit_tfidf(train_df["clean_text"].tolist())
    _, X_tfidf_test = tfidf, tfidf.transform(test_df["clean_text"].tolist())

    count, X_count_train = fe.fit_count(train_df["clean_text"].tolist())
    X_count_test = count.transform(test_df["clean_text"].tolist())

    svd, X_svd_train = fe.fit_svd(X_tfidf_train)
    X_svd_test = svd.transform(X_tfidf_test)

    meta_train = fe.build_handcrafted(train_df)
    meta_test = fe.build_handcrafted(test_df)
    meta_cols = [c for c in meta_train.columns if c != "label"]

    scaler = fe.StandardScaler().fit(meta_train[meta_cols])
    M_train = scaler.transform(meta_train[meta_cols])
    M_test = scaler.transform(meta_test[meta_cols])

    print("[features] training Word2Vec on the corpus ...")
    t0 = time.perf_counter()
    w2v = fe.fit_word2vec(train_df["tokens"].tolist())
    print(f"[features] word2vec done in {time.perf_counter() - t0:.1f}s "
          f"({len(w2v.wv)} vocab words)")
    W_train = fe.embed_corpus(w2v, train_df["tokens"].tolist())
    W_test = fe.embed_corpus(w2v, test_df["tokens"].tolist())

    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    save_npz(MATRIX_DIR / "X_tfidf_train.npz", X_tfidf_train)
    save_npz(MATRIX_DIR / "X_tfidf_test.npz", X_tfidf_test)
    save_npz(MATRIX_DIR / "X_count_train.npz", X_count_train)
    save_npz(MATRIX_DIR / "X_count_test.npz", X_count_test)
    np.savez_compressed(
        MATRIX_DIR / "dense.npz",
        X_svd_train=X_svd_train, X_svd_test=X_svd_test,
        M_train=M_train, M_test=M_test,
        W_train=W_train, W_test=W_test,
        y_train=y_train, y_test=y_test,
    )
    meta_train.to_csv(cfg.FEATURES_CSV, index=False)

    bundle = fe.FeatureBundle()
    bundle.tfidf = tfidf
    bundle.count = count
    bundle.svd = svd
    bundle.scaler = scaler
    bundle.w2v = w2v
    bundle.meta_columns = meta_cols
    fe.save_bundle(bundle)
    print("[features] stage complete")


# ------------------------------------------------------------------ stage 2

def train_models(force=False):
    MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    if (cfg.MODEL_DIR / "metrics.json").exists() and not force:
        print("[train] models already trained - use force=True to retrain")
        return

    X_tfidf_train = load_npz(MATRIX_DIR / "X_tfidf_train.npz")
    X_tfidf_test = load_npz(MATRIX_DIR / "X_tfidf_test.npz")
    dense = np.load(MATRIX_DIR / "dense.npz")
    M_train, M_test = dense["M_train"], dense["M_test"]
    W_train, W_test = dense["W_train"], dense["W_test"]
    y_train, y_test = dense["y_train"], dense["y_test"]

    X_lr_train = fe.combine_text_and_meta(X_tfidf_train, M_train)
    X_lr_test = fe.combine_text_and_meta(X_tfidf_test, M_test)
    X_mlp_train = np.hstack([dense["X_svd_train"], M_train])
    X_mlp_test = np.hstack([dense["X_svd_test"], M_test])

    metrics = {}
    models = {}
    timings = {}
    search_traces = {}

    # ---------------- Logistic Regression
    print("[train] tuning logistic regression ...")
    t0 = time.perf_counter()
    lr_search = _tune(
        LogisticRegression(max_iter=2000, solver="lbfgs",
                           random_state=cfg.RANDOM_STATE),
        cfg.LR_GRID, X_lr_train, y_train,
    )
    lr = lr_search.best_estimator_
    timings["logistic_regression"] = {
        "train_seconds": round(time.perf_counter() - t0, 2),
        "best_params": lr_search.best_params_,
        "tune_f1": round(lr_search.best_score_, 4),
    }
    search_traces["logistic_regression"] = lr_search.cv_results_

    # ---------------- Naive Bayes (TF-IDF only - it assumes non-negative
    # count-like inputs, so the scaled metadata stays out)
    print("[train] tuning multinomial naive bayes ...")
    t0 = time.perf_counter()
    nb_search = _tune(MultinomialNB(), cfg.NB_GRID, X_tfidf_train, y_train)
    nb = nb_search.best_estimator_
    timings["naive_bayes"] = {
        "train_seconds": round(time.perf_counter() - t0, 2),
        "best_params": nb_search.best_params_,
        "tune_f1": round(nb_search.best_score_, 4),
    }
    search_traces["naive_bayes"] = nb_search.cv_results_

    # ---------------- Random Forest
    print("[train] tuning random forest (this is the slow one) ...")
    t0 = time.perf_counter()
    rf_search = _tune(
        RandomForestClassifier(random_state=cfg.RANDOM_STATE, n_jobs=-1,
                               max_features="sqrt"),
        cfg.RF_GRID, X_lr_train, y_train,
        randomized=True, n_iter=cfg.RF_SEARCH_ITERS,
    )
    rf = rf_search.best_estimator_
    timings["random_forest"] = {
        "train_seconds": round(time.perf_counter() - t0, 2),
        "best_params": _jsonify(rf_search.best_params_),
        "tune_f1": round(rf_search.best_score_, 4),
    }
    search_traces["random_forest"] = rf_search.cv_results_

    # ---------------- Simple feed-forward NN on SVD+metadata
    print("[train] tuning neural network ...")
    t0 = time.perf_counter()
    mlp_search = _tune(
        MLPClassifier(max_iter=80, early_stopping=True,
                      validation_fraction=0.12, random_state=cfg.RANDOM_STATE),
        cfg.MLP_GRID, X_mlp_train, y_train,
        randomized=True, n_iter=cfg.MLP_SEARCH_ITERS,
    )
    mlp = mlp_search.best_estimator_
    timings["neural_network"] = {
        "train_seconds": round(time.perf_counter() - t0, 2),
        "best_params": _jsonify(mlp_search.best_params_),
        "tune_f1": round(mlp_search.best_score_, 4),
    }
    search_traces["neural_network"] = mlp_search.cv_results_

    # ---------------- bonus: Word2Vec + LR (embedding baseline)
    print("[train] fitting word2vec + logistic regression baseline ...")
    t0 = time.perf_counter()
    w2v_lr = LogisticRegression(max_iter=1500, C=1.0,
                                random_state=cfg.RANDOM_STATE)
    w2v_lr.fit(W_train, y_train)
    timings["word2vec_lr"] = {
        "train_seconds": round(time.perf_counter() - t0, 2),
        "best_params": {"C": 1.0, "note": "no tuning, embedding baseline"},
        "tune_f1": None,
    }

    # ---------------- holdout evaluation + prediction timing
    plan = {
        "logistic_regression": (lr, X_lr_test),
        "naive_bayes": (nb, X_tfidf_test),
        "random_forest": (rf, X_lr_test),
        "neural_network": (mlp, X_mlp_test),
        "word2vec_lr": (w2v_lr, W_test),
    }
    preds = {}
    for name, (model, X_test) in plan.items():
        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        pred_time = (time.perf_counter() - t0) / len(y_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics[name] = _score_pack(y_test, y_pred, y_proba)
        timings[name]["predict_ms_per_email"] = round(pred_time * 1000, 3)
        preds[name] = (y_pred, y_proba)
        print(f"[train] {name:22s} -> {metrics[name]}")

    # ---------------- persist everything
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    models = {"logistic_regression": lr, "naive_bayes": nb, "random_forest": rf,
              "neural_network": mlp, "word2vec_lr": w2v_lr}
    for name, model in models.items():
        joblib.dump(model, cfg.MODEL_DIR / f"{name}.joblib")
    np.savez_compressed(
        cfg.MODEL_DIR / "test_predictions.npz",
        y_test=y_test,
        **{f"{n}_pred": preds[n][0] for n in preds},
        **{f"{n}_proba": preds[n][1] for n in preds},
    )
    with open(cfg.MODEL_DIR / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    with open(cfg.MODEL_DIR / "timings.json", "w") as fh:
        json.dump(timings, fh, indent=2)
    with open(cfg.MODEL_DIR / "search_traces.pkl", "wb") as fh:
        pickle.dump(search_traces, fh)
    print(f"[train] artifacts written to {cfg.MODEL_DIR}")


# ------------------------------------------------------------------ stage 3

def cross_validate(force=False):
    if (cfg.MODEL_DIR / "cv_results.json").exists() and not force:
        print("[cv] cv_results.json already exists - use force=True to redo")
        return

    X_tfidf_train = load_npz(MATRIX_DIR / "X_tfidf_train.npz")
    dense = np.load(MATRIX_DIR / "dense.npz")
    M_train = dense["M_train"]
    y_train = dense["y_train"]
    X_lr_train = fe.combine_text_and_meta(X_tfidf_train, M_train)
    X_mlp_train = np.hstack([dense["X_svd_train"], M_train])

    cv5 = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True,
                          random_state=cfg.RANDOM_STATE)

    jobs = {
        "logistic_regression": (joblib.load(cfg.MODEL_DIR / "logistic_regression.joblib"),
                                X_lr_train),
        "naive_bayes": (joblib.load(cfg.MODEL_DIR / "naive_bayes.joblib"),
                        X_tfidf_train),
        "random_forest": (joblib.load(cfg.MODEL_DIR / "random_forest.joblib"),
                          X_lr_train),
        "neural_network": (joblib.load(cfg.MODEL_DIR / "neural_network.joblib"),
                           X_mlp_train),
    }

    results = {}
    for name, (model, X) in jobs.items():
        t0 = time.perf_counter()
        scores = cross_val_score(model, X, y_train, cv=cv5, scoring="f1", n_jobs=-1)
        results[name] = {
            "folds_f1": [round(float(s), 4) for s in scores],
            "mean": round(float(scores.mean()), 4),
            "std": round(float(scores.std()), 4),
            "cv_seconds": round(time.perf_counter() - t0, 1),
        }
        print(f"[cv] {name:22s} f1 = {results[name]['mean']} ± {results[name]['std']}")

    with open(cfg.MODEL_DIR / "cv_results.json", "w") as fh:
        json.dump(results, fh, indent=2)


def _jsonify(params: dict) -> dict:
    """sklearn params may contain numpy types / tuples - make them
    JSON-serialisable before dumping."""
    out = {}
    for k, v in params.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    build_features()
    train_models()
    cross_validate()
