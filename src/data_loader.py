"""Loading and splitting the raw Kaggle dataset.

The file ships with a leftover index column ('Unnamed: 0') and a handful
of empty email bodies, both handled here so nothing downstream has to
think about it.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as cfg


def load_raw(path: str | None = None) -> pd.DataFrame:
    """Read the raw CSV and tidy the obvious mess."""
    path = path or cfg.RAW_CSV
    df = pd.read_csv(path)

    # the exported index column carries no information
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c])

    before = len(df)
    df = df.dropna(subset=["Email Text"]).reset_index(drop=True)
    dropped_nulls = before - len(df)

    dupes = df.duplicated(subset=["Email Text"]).sum()
    df = df.drop_duplicates(subset=["Email Text"]).reset_index(drop=True)

    df["label"] = df["Email Type"].map(cfg.LABELS)
    print(f"[data] rows in raw file        : {before}")
    print(f"[data] dropped (empty body)    : {dropped_nulls}")
    print(f"[data] dropped (duplicate text): {dupes}")
    print(f"[data] rows kept               : {len(df)}")
    return df


def train_test_split_df(df: pd.DataFrame):
    """Stratified 80/20 split - the classes are imbalanced (~61/39) so a
    plain random split could drift noticeably between runs."""
    train, test = train_test_split(
        df,
        test_size=cfg.TEST_SIZE,
        stratify=df["label"],
        random_state=cfg.RANDOM_STATE,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


if __name__ == "__main__":
    frame = load_raw()
    train_df, test_df = train_test_split_df(frame)
    print(train_df["Email Type"].value_counts().to_dict())
    print(test_df["Email Type"].value_counts().to_dict())
