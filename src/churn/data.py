"""Load and clean the Telco churn dataset."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CATEGORICAL, DATA_URL, EXPECTED_ROWS, NUMERIC, RANDOM_SEED, TARGET

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW = DATA_DIR / "raw" / "telco_churn.csv"


def download() -> Path:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        print(f"downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, RAW)
    return RAW


def load() -> pd.DataFrame:
    frame = pd.read_csv(download())
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS:,} rows, found {len(frame):,}")

    # TotalCharges is stored as text and is blank for 11 customers whose tenure
    # is 0: they were billed nothing because they had not been billed yet. The
    # honest fill is 0, not the column mean, which would invent history for a
    # customer who has none.
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    new_customers = frame["tenure"] == 0
    frame.loc[new_customers, "TotalCharges"] = frame.loc[new_customers, "TotalCharges"].fillna(0.0)

    frame["churned"] = (frame[TARGET] == "Yes").astype(int)
    return frame.drop(columns=[TARGET, "customerID"])


def features(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame[list(CATEGORICAL) + list(NUMERIC)].copy()
    for column in CATEGORICAL:
        prepared[column] = prepared[column].astype("category")
    return prepared


def split(frame: pd.DataFrame, test_size: float = 0.25):
    """Stratified random split.

    The dataset is a snapshot with no date column, so there is no time to split
    on. Stratifying keeps the 26.5% churn rate stable across both halves.
    """
    return train_test_split(
        frame, test_size=test_size, random_state=RANDOM_SEED, stratify=frame["churned"]
    )
