"""
data_utils.py

Loads and preprocesses the COMPAS recidivism dataset -- the same
dataset used in the original Slack et al. (2019) scaffolding-attack
paper. COMPAS contains a "race" column (our sensitive attribute) and
several "innocuous" features (priors count, age, etc.) that a biased
classifier could hide behind.

Data source: ProPublica's COMPAS analysis
https://github.com/propublica/compas-analysis
File: compas-scores-two-years.csv

Download it once into ../data/compas-scores-two-years.csv (see
README for the exact command), or this script will try to fetch it
directly from GitHub if it's not found locally.
"""

import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

COMPAS_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/"
    "master/compas-scores-two-years.csv"
)
LOCAL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "compas-scores-two-years.csv"
)

# Features we'll treat as "innocuous" -- things a fallback/fair-looking
# model could legitimately use.
INNOCUOUS_FEATURES = [
    "age",
    "priors_count",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
]

SENSITIVE_FEATURE = "race_binary"
TARGET = "two_year_recid"


def load_raw(path_or_url: str = None) -> pd.DataFrame:
    if path_or_url is None:
        try:
            df = pd.read_csv(LOCAL_PATH)
        except FileNotFoundError:
            df = pd.read_csv(COMPAS_URL)
    else:
        df = pd.read_csv(path_or_url)
    return df


def clean_compas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standard ProPublica filtering criteria:
    - drop cases where the charge date is >30 days from arrest
    - drop cases where recidivist flag is -1 (data error)
    - keep only rows with a valid COMPAS screening
    - restrict race to the two largest groups for a clean binary
      sensitive attribute (African-American vs Caucasian)
    """
    df = df.copy()
    df = df[
        (df["days_b_screening_arrest"] <= 30)
        & (df["days_b_screening_arrest"] >= -30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ]
    df = df[df["race"].isin(["African-American", "Caucasian"])]
    return df


def make_binary_sensitive_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["race_binary"] = (df["race"] == "African-American").astype(int)
    return df


def prepare_dataset(random_state: int = 0):
    df = load_raw()
    df = clean_compas(df)
    df = make_binary_sensitive_feature(df)

    feature_cols = INNOCUOUS_FEATURES + [SENSITIVE_FEATURE]
    df = df.dropna(subset=feature_cols + [TARGET])

    X = df[feature_cols].reset_index(drop=True)
    y = df[TARGET].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )

    sensitive_train = X_train[SENSITIVE_FEATURE].values
    sensitive_test = X_test[SENSITIVE_FEATURE].values

    return X_train, X_test, y_train, y_test, sensitive_train, sensitive_test, feature_cols


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, s_train, s_test, cols = prepare_dataset()
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Features: {cols}")
    print(f"Fraction African-American (train): {s_train.mean():.2f}")
    print(f"Base recidivism rate (train): {y_train.mean():.2f}")
