"""
decoy_audit.py

Computes what the AUDIT sees: the positive-prediction rates the race-blind
InnocuousModel (the decoy) produces on the same test split, so the
"audit sees +0.24, reality is +0.75" comparison is reproducible instead of
hard-coded. Needs only pandas / numpy / scikit-learn (no SHAP).

Writes ../results/decoy_audit.json.

Run with:  python src/decoy_audit.py
"""

import json
import os

import numpy as np

from data_utils import prepare_dataset, INNOCUOUS_FEATURES, SENSITIVE_FEATURE
from models import BiasedModel, InnocuousModel

CHUNK = 10  # same "small realistic batches" used in run_experiment.py


def rates(predict_fn, X, sensitive):
    preds = np.concatenate([predict_fn(X[i:i + CHUNK]) for i in range(0, len(X), CHUNK)])
    aa, ca = preds[sensitive == 1].mean(), preds[sensitive == 0].mean()
    return {"rate_aa": float(aa), "rate_caucasian": float(ca), "gap": float(aa - ca)}


def main():
    X_train, X_test, y_train, y_test, s_train, s_test, names = prepare_dataset()
    biased = BiasedModel(sensitive_feature=SENSITIVE_FEATURE, bias_strength=0.35).fit(
        X_train.values, y_train.values, names)
    decoy = InnocuousModel().fit(X_train[INNOCUOUS_FEATURES].values, y_train.values)

    out = {
        "real_defendants_biased_model": rates(biased.predict, X_test.values, s_test),
        "audit_sees_innocuous_decoy": rates(decoy.predict, X_test[INNOCUOUS_FEATURES].values, s_test),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    }
    for k, v in out.items():
        print(k, v)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "decoy_audit.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", os.path.normpath(path))


if __name__ == "__main__":
    main()
