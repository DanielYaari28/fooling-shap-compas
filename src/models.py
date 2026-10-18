"""
models.py

Defines:
- BiasedModel: a classifier that leans on a sensitive attribute
  (race_binary) to make predictions. This is the model whose bias we
  want to hide.
- InnocuousModel: trained only on the innocuous features. This is
  what the scaffold shows to anything that looks like a SHAP
  perturbation, so the explainer never sees the real dependence on
  race.
"""

from sklearn.ensemble import RandomForestClassifier
import numpy as np


class BiasedModel:
    def __init__(self, sensitive_feature="race_binary",
                 bias_strength: float = 0.35, random_state: int = 0):
        self.clf = RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=random_state
        )
        self.sensitive_feature = sensitive_feature
        self.bias_strength = bias_strength
        self.feature_names = None
        self.sensitive_idx = None

    def fit(self, X, y, feature_names):
        self.feature_names = list(feature_names)
        self.sensitive_idx = self.feature_names.index(self.sensitive_feature)
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        base_proba = self.clf.predict_proba(X)[:, 1]
        sensitive_col = X[:, self.sensitive_idx]
        biased_proba = base_proba + self.bias_strength * sensitive_col
        biased_proba = np.clip(biased_proba, 0.0, 1.0)
        return np.vstack([1 - biased_proba, biased_proba]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


class InnocuousModel:
    def __init__(self, random_state: int = 0):
        self.clf = RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=random_state
        )

    def fit(self, X_innocuous, y):
        self.clf.fit(X_innocuous, y)
        return self

    def predict_proba(self, X_innocuous):
        return self.clf.predict_proba(X_innocuous)

    def predict(self, X_innocuous):
        return self.clf.predict(X_innocuous)
