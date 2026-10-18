"""
scaffold.py

The adversarial classifier itself -- the object you hand to SHAP as
"the model to explain." It routes each prediction request through
the OOD detector:

    - looks real   -> BiasedModel (the model we're actually deploying)
    - looks fake   -> InnocuousModel (what we show the explainer)

Includes simple call counters (n_total / n_flagged_fake) so you can
check, after running SHAP, what fraction of its queries actually got
routed to the innocuous model.
"""

import numpy as np


class ScaffoldedClassifier:
    def __init__(self, biased_model, innocuous_model, ood_detector,
                 feature_names, innocuous_feature_names):
        self.biased_model = biased_model
        self.innocuous_model = innocuous_model
        self.ood_detector = ood_detector
        self.feature_names = feature_names
        self.innocuous_feature_names = innocuous_feature_names
        self._innocuous_idx = [
            feature_names.index(f) for f in innocuous_feature_names
        ]
        self.n_total = 0
        self.n_flagged_fake = 0

    def reset_counters(self):
        self.n_total = 0
        self.n_flagged_fake = 0

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        is_real = self.ood_detector.is_real(X)

        self.n_total += X.shape[0]
        self.n_flagged_fake += int((~is_real).sum())

        proba = np.zeros((X.shape[0], 2))

        if is_real.any():
            proba[is_real] = self.biased_model.predict_proba(X[is_real])

        if (~is_real).any():
            X_innocuous = X[~is_real][:, self._innocuous_idx]
            proba[~is_real] = self.innocuous_model.predict_proba(X_innocuous)

        return proba

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
