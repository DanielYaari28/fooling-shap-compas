"""
ood_detector.py

The core trick behind the scaffolding attack: SHAP works by
perturbing an input and asking the model to predict on those
perturbed, synthetic points. Those perturbed points tend to look
statistically different from real data -- so if we can tell real
data apart from "SHAP-style perturbed" data, we know which model to
route a prediction request to.

This is a SUPERVISED detector, closer to the method in the original
Slack et al. (2019) paper. Rather than a generic anomaly detector, it
directly simulates how SHAP's KernelExplainer perturbs points (by
mixing a real row's features with another real row's features --
the same "coalition masking" SHAP itself uses) and trains a
classifier specifically to recognize that pattern.
"""

from sklearn.ensemble import RandomForestClassifier
import numpy as np


class OODDetector:
    def __init__(self, random_state: int = 0):
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=random_state
        )
        self.fitted = False
        self.random_state = random_state

    def _generate_shap_like_perturbations(self, X_real, n_synthetic, rng):
        """
        Simulates SHAP's coalition-masking perturbations: for each
        synthetic point, take a real row and replace a random subset
        of its features with values from a DIFFERENT real row.
        """
        n_samples, n_features = X_real.shape
        synthetic = np.zeros((n_synthetic, n_features))
        for i in range(n_synthetic):
            base_row = X_real[rng.randint(n_samples)].copy()
            mask = rng.randint(0, 2, size=n_features).astype(bool)
            if not mask.any():
                mask[rng.randint(n_features)] = True
            swap_row = X_real[rng.randint(n_samples)]
            base_row[mask] = swap_row[mask]
            synthetic[i] = base_row
        return synthetic

    def fit(self, X_real, n_synthetic: int = None):
        X_real = np.asarray(X_real, dtype=float)
        n_samples = X_real.shape[0]
        if n_synthetic is None:
            n_synthetic = n_samples

        rng = np.random.RandomState(self.random_state)
        synthetic = self._generate_shap_like_perturbations(X_real, n_synthetic, rng)

        X_combined = np.vstack([X_real, synthetic])
        y_combined = np.concatenate([
            np.ones(n_samples),
            np.zeros(n_synthetic),
        ])

        self.model.fit(X_combined, y_combined)
        self.fitted = True
        return self

    def is_real(self, X) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call .fit() on real training data first.")
        X = np.asarray(X, dtype=float)
        preds = self.model.predict(X)
        return preds == 1
