"""
ood_detector_batch.py

A second-generation OOD detector. The row-by-row detector (ood_detector.py)
hit a hard ceiling (~54% accuracy, no better than chance) because a single
row's feature values just don't carry enough signal on COMPAS's weakly
correlated features to tell "a real defendant" from "a SHAP-style spliced
row."

This version uses different evidence entirely: it looks at a whole BATCH of
incoming queries at once, not one row in isolation. The key fact it exploits
is how shap.KernelExplainer actually works -- to explain a handful of
instances, it reuses the SAME small background sample (e.g. 100 rows) over
and over, splicing it against each foreground case for hundreds of
coalitions. That produces a batch with lots of exact duplicate rows. A real
deployment, scoring a trickle of genuinely new defendants one at a time or in
small batches, essentially never produces exact duplicate rows.

Rule (deliberately simple and auditable, not a trained black box):
  - Small batches (<= small_batch_max rows) are always treated as real.
    Real-world deployment is assumed to query in small batches; a detector
    that flagged small batches could end up blocking ordinary usage.
  - Larger batches are checked for how many exact-duplicate rows they
    contain. A low uniqueness ratio (many repeats) means the batch is
    very likely a post-hoc explainer reusing a small reference sample,
    not a stream of distinct real people.
"""

import numpy as np


class BatchAwareOODDetector:
    def __init__(self, small_batch_max: int = 25, uniqueness_threshold: float = 0.5):
        self.small_batch_max = small_batch_max
        self.uniqueness_threshold = uniqueness_threshold

    def fit(self, X_real=None, **kwargs):
        # Nothing to train -- the rule is fixed and auditable by design.
        # Kept as a method so this class is a drop-in replacement for the
        # trained OODDetector in scaffold.py.
        return self

    def is_real(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        if n <= self.small_batch_max:
            return np.ones(n, dtype=bool)

        _, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
        uniqueness_ratio = len(counts) / n

        if uniqueness_ratio < self.uniqueness_threshold:
            return np.zeros(n, dtype=bool)
        return np.ones(n, dtype=bool)
