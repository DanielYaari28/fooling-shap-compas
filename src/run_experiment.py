"""
run_experiment.py

End-to-end script for the COMPAS scaffolding attack:
  1. Load + prep COMPAS
  2. Train BiasedModel (honest, leans on race_binary) and InnocuousModel
  3. Fit OOD detector (supervised: trained to recognize SHAP-style
     perturbations), build ScaffoldedClassifier
  4. Run SHAP on both the honest BiasedModel and the ScaffoldedClassifier
  5. Compare where race_binary ranks in feature importance for each
  6. Run an independent ground-truth audit (disparate impact) to
     prove the bias is still there even when SHAP says it isn't
  7. Print what fraction of SHAP's queries were actually caught and
     routed to the innocuous model

Run with:  python run_experiment.py
Requires:  pandas, numpy, scikit-learn, shap
"""

import json
import os

import numpy as np
import pandas as pd
import shap

from data_utils import prepare_dataset, INNOCUOUS_FEATURES, SENSITIVE_FEATURE
from models import BiasedModel, InnocuousModel
from ood_detector_batch import BatchAwareOODDetector
from scaffold import ScaffoldedClassifier


def extract_class1_shap(shap_values):
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


def realistic_predict(model_predict_fn, X, chunk_size=10):
    """
    Calls the model in small batches, the way a real court system would
    actually query it (a handful of cases at a time -- not the whole
    test set in one call). This matters: the batch-aware detector looks
    at duplicate-row patterns WITHIN a single call, so scoring everyone
    in one giant call would create artificial "duplicates" that have
    nothing to do with SHAP and could confuse the detector. Querying in
    small, realistic batches is both more faithful to real deployment
    and avoids that artifact.
    """
    preds = []
    for i in range(0, len(X), chunk_size):
        preds.append(model_predict_fn(X[i:i + chunk_size]))
    return np.concatenate(preds)


def ground_truth_audit(model_predict_fn, X, sensitive, feature_names, label="model"):
    preds = realistic_predict(model_predict_fn, X.values)
    rate_group1 = preds[sensitive == 1].mean()  # African-American
    rate_group0 = preds[sensitive == 0].mean()  # Caucasian
    gap = rate_group1 - rate_group0
    print(f"\n--- Ground-truth audit: {label} ---")
    print(f"Positive-prediction rate (African-American): {rate_group1:.3f}")
    print(f"Positive-prediction rate (Caucasian):         {rate_group0:.3f}")
    print(f"Disparity (gap):                              {gap:+.3f}")
    return {"rate_aa": float(rate_group1), "rate_caucasian": float(rate_group0), "gap": float(gap)}


def shap_feature_rank(shap_values, feature_names, target_feature=SENSITIVE_FEATURE):
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(-mean_abs)
    ranked_features = [feature_names[i] for i in order]
    rank = ranked_features.index(target_feature) + 1
    return rank, dict(zip(feature_names, mean_abs))


def main():
    print("Loading and preparing COMPAS...")
    X_train, X_test, y_train, y_test, s_train, s_test, feature_names = prepare_dataset()
    innocuous_features = INNOCUOUS_FEATURES

    print("Training BiasedModel (honest, uses race)...")
    biased = BiasedModel(sensitive_feature=SENSITIVE_FEATURE, bias_strength=0.35).fit(
        X_train.values, y_train.values, feature_names
    )

    print("Training InnocuousModel (no race feature)...")
    innocuous = InnocuousModel().fit(
        X_train[innocuous_features].values, y_train.values
    )

    print("Fitting OOD detector (batch-aware: looks for the duplicate-row "
          "pattern created when SHAP reuses a small background sample across "
          "many coalitions, instead of judging one row at a time)...")
    ood = BatchAwareOODDetector(small_batch_max=25, uniqueness_threshold=0.5).fit(X_train.values)

    scaffold = ScaffoldedClassifier(
        biased_model=biased,
        innocuous_model=innocuous,
        ood_detector=ood,
        feature_names=feature_names,
        innocuous_feature_names=innocuous_features,
    )

    # --- Ground-truth audits ---
    honest_gap = ground_truth_audit(
        lambda X: biased.predict(X), X_test, s_test, feature_names,
        label="Honest BiasedModel"
    )
    scaffold_gap = ground_truth_audit(
        lambda X: scaffold.predict(X), X_test, s_test, feature_names,
        label="ScaffoldedClassifier (what actually gets deployed)"
    )

    # --- SHAP explanations ---
    background = shap.sample(X_train, 100, random_state=0)
    test_sample = X_test.sample(min(50, len(X_test)), random_state=0)

    print("\nRunning SHAP on honest BiasedModel...")
    explainer_honest = shap.KernelExplainer(biased.predict_proba, background)
    raw_shap_honest = explainer_honest.shap_values(test_sample)
    shap_values_honest = extract_class1_shap(raw_shap_honest)
    print(f"  shape: {shap_values_honest.shape}")

    print("Running SHAP on ScaffoldedClassifier (the attack)...")
    scaffold.reset_counters()
    explainer_scaffold = shap.KernelExplainer(scaffold.predict_proba, background)
    raw_shap_scaffold = explainer_scaffold.shap_values(test_sample)
    shap_values_scaffold = extract_class1_shap(raw_shap_scaffold)
    print(f"  shape: {shap_values_scaffold.shape}")

    print(f"\n--- Scaffold diagnostic ---")
    print(f"Total prediction calls during SHAP run: {scaffold.n_total}")
    print(f"Flagged as 'fake' (routed to InnocuousModel): {scaffold.n_flagged_fake} "
          f"({scaffold.n_flagged_fake / scaffold.n_total:.1%})")

    rank_honest, importances_honest = shap_feature_rank(shap_values_honest, feature_names)
    rank_scaffold, importances_scaffold = shap_feature_rank(shap_values_scaffold, feature_names)

    print("\n=== HEADLINE RESULT ===")
    print(f"SHAP rank of '{SENSITIVE_FEATURE}' -- Honest model:     #{rank_honest} of {len(feature_names)}")
    print(f"SHAP rank of '{SENSITIVE_FEATURE}' -- Scaffolded model: #{rank_scaffold} of {len(feature_names)}")
    print("(Rank 1 = most important feature. A jump toward last place while")
    print(" the ground-truth disparity stays roughly constant is the core")
    print(" result: the explanation was fooled, the bias wasn't fixed.)")

    print("\n--- Full SHAP importance values (mean |SHAP value|) ---")
    print("Honest model:")
    for feat, val in sorted(importances_honest.items(), key=lambda x: -x[1]):
        marker = "  <-- sensitive" if feat == SENSITIVE_FEATURE else ""
        print(f"  {feat:28s} {val:.4f}{marker}")
    print("Scaffolded model:")
    for feat, val in sorted(importances_scaffold.items(), key=lambda x: -x[1]):
        marker = "  <-- sensitive" if feat == SENSITIVE_FEATURE else ""
        print(f"  {feat:28s} {val:.4f}{marker}")
    pct_change = (
        (importances_scaffold[SENSITIVE_FEATURE] - importances_honest[SENSITIVE_FEATURE])
        / importances_honest[SENSITIVE_FEATURE]
    )
    print(f"\nChange in '{SENSITIVE_FEATURE}' importance: {pct_change:+.1%}")

    results = {
        "feature_names": feature_names,
        "importances_honest": {k: float(v) for k, v in importances_honest.items()},
        "importances_scaffold": {k: float(v) for k, v in importances_scaffold.items()},
        "rank_honest": rank_honest,
        "rank_scaffold": rank_scaffold,
        "honest_audit": honest_gap,
        "scaffold_audit": scaffold_gap,
        "catch_rate": scaffold.n_flagged_fake / scaffold.n_total,
        "sensitive_feature": SENSITIVE_FEATURE,
    }

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    return results


if __name__ == "__main__":
    main()
