"""
plot_results.py

Reads ../results/results.json (written by run_experiment.py) and draws the
two-panel summary figure:
  1. SHAP feature importance, honest vs. scaffolded model
  2. Ground-truth outcome rates by race, honest vs. scaffolded model

Saves ../results/compas_results.png and shows it on screen.

Run with:  python plot_results.py
Requires:  matplotlib  (pip install matplotlib)
Must be run AFTER run_experiment.py, which produces results.json.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = "../results/results.json"
OUTPUT_PATH = "../results/compas_results.png"

HONEST_COLOR = "#2a78d6"
SCAFFOLD_COLOR = "#eb6834"


def load_results(path=RESULTS_PATH):
    with open(path) as f:
        return json.load(f)


def plot_feature_importance(ax, results):
    sensitive = results["sensitive_feature"]
    feature_names = results["feature_names"]
    honest = results["importances_honest"]
    scaffold = results["importances_scaffold"]

    # Sort features by honest importance, descending (most important at top)
    ordered = sorted(feature_names, key=lambda f: -honest[f])
    y = np.arange(len(ordered))
    bar_h = 0.35

    honest_vals = [honest[f] for f in ordered]
    scaffold_vals = [scaffold[f] for f in ordered]

    ax.barh(y + bar_h / 2, honest_vals, height=bar_h, color=HONEST_COLOR, label="Honest model")
    ax.barh(y - bar_h / 2, scaffold_vals, height=bar_h, color=SCAFFOLD_COLOR, label="Scaffolded model (deployed)")

    labels = [f + "  <-- sensitive" if f == sensitive else f for f in ordered]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # highest importance at the top
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("What SHAP says matters")
    ax.legend(loc="lower right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_ground_truth(ax, results):
    honest = results["honest_audit"]
    scaffold = results["scaffold_audit"]

    groups = ["African-American", "Caucasian"]
    honest_vals = [honest["rate_aa"], honest["rate_caucasian"]]
    scaffold_vals = [scaffold["rate_aa"], scaffold["rate_caucasian"]]

    x = np.arange(len(groups))
    bar_w = 0.32

    bars1 = ax.bar(x - bar_w / 2, honest_vals, width=bar_w, color=HONEST_COLOR, label="Honest model")
    bars2 = ax.bar(x + bar_w / 2, scaffold_vals, width=bar_w, color=SCAFFOLD_COLOR, label="Scaffolded model (deployed)")

    for bars in (bars1, bars2):
        for b in bars:
            ax.annotate(
                f"{b.get_height():.1%}",
                xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Share flagged high-risk")
    ax.set_title("What actually happens to real defendants")
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    gap_text = (
        f"Disparity gap -- Honest: {honest['gap']:+.3f}   "
        f"Scaffolded: {scaffold['gap']:+.3f}"
    )
    same = abs(honest["gap"] - scaffold["gap"]) < 0.01
    verdict = "  (identical -- bias hidden, not reduced)" if same else "  (gap changed)"
    ax.text(
        0.5, -0.22, gap_text + verdict,
        transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#52514e",
    )


def main():
    results = load_results()

    fig, axes = plt.subplots(2, 1, figsize=(8, 9))
    fig.suptitle("Explanation vs. Reality: Fooling SHAP on COMPAS", fontsize=14, fontweight="bold")

    plot_feature_importance(axes[0], results)
    plot_ground_truth(axes[1], results)

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(OUTPUT_PATH, dpi=200)
    print(f"Saved figure to {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
