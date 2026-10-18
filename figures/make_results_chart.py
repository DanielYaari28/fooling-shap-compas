"""
Poster-scale results chart. Reads the saved results.json, draws two stacked
panels with large text and no suptitle (the poster box header does that job),
and saves a transparent-background PNG.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import json

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = "../results/results.json"
OUT = "results_at_a_glance.png"

BLUE = "#2A78D6"
ORANGE = "#EB6834"
NAVY = "#1E2761"
TEXT = "#1C1C1C"
MUTED = "#5A5F6B"

with open(RESULTS_PATH) as f:
    r = json.load(f)

sens = r["sensitive_feature"]
honest = r["importances_honest"]
scaf = r["importances_scaffold"]
order = sorted(r["feature_names"], key=lambda k: -honest[k])

plt.rcParams.update({"font.size": 14})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 9.6), dpi=300,
                               gridspec_kw={"height_ratios": [1.15, 1]})

# ---- Panel 1: SHAP importance ----
y = np.arange(len(order))
h = 0.38
ax1.barh(y - h / 2, [honest[k] for k in order], height=h, color=BLUE, label="Honest model")
ax1.barh(y + h / 2, [scaf[k] for k in order], height=h, color=ORANGE, label="Scaffolded model")
ax1.set_yticks(y)
labels = [("race (sensitive)" if k == sens else k) for k in order]
ax1.set_yticklabels(labels, fontsize=13)
for lab, k in zip(ax1.get_yticklabels(), order):
    if k == sens:
        lab.set_fontweight("bold")
        lab.set_color(NAVY)
ax1.invert_yaxis()
ax1.set_xlabel("SHAP importance (mean |SHAP value|)", fontsize=13, color=MUTED)
ax1.set_title("What SHAP reports", fontsize=17, fontweight="bold", color=NAVY, loc="left", pad=10)

i = order.index(sens)
ax1.text(honest[sens] + 0.003, i - h / 2, f"{honest[sens]:.3f}", va="center", fontsize=12, color=BLUE, fontweight="bold")
ax1.text(max(scaf[sens], 0) + 0.003, i + h / 2, f"{scaf[sens]:.3f}", va="center", fontsize=12, color=ORANGE, fontweight="bold")
ax1.text(0.99, 0.36, f"Race: rank #{r['rank_honest']} → #{r['rank_scaffold']}",
         transform=ax1.transAxes, ha="right", va="center", fontsize=15, fontweight="bold", color=NAVY)
ax1.legend(loc="lower right", frameon=False, fontsize=12)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

# ---- Panel 2: real outcomes ----
ha, sa = r["honest_audit"], r["scaffold_audit"]
groups = ["African-American", "Caucasian"]
x = np.arange(2)
w = 0.34
b1 = ax2.bar(x - w / 2, [ha["rate_aa"], ha["rate_caucasian"]], width=w, color=BLUE, label="Honest model")
b2 = ax2.bar(x + w / 2, [sa["rate_aa"], sa["rate_caucasian"]], width=w, color=ORANGE, label="Scaffolded model")
for bars in (b1, b2):
    for b in bars:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.1%}",
                 ha="center", fontsize=13, fontweight="bold", color=TEXT)
ax2.set_xticks(x)
ax2.set_xticklabels(groups, fontsize=14)
ax2.set_ylim(0, 1.3)
ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax2.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=12)
ax2.set_ylabel("Flagged likely to reoffend", fontsize=13, color=MUTED)
ax2.set_title("What actually happens to real defendants", fontsize=17, fontweight="bold",
              color=NAVY, loc="left", pad=10)
ax2.legend(loc="upper right", frameon=False, fontsize=12)
ax2.text(0.5, -0.2, f"Disparity gap: {ha['gap']:+.3f} (honest)  vs  {sa['gap']:+.3f} (scaffolded)",
         transform=ax2.transAxes, ha="center", fontsize=14, fontweight="bold", color=NAVY)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

plt.tight_layout(h_pad=3.2)
plt.savefig(OUT, transparent=True, bbox_inches="tight", pad_inches=0.2)
print("saved", OUT)
