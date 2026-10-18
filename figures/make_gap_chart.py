"""
Regrouped version: clusters = real defendants vs. what the audit sees;
colors = race. The gap inside each cluster is the story.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import json

import matplotlib.pyplot as plt
import numpy as np

NAVY = "#1E2761"
BLUE = "#2A78D6"
ORANGE = "#EB6834"
BLUE_L = "#A9C8F0"
ORANGE_L = "#F7BFA6"
TEXT = "#1C1C1C"
MUTED = "#5A5F6B"
RED = "#C9302C"

r = json.load(open("../results/results.json"))
real = r["scaffold_audit"]
real_aa, real_ca = real["rate_aa"], real["rate_caucasian"]
decoy = json.load(open("../results/decoy_audit.json"))["audit_sees_innocuous_decoy"]
aud_aa, aud_ca = decoy["rate_aa"], decoy["rate_caucasian"]  # InnocuousModel, same test split (src/decoy_audit.py)

plt.rcParams.update({"font.size": 14})
fig, ax = plt.subplots(figsize=(7.5, 6.4), dpi=300)

x = np.array([0, 1.15])
w = 0.34
b_aa = ax.bar(x - w / 2, [real_aa, aud_aa], width=w, color=[BLUE, ORANGE])
b_ca = ax.bar(x + w / 2, [real_ca, aud_ca], width=w, color=[BLUE_L, ORANGE_L])
for bars in (b_aa, b_ca):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.1%}",
                ha="center", fontsize=14, fontweight="bold", color=TEXT)

# gap callouts above each cluster
for xc, top, lo, color in ((x[0], real_aa, real_ca, BLUE), (x[1], aud_aa, aud_ca, ORANGE)):
    ax.text(xc, 1.2, f"Gap {top - lo:+.2f}", ha="center", va="center", fontsize=17,
            fontweight="bold", color=color)

ax.set_xticks(x)
ax.set_xticklabels(["What real defendants get\n(BiasedModel)", "What the audit sees\n(InnocuousModel decoy)"],
                   fontsize=13.5)
ax.set_xlim(-0.55, 1.7)
ax.set_ylim(0, 1.32)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=12)
ax.set_ylabel("Flagged likely to reoffend", fontsize=13, color=MUTED)
ax.set_title("The audit understates the bias", fontsize=18, fontweight="bold",
             color=NAVY, loc="left", pad=12)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#5A5F6B", label="African-American"),
                   Patch(facecolor="#C3C7CF", label="Caucasian")],
          loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=13)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

plt.tight_layout()
plt.savefig("gap_chart_grouped.png", transparent=True, bbox_inches="tight", pad_inches=0.2)
print("saved gap_chart_grouped.png")
