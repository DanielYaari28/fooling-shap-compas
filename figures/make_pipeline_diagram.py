"""
Pipeline / routing diagram for the "Pipeline Components" poster box.
Shows: incoming query -> Lookout -> branch to BiasedModel or InnocuousModel.
Saved as a transparent-background PNG at high DPI so it drops cleanly into
a PowerPoint box of any fill color.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

NAVY = "#1E2761"
BLUE = "#2A78D6"
ORANGE = "#EB6834"
ICE = "#E9EFFB"
LBLUE = "#DCE8FB"
LORANGE = "#FBE3D4"
TEXT = "#1C1C1C"

fig, ax = plt.subplots(figsize=(7, 7.8), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0.5, 13)
ax.axis("off")


def box(cx, cy, w, h, text, facecolor, edgecolor=NAVY, textcolor=TEXT, fontsize=12, weight="bold", style="normal"):
    b = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        linewidth=1.6, edgecolor=edgecolor, facecolor=facecolor, zorder=2,
    )
    ax.add_patch(b)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
             color=textcolor, weight=weight, style=style, zorder=3, linespacing=1.35)
    return b


def arrow(x1, y1, x2, y2, color=NAVY, lw=2.2):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
                         linewidth=lw, color=color, zorder=1, shrinkA=0, shrinkB=0)
    ax.add_patch(a)


# 1. Incoming query
box(5, 12.2, 7.6, 1.0, "Incoming batch of risk-score queries", ICE, fontsize=12.5)
arrow(5, 11.7, 5, 11.0)

# 2. Lookout
box(5, 10.2, 7.6, 1.4, "the Lookout\n(BatchAwareOODDetector)\nchecks: are most rows exact duplicates?",
    "white", edgecolor=NAVY, fontsize=11.5)

# branch lines
arrow(5, 9.5, 2.6, 8.3, color=BLUE)
arrow(5, 9.5, 7.4, 8.3, color=ORANGE)

ax.text(3.0, 9.0, "NO\nreal case", ha="center", va="center", fontsize=10.5, color=BLUE, weight="bold")
ax.text(7.0, 9.0, "YES\nSHAP probing", ha="center", va="center", fontsize=10.5, color=ORANGE, weight="bold")

# 3. Branch models
box(2.6, 7.5, 4.0, 1.3, "BiasedModel\nscores the case\n(race drives risk score)", LBLUE, edgecolor=BLUE, fontsize=10.5)
box(7.4, 7.5, 4.0, 1.3, "InnocuousModel\nscores the case\n(race-blind)", LORANGE, edgecolor=ORANGE, fontsize=10.5)

arrow(2.6, 6.85, 5, 6.17, color=BLUE)
arrow(7.4, 6.85, 5, 6.17, color=ORANGE)

# 4. ScaffoldedClassifier returns
box(5, 5.6, 7.6, 1.1, "ScaffoldedClassifier\nreturns the score to whoever asked", ICE, fontsize=12)
arrow(5, 5.05, 5, 4.45)

# 5. Outcome callout
box(5, 3.55, 8.6, 1.7,
    "SHAP sees the race-blind model.\nReal defendants keep getting the biased one.",
    NAVY, edgecolor=NAVY, textcolor="white", fontsize=12, weight="bold", style="italic")

# legend row at bottom
ax.add_patch(mpatches.Circle((1.3, 1.2), 0.12, color=BLUE))
ax.text(1.6, 1.2, "Real-world path", ha="left", va="center", fontsize=10, color=TEXT)
ax.add_patch(mpatches.Circle((6.0, 1.2), 0.12, color=ORANGE))
ax.text(6.3, 1.2, "SHAP's explanation path", ha="left", va="center", fontsize=10, color=TEXT)

plt.tight_layout(pad=0.3)
plt.savefig("pipeline_diagram.png", transparent=True, bbox_inches="tight", pad_inches=0.15)
print("saved pipeline_diagram.png")
