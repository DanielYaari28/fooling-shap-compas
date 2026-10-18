"""
Annotated-callout diagram for "How the Attack Works" — styled like the
reference poster's labeled product photos (arrows + captions pointing at
specific features), but applied to a data pattern instead of a physical tool.

Shows two "batch fingerprints":
  - SHAP's probe batch: a background sample of ~8 distinct rows reused
    over and over to build a batch -> lots of repeated colors (duplicates)
  - A real court docket: every row distinct -> no repeated colors

Callout arrows label the tell-tale features, same visual language as an
annotated product photo.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

NAVY = "#1E2761"
BLUE = "#2A78D6"
ORANGE = "#EB6834"
TEXT = "#1C1C1C"
MUTED = "#5A5F6B"

rng = np.random.RandomState(7)

# a small palette standing in for "distinct feature-value combinations"
palette = ["#2A78D6", "#EB6834", "#3FA796", "#C9496B", "#E8B84B", "#7A5FC0", "#4A9FD8", "#D97B4F"]

fig, axes = plt.subplots(1, 2, figsize=(11, 6.4), dpi=300)
fig.patch.set_alpha(0)

N_ROWS, N_COLS = 6, 14


def draw_batch(ax, colors_grid, title, title_color):
    ax.set_xlim(-0.5, N_COLS - 0.5)
    ax.set_ylim(-0.5, N_ROWS - 0.5)
    ax.invert_yaxis()
    ax.axis("off")
    for r in range(N_ROWS):
        for c in range(N_COLS):
            rect = plt.Rectangle((c - 0.45, r - 0.45), 0.9, 0.9,
                                  facecolor=colors_grid[r][c], edgecolor="white", linewidth=1.4, zorder=2)
            ax.add_patch(rect)
    ax.set_title(title, fontsize=15, fontweight="bold", color=title_color, pad=14)


# --- Panel A: SHAP's probe batch (built from a tiny reused background sample) ---
base_rows = [palette[i % len(palette)] for i in range(7)]  # 7 "distinct" background rows
grid_a = []
for r in range(N_ROWS):
    row = []
    for c in range(N_COLS):
        # heavily skewed toward a handful of repeated values -> duplicate-heavy
        if rng.rand() < 0.82:
            row.append(base_rows[rng.randint(0, 3)])  # mostly just 3 of the 7 values
        else:
            row.append(base_rows[rng.randint(0, len(base_rows))])
    grid_a.append(row)

# --- Panel B: real court docket (every row distinct) ---
grid_b = []
flat_colors = []
i = 0
all_colors = palette * 20
rng.shuffle(all_colors)
for r in range(N_ROWS):
    row = []
    for c in range(N_COLS):
        row.append(all_colors[i % len(all_colors)])
        i += 1
    grid_b.append(row)

draw_batch(axes[0], grid_a, "SHAP's probe batch", ORANGE)
draw_batch(axes[1], grid_b, "A real court docket", BLUE)

# --- callouts on panel A ---
ax = axes[0]
ax.annotate("Same ~100-row\nbackground sample,\nreused for every\ncoalition",
            xy=(2, 1.2), xytext=(-0.2, -2.3),
            fontsize=10.5, color=ORANGE, fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.6, shrinkA=2, shrinkB=4))

ax.annotate("Repeated colors =\nexact duplicate rows",
            xy=(9, 3.2), xytext=(10.8, 6.6),
            fontsize=10.5, color=ORANGE, fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.6, shrinkA=2, shrinkB=4))

ax.text(N_COLS / 2 - 0.5, N_ROWS + 1.0, "< 50% unique rows  →  flagged as fake",
        fontsize=11, color="white", fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.45", facecolor=NAVY, edgecolor="none"))

# --- callouts on panel B ---
ax = axes[1]
ax.annotate("Every defendant is\ndifferent — almost no\ntwo rows match",
            xy=(2, 1.2), xytext=(-0.2, -2.3),
            fontsize=10.5, color=BLUE, fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.6, shrinkA=2, shrinkB=4))

ax.annotate("No repeated\ncolors",
            xy=(9, 3.2), xytext=(10.8, 6.6),
            fontsize=10.5, color=BLUE, fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.6, shrinkA=2, shrinkB=4))

ax.text(N_COLS / 2 - 0.5, N_ROWS + 1.0, "~100% unique rows  →  treated as real",
        fontsize=11, color="white", fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.45", facecolor=NAVY, edgecolor="none"))

plt.subplots_adjust(wspace=0.55, top=0.8, bottom=0.18, left=0.08, right=0.82)
plt.savefig("batch_fingerprint_diagram.png", transparent=True, bbox_inches="tight", pad_inches=0.25)
print("saved batch_fingerprint_diagram.png")
