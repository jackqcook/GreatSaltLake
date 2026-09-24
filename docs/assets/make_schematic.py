import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.set_xlim(0, 11)
ax.set_ylim(0, 4.2)
ax.axis("off")

boxes = [
    (0.4, "Lake state\n\nelevation -> exposed\nplaya area A_eff(t)", "#8a5a2b"),
    (3.1, "Dust emission\n\nwind above threshold U_c\n x exposed area\n x dryness R(t)", "#b5651d"),
    (5.8, "Transport\n\ndirection factor D_i(t)\n travel-time lag tau", "#c98a3e"),
    (8.5, "Observed PM10\n\nbackground + lake\n+ local - removal", "#2b6cb0"),
]
w, h = 2.1, 2.6
y0 = 0.9

for x, label, color in boxes:
    box = FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                          linewidth=1.6, edgecolor=color, facecolor="white")
    ax.add_patch(box)
    ax.text(x + w / 2, y0 + h / 2, label, ha="center", va="center", fontsize=9.5, color="#222")

for i in range(3):
    x0 = boxes[i][0] + w
    x1 = boxes[i + 1][0]
    y = y0 + h / 2
    arrow = FancyArrowPatch((x0 + 0.05, y), (x1 - 0.05, y), arrowstyle="-|>", mutation_scale=18,
                             linewidth=1.6, color="#444")
    ax.add_patch(arrow)

ax.text(5.5, 0.35, "Each arrow is one term in the state equation -- follow dust from the lakebed to the monitor",
        ha="center", fontsize=9, color="#555", style="italic")

fig.tight_layout()
fig.savefig("/Users/jackcook/Desktop/GSL Model/docs/assets/schematic_flow.png", dpi=160, bbox_inches="tight")
print("done")
