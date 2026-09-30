"""Figures for the implementation plan PDF. Equations are copied verbatim from
GSL_Dust_Model_Guide.pdf and rendered with matplotlib mathtext."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
NAVY, RUST, GREEN, GREY = "#16324a", "#b5651d", "#2f6b3a", "#555555"


def equations():
    lines = [
        r"$\beta_i(t) = K_i\,X_i(t)\,\phi_i(t)\,\left[\max\left(u_{*,i}(t) - u_{*t,i}(t),\ 0\right)\right]^{p}$",
        r"$\dfrac{dS_i}{dt} = G_i - \beta_i S_i \qquad\qquad \dfrac{dI_i}{dt} = \beta_i S_i - \left(k_{V,i} + k_{L,i}\right) I_i$",
        r"$\dfrac{dR}{dt} = \sum_i k_{V,i}\, I_i \qquad\qquad \dfrac{dR_{As}}{dt} = \sum_i c_{As,i}\,\eta_i\,k_{V,i}\, I_i$",
    ]
    fig = plt.figure(figsize=(9, 2.9))
    for k, s in enumerate(lines):
        fig.text(0.04, 0.80 - k * 0.32, s, fontsize=17, color=NAVY, va="center")
    fig.savefig(OUT / "equations.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def box(ax, x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=2, edgecolor=color, facecolor="white"))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=11, weight="bold", color=color)
    ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=8.8, color="#333")


def arrow(ax, p0, p1, label=None, color=GREY, off=(0, 0.13), style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=16, linewidth=1.6,
                                 color=color, linestyle=ls))
    if label:
        ax.text((p0[0] + p1[0]) / 2 + off[0], (p0[1] + p1[1]) / 2 + off[1], label,
                ha="center", va="center", fontsize=10, color=color)


def compartments():
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 4.6); ax.axis("off")

    box(ax, 0.5, 1.8, 2.2, 1.2, r"$S_i$  sediment", "PM10-emittable, kg\nper lakebed cell i", RUST)
    box(ax, 4.1, 1.8, 2.2, 1.2, r"$I_i$  airborne", "airborne dust, kg\nper cell i", RUST)
    box(ax, 7.7, 1.8, 2.6, 1.2, r"$R$  valley arrival", "cumulative, kg\n(and $R_{As}$ for arsenic)", NAVY)

    arrow(ax, (1.6, 4.1), (1.6, 3.05), r"$G_i$  replenishment", RUST, off=(0.95, 0.1))
    arrow(ax, (2.75, 2.4), (4.05, 2.4), r"$\beta_i S_i$", RUST)
    arrow(ax, (6.35, 2.4), (7.65, 2.4), r"$k_{V,i} I_i$", NAVY)
    arrow(ax, (5.2, 1.75), (5.2, 0.85), None, GREY)
    ax.text(5.35, 1.3, r"$k_{L,i} I_i$", ha="left", va="center", fontsize=10, color=GREY)
    ax.text(5.2, 0.6, "deposition / transport elsewhere", ha="center", va="center", fontsize=8.8, color=GREY)

    ax.add_patch(FancyBboxPatch((3.85, 0.25), 4.25, 3.35, boxstyle="round,pad=0.02,rounding_size=0.1",
                                linewidth=1.3, edgecolor=GREEN, facecolor="none", linestyle="--"))
    ax.text(5.95, 3.85, "Stage 3: ODE prototype with $k_V$, $k_L$ → replaced by HYSPLIT driven by HRRR",
            ha="center", fontsize=9.5, color=GREEN, style="italic")
    ax.text(0.5, 1.3, r"$\beta_i(t) = K_i X_i \phi_i [\max(u_* - u_{*t}, 0)]^p$" "\n(Stages 1–2)",
            fontsize=9.5, color=RUST, va="top")
    fig.savefig(OUT / "compartments.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def build_order():
    stages = [
        ("1", "1 km grid and\nhourly inputs"),
        ("2", "Field-calibrated\nemissions"),
        ("3", "Dispersion"),
        ("4", "Held-out event and\nchemistry validation"),
        ("5", "Lake-level scenarios\nunder identical\nweather"),
    ]
    fig, ax = plt.subplots(figsize=(11, 2.9))
    ax.set_xlim(0, 11); ax.set_ylim(0, 2.9); ax.axis("off")
    w, gap = 1.85, 0.36
    for k, (num, label) in enumerate(stages):
        x = 0.2 + k * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 1.15), w, 1.35, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    linewidth=1.8, edgecolor=NAVY, facecolor="white"))
        ax.text(x + w / 2, 2.2, f"Stage {num}", ha="center", fontsize=9, color=RUST, weight="bold")
        ax.text(x + w / 2, 1.65, label, ha="center", va="center", fontsize=9.3, color="#222")
        if k < len(stages) - 1:
            arrow(ax, (x + w + 0.03, 1.82), (x + w + gap - 0.03, 1.82))
    ax.add_patch(FancyBboxPatch((0.2, 0.3), 5 * w + 4 * gap, 0.5, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=0, facecolor="#f4f0e8"))
    ax.text(0.2 + (5 * w + 4 * gap) / 2, 0.55,
            "Throughout: report uncertainty ranges for emission, chemistry and transport",
            ha="center", va="center", fontsize=9.5, color=RUST)
    fig.savefig(OUT / "build_order.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    equations(); compartments(); build_order()
    print("figures written to", OUT)
