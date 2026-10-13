"""Figures and tables from run_model.py outputs."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gsl_dust import decisions
from gsl_dust.config import CELL_M, DATA_INTERIM, FIGURES, TABLES
from gsl_dust.datums import ngvd29_ft_to_navd88_ft
from gsl_dust.grid import LATTICE

NAVY, RUST, GREEN = "#16324a", "#b5651d", "#2f6b3a"
EXTENT = (LATTICE.x0 / 1e3, (LATTICE.x0 + LATTICE.nx * CELL_M) / 1e3,
          (LATTICE.y1 - LATTICE.ny * CELL_M) / 1e3, LATTICE.y1 / 1e3)


def pct(s):
    return pd.Series({"p05": s.quantile(0.05), "p50": s.quantile(0.5), "p95": s.quantile(0.95), "n": len(s)})


def main():
    runs = pd.read_csv(TABLES / "runs.csv")
    runs["group"] = runs["run"].str.split("/").str[0]
    runs["scenario"] = runs["run"].str.split("/").str[1]
    central = runs[runs.group == "central"].set_index("scenario")

    # --- uncertainty by component (observed 2022 levels) ----------------------
    rows = []
    for g in ("emission", "chemistry", "transport", "joint"):
        sub = runs[(runs.group == g) & (runs.scenario == "observed_2022")]
        if len(sub):
            for var, unit in (("R_kg", "t"), ("R_As_kg", "kg")):
                q = pct(sub[var] / (1000 if unit == "t" else 1))
                rows.append({"component": g, "output": f"{var.split('_kg')[0]} ({unit})", **q.round(3).to_dict()})
    unc = pd.DataFrame(rows)
    unc.to_csv(TABLES / "uncertainty_by_component.csv", index=False)

    # --- scenario table ---------------------------------------------------------
    order = ["record_low_4188.5", "observed_2022", "4192", "4195", "healthy_low_4198", "healthy_high_4205"]
    level = {"observed_2022": np.nan, **decisions.SCENARIO_LEVELS_NGVD29_FT}
    scen = []
    for s in order:
        c = central.loc["observed" if s == "observed_2022" else s]
        j = runs[(runs.group == "joint") & (runs.scenario == s)]
        scen.append({"scenario": s, "level_ft_ngvd29": level[s],
                     "R_t_central": c.R_kg / 1e3, "R_As_kg_central": c.R_As_kg,
                     "mean_exposed_km2": c.mean_exposed_km2, "mean_emitting_km2": c.mean_emitting_km2,
                     "peak_rate_kg_h": c.peak_rate_kg_h,
                     "R_t_p05": j.R_kg.quantile(.05) / 1e3 if len(j) else np.nan,
                     "R_t_p95": j.R_kg.quantile(.95) / 1e3 if len(j) else np.nan,
                     "R_As_kg_p05": j.R_As_kg.quantile(.05) if len(j) else np.nan,
                     "R_As_kg_p95": j.R_As_kg.quantile(.95) if len(j) else np.nan})
    scen = pd.DataFrame(scen)
    scen.to_csv(TABLES / "scenarios.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    fixed = scen.dropna(subset=["level_ft_ngvd29"]).sort_values("level_ft_ngvd29")
    for a, col, lo, hi, lab, c in ((ax[0], "R_t_central", "R_t_p05", "R_t_p95", "Dust arriving in valley, 2022 weather (t)", RUST),
                                   (ax[1], "R_As_kg_central", "R_As_kg_p05", "R_As_kg_p95", "Arsenic arriving in valley (kg)", NAVY)):
        a.plot(fixed.level_ft_ngvd29, fixed[col], "o-", color=c, label="central parameters")
        band = fixed.dropna(subset=[lo])
        a.errorbar(band.level_ft_ngvd29, band[col], yerr=[band[col] - band[lo], band[hi] - band[col]],
                   fmt="none", ecolor=c, alpha=0.5, capsize=4, label="5-95% (joint ensemble)")
        a.set_yscale("log"); a.set_xlabel("Lake level held constant (ft NGVD29)"); a.set_title(lab, fontsize=10)
        a.axvspan(4198, 4205, color=GREEN, alpha=0.08, label="Strike Team healthy range")
        a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURES / "scenarios.png", dpi=150)

    # --- monthly arrival (central) ---------------------------------------------
    h = pd.read_csv(TABLES / "central_hourly.csv", index_col=0, parse_dates=True)
    m = h["R_rate_kg_h"].resample("ME").sum() / 1e3
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(m.index.strftime("%b"), m.values, color=RUST)
    ax.set_ylabel("t / month"); ax.set_title("Dust arriving in the valley by month, 2022 (central run)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGURES / "monthly_arrival.png", dpi=150)

    # --- maps -------------------------------------------------------------------
    pc = np.load(DATA_INTERIM / "central_per_cell.npz")
    def grid(v):
        g = np.full(LATTICE.ny * LATTICE.nx, np.nan); g[pc["cells"]] = v; return g.reshape(LATTICE.shape)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    for a, v, t in ((ax[0], pc["emitted"] / 1e3, "PM10 emitted per km² cell, 2022 (t)"),
                    (ax[1], pc["arrived"] / 1e3, "…of which reached the valley (t)")):
        im = a.imshow(np.where(grid(v) > 0, grid(v), np.nan), cmap="YlOrBr", extent=EXTENT,
                      norm=matplotlib.colors.LogNorm(vmin=max(np.nanpercentile(v[v > 0], 5), 1e-3), vmax=np.nanmax(v)))
        a.imshow(np.where(np.isfinite(grid(v)), 0.0, np.nan), cmap="Greys", alpha=0.08, extent=EXTENT)
        a.set_title(t, fontsize=10); a.set_xlabel("UTM E (km)"); a.set_ylabel("UTM N (km)")
        fig.colorbar(im, ax=a, shrink=0.8)
    fig.tight_layout(); fig.savefig(FIGURES / "emission_maps.png", dpi=150)

    summary = {"central": central[["R_kg", "R_As_kg", "emitted_kg", "mean_exposed_km2", "mean_emitting_km2"]].round(2).to_dict("index"),
               "uncertainty_observed_2022": unc.to_dict("records")}
    (TABLES / "results_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(scen.round(2).to_string(index=False))
    print(unc.to_string(index=False))


if __name__ == "__main__":
    main()
