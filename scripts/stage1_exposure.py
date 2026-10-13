"""Stage 1: terrain -> lakebed grid and arms -> per-cell hypsometry -> check
against USGS elevation-area tables -> X_i(t) for the simulation period."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from affine import Affine

from gsl_dust import boundaries, decisions, exposure, lake_levels
from gsl_dust.config import CELL_M, DATA_INTERIM, FIGURES, TABLES
from gsl_dust.datums import navd88_ft_to_m, navd88_m_to_ft
from gsl_dust.grid import LATTICE

HYPS_PATH = DATA_INTERIM / "cell_hypsometry.npz"
ARMS_PATH = DATA_INTERIM / "arms_20m.npy"


def main():
    print("pages in overview file:", exposure.list_overview_pages())
    terrain = exposure.load_terrain()
    f = int(round(CELL_M / exposure.ANALYSIS_RES_M))
    top_m = navd88_ft_to_m(decisions.LAKEBED_TOP_FT_NAVD88)
    lakebed = np.isfinite(terrain) & (terrain <= top_m)
    print(f"terrain {terrain.shape}, valid {np.isfinite(terrain).mean():.3f}, "
          f"lakebed {lakebed.sum() * exposure.ANALYSIS_RES_M ** 2 / 1e6:.0f} km2")

    raster_transform = LATTICE.transform * Affine.scale(1 / f)
    arm = exposure.label_arms(lakebed, raster_transform, boundaries.causeway_utm())
    np.save(ARMS_PATH, arm)
    hyps = exposure.build_cell_hypsometry(terrain, arm, f)
    hyps.save(HYPS_PATH)
    print(f"lakebed cells: {len(hyps.cells)}")

    val = exposure.validate_against_usgs(hyps)
    val.to_csv(TABLES / "stage1_area_validation.csv", index=False)
    sel = val[val["level_ft_navd88"].isin([4180.0, 4190.0, 4192.0, 4195.0, 4200.0, 4205.0, 4210.0, 4215.0])]
    print(sel.round(3).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for k, c in (("north", "#b5651d"), ("south", "#16324a"), ("total", "#2f6b3a")):
        ax[0].plot(val["level_ft_navd88"], val[f"usgs_{k}_km2"], color=c, lw=2.5, alpha=0.4, label=f"USGS {k}")
        ax[0].plot(val["level_ft_navd88"], val[f"model_{k}_km2"], color=c, lw=1, ls="--", label=f"model {k}")
        ax[1].plot(val["level_ft_navd88"], 100 * val[f"rel_err_{k}"], color=c, label=k)
    ax[0].set(xlabel="Lake level (ft NAVD88)", ylabel="Wet area (km²)", title="Wet area vs USGS tables")
    ax[1].set(xlabel="Lake level (ft NAVD88)", ylabel="Model − USGS (%)", title="Relative difference")
    ax[1].axhline(0, color="grey", lw=0.8)
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURES / "stage1_area_validation.png", dpi=150)

    days = pd.date_range(decisions.SIM_START, decisions.SIM_END, freq="D", inclusive="left")
    lv = lake_levels.hourly_levels_navd88_m(days)
    hn, hs = navd88_m_to_ft(lv["north_arm"].to_numpy()), navd88_m_to_ft(lv["south_arm"].to_numpy())
    X = hyps.exposure(hn, hs)
    exposed = pd.DataFrame({"north_arm_ft_ngvd29": hn - 3.47585958, "south_arm_ft_ngvd29": hs - 3.47585958,
                            "exposed_km2": X.sum(1) * CELL_M ** 2 / 1e6}, index=days)
    exposed.to_csv(TABLES / "stage1_exposed_area_daily.csv")
    print(exposed.describe().round(1).to_string())

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(exposed.index, exposed["exposed_km2"], color="#b5651d")
    ax[0].set(ylabel="Exposed lakebed (km²)", title="Total exposed lakebed, 2022")
    ax2 = ax[0].twinx(); ax2.plot(exposed.index, exposed["south_arm_ft_ngvd29"], color="#16324a", lw=0.8)
    ax2.set_ylabel("South-arm level (ft NGVD29)", color="#16324a")
    i_low = int(np.argmin(hs))
    grid = np.full(LATTICE.ny * LATTICE.nx, np.nan); grid[hyps.cells] = X[i_low]
    im = ax[1].imshow(grid.reshape(LATTICE.shape), cmap="YlOrBr", vmin=0, vmax=1,
                      extent=(LATTICE.x0 / 1e3, (LATTICE.x0 + LATTICE.nx * CELL_M) / 1e3,
                              (LATTICE.y1 - LATTICE.ny * CELL_M) / 1e3, LATTICE.y1 / 1e3))
    ax[1].set(title=f"X_i on {days[i_low]:%Y-%m-%d} (lowest level)", xlabel="UTM E (km)", ylabel="UTM N (km)")
    fig.colorbar(im, ax=ax[1], label="exposed fraction of cell")
    fig.tight_layout(); fig.savefig(FIGURES / "stage1_exposure.png", dpi=150)
    (TABLES / "stage1_summary.json").write_text(json.dumps({
        "lakebed_cells": int(len(hyps.cells)),
        "lakebed_km2_at_top": float(hyps.lakebed_fraction.sum()),
        "max_abs_rel_err_total_above_4180ft": float(val.loc[val.level_ft_navd88 >= 4180, "rel_err_total"].abs().max()),
        "terrain": json.loads((DATA_INTERIM / "terrain_meta.json").read_text()),
    }, indent=1))


if __name__ == "__main__":
    main()
