"""Figures for the model report (docs/GSL_Dust_Model_Report.pdf).

Reads the outputs of run_model.py (HYSPLIT transport), run_episodes.py,
extract_perry_report.py and the pilot/resolution tables, and writes
results/figures/report_*.png."""
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gsl_dust import decisions, hysplit_run, hysplit_transport
from gsl_dust.config import CELL_M, DATA_INTERIM, DATA_RAW, FIGURES, ROOT, TABLES
from gsl_dust.grid import LATTICE

NAVY, RUST, GREY, GREEN, LIGHT = "#16324a", "#b5651d", "#9aa3a8", "#2f6b3a", "#c9d6e3"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150})
EXTENT = (LATTICE.x0 / 1e3, (LATTICE.x0 + LATTICE.nx * CELL_M) / 1e3,
          (LATTICE.y1 - LATTICE.ny * CELL_M) / 1e3, LATTICE.y1 / 1e3)
SCEN = ["record_low_4188.5", "4192", "4195", "healthy_low_4198", "healthy_high_4205"]
LEVEL = decisions.SCENARIO_LEVELS_NGVD29_FT


def save(fig, name):
    fig.savefig(FIGURES / f"report_{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def runs():
    r = pd.read_csv(TABLES / "runs.csv")
    r["group"], r["scen"] = r.run.str.split("/").str[0], r.run.str.split("/").str[1]
    return r


def fig_arsenic():
    p = pd.read_csv(DATA_RAW / "fieldflux" / "perry2019_appendixG_pm10_arsenic.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.1))
    ax[0].hist(p.pm10_as_ppm, bins=np.arange(10, 86, 4), color=RUST, edgecolor="white")
    ax[0].axvline(18.5, color=NAVY, ls="--", lw=1.4)
    ax[0].text(19.5, ax[0].get_ylim()[1] * 0.92, "lakebed sediment\n(USGS, kriged) ≈ 18.5", color=NAVY, fontsize=8, va="top")
    ax[0].axvline(p.pm10_as_ppm.median(), color=NAVY, lw=1.4)
    ax[0].text(0.98, 0.95, f"PM10 median {p.pm10_as_ppm.median():.0f} mg/kg\n(solid line)", transform=ax[0].transAxes,
               ha="right", va="top", color=NAVY, fontsize=8, bbox=dict(fc="white", ec="none", alpha=0.9))
    ax[0].set(title="Arsenic in PM10-size lakebed dust (120 sites)", xlabel="Arsenic (mg/kg)", ylabel="Number of sites")
    e = p.eta_pair.dropna().sort_values().to_numpy()
    ax[1].axvspan(0.8, 2.0, color=LIGHT, alpha=0.6, label="previous assumed range (0.8–2.0)")
    ax[1].scatter(e, np.arange(len(e)), color=RUST, s=22, zorder=3, label="site pairs within 5 km")
    ax[1].axvline(1.75, color=NAVY, lw=1.6, label="measured central η = 1.75")
    ax[1].axvline(1.3, color=GREY, ls="--", lw=1.2, label="previous central η = 1.3")
    ax[1].set(title="Measured enrichment η (PM10 ÷ sediment arsenic)", xlabel="η", yticks=[], xlim=(0, 3.2))
    ax[1].legend(fontsize=7.5, loc="upper left")
    save(fig, "arsenic")


def fig_transport_checks():
    pil = pd.read_csv(TABLES / "pilot_hysplit_vs_ode.csv").groupby("zone")[["hysplit", "ode"]].mean() * 100
    order = ["South shore (Saltair)", "Farmington Bay", "North arm, west"]
    pil = pil.loc[order]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.0), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    y = np.arange(len(pil))
    ax[0].barh(y - 0.2, pil.hysplit, 0.38, color=NAVY, label="HYSPLIT (HRRR)")
    ax[0].barh(y + 0.2, pil.ode, 0.38, color=GREY, label="ODE prototype")
    for i, (h, o) in enumerate(zip(pil.hysplit, pil.ode)):
        ax[0].text(h + 1, i - 0.2, f"{h:.0f}%", va="center", fontsize=8)
        ax[0].text(o + 1, i + 0.2, f"{o:.0f}%", va="center", fontsize=8)
    ax[0].set(yticks=y, yticklabels=["South shore\n(Saltair)", "Farmington\nBay", "North arm, west\n(~90 km)"],
              xlabel="% of released dust reaching the valley", title="Pilot: May 3, 2022 storm", xlim=(0, 100))
    ax[0].invert_yaxis(); ax[0].legend(fontsize=7.5, loc="lower right")
    sizes = [47.5, 43.6, 40.2]
    ax[1].bar(["2.5 µm", "5 µm\n(central)", "10 µm"], sizes, color=[LIGHT, NAVY, LIGHT], width=0.55)
    for i, v in enumerate(sizes):
        ax[1].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=8)
    ax[1].set(title="Particle size bounds", ylabel="% reaching the valley", ylim=(0, 60))
    zr = pd.read_csv(TABLES / "zone_resolution_test.csv").set_index("area") * 100
    yy = np.arange(len(zr))
    ax[2].scatter(zr["10"], yy, color=NAVY, s=40, label="10 km zone", zorder=3)
    ax[2].scatter(zr["5"], yy, color=RUST, s=40, marker="D", label="its 5 km zones (avg)", zorder=3)
    ax[2].set(yticks=yy, yticklabels=[a.replace(" (", "\n(") for a in zr.index], xlim=(0, 100),
              xlabel="% reaching the valley", title="Source-zone size")
    ax[2].legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    fig.tight_layout()
    save(fig, "transport_checks")


def fig_superposition():
    s = pd.read_csv(TABLES / "superposition_check.csv")
    s = s[(s.R_direct_kg > 1) & (s.R_superposition_kg > 1)]
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    lim = (1, 1e5)
    ax.plot(lim, lim, color=GREY, lw=1, label="1 : 1")
    ax.scatter(s.R_direct_kg, s.R_superposition_kg, color=NAVY, s=26, zorder=3)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="Direct HYSPLIT run (kg)",
           ylabel="Unit-source superposition (kg)", title="Valley arrival per validation episode")
    ax.legend(fontsize=8)
    save(fig, "superposition")


def fig_source_maps():
    hours = pd.date_range(pd.Timestamp(decisions.SIM_START), pd.Timestamp(decisions.SIM_END), freq="h", inclusive="left")
    e = np.load(DATA_INTERIM / "central_e_zone.npy")
    tr = hysplit_transport.Transfer(hours, e.shape[1])
    arrived = (e * tr.frac).sum(0)
    zones, zone_of_cell = hysplit_run.zones()
    cells = np.load(DATA_INTERIM / "central_per_cell.npz")["cells"]
    em_cell = np.load(DATA_INTERIM / "central_per_cell.npz")["emitted"]
    share = arrived / np.maximum(e.sum(0), 1e-9)
    g_em, g_ar = np.full(LATTICE.shape, np.nan), np.full(LATTICE.shape, np.nan)
    g_em.ravel()[cells] = em_cell / 1e3
    g_ar.ravel()[cells] = em_cell / 1e3 * share[zone_of_cell]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.6))
    from matplotlib.colors import LogNorm
    for a, g, title, norm in ((ax[0], g_em, "PM10 dust emitted, 2022 (t per km²)", LogNorm(1e-3, 30)),
                              (ax[1], g_ar, "…of which reached the Salt Lake Valley (t per km²)", LogNorm(1e-4, 5))):
        im = a.imshow(g, extent=EXTENT, cmap="YlOrBr", norm=norm)
        a.set(title=title, xlabel="UTM east (km)", ylabel="UTM north (km)")
        a.grid(False)
        fig.colorbar(im, ax=a, shrink=0.75)
    save(fig, "source_maps")
    return arrived.sum(), e.sum()


def fig_seasonal():
    h = pd.read_csv(TABLES / "central_hourly.csv", index_col=0, parse_dates=True)
    m = h[["R_rate_kg_h", "R_rate_ode_kg_h"]].resample("ME").sum() / 1e3
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.1), gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.arange(12)
    ax[0].bar(x - 0.2, m.R_rate_kg_h, 0.38, color=NAVY, label="HYSPLIT")
    ax[0].bar(x + 0.2, m.R_rate_ode_kg_h, 0.38, color=GREY, label="ODE prototype")
    ax[0].set(xticks=x, xticklabels=[d.strftime("%b") for d in m.index], ylabel="t per month",
              title="Dust reaching the valley by month, 2022 (central run)")
    ax[0].legend(fontsize=8)
    cum = h.R_rate_kg_h.cumsum() / 1e3
    ax[1].plot(cum.index, cum, color=NAVY, lw=1.8)
    ax[1].fill_between(cum.index, cum, color=NAVY, alpha=0.08)
    top = h.R_rate_kg_h.resample("D").sum().sort_values(ascending=False).head(3)
    for d, v in top.items():
        ax[1].annotate(f"{d:%b %d}: +{v / 1e3:.0f} t", (d, cum.loc[d:].iloc[0]), xytext=(-70, 12),
                       textcoords="offset points", fontsize=7.5, arrowprops=dict(arrowstyle="-", color=GREY))
    ax[1].set(title="Cumulative arrival through 2022", ylabel="t")
    ax[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    fig.tight_layout()
    save(fig, "seasonal")
    return h.R_rate_kg_h.sum() / 1e3, m


def fig_scenarios(r):
    c = r[r.group == "central"].set_index("scen")
    lv = np.array([LEVEL[s] for s in SCEN])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, col, unit, scale, color in ((ax[0], "R_kg", "t", 1e3, NAVY), (ax[1], "R_As_kg", "kg", 1, RUST)):
        j = r[r.group == "joint"]
        q = np.array([[j[j.scen == s][col].quantile(p) / scale for p in (0.05, 0.25, 0.5, 0.75, 0.95)] for s in SCEN])
        a.fill_between(lv, q[:, 0], q[:, 4], color=color, alpha=0.12, label="joint ensemble 5–95%")
        a.fill_between(lv, q[:, 1], q[:, 3], color=color, alpha=0.25, label="joint ensemble 25–75%")
        a.plot(lv, [c.loc[s, col] / scale for s in SCEN], "o-", color=color, lw=1.8, label="central, HYSPLIT")
        if col == "R_kg":
            a.plot(lv, [c.loc[s, "R_ode_kg"] / scale for s in SCEN], "s--", color=GREY, lw=1.2, label="central, ODE prototype")
        a.axvspan(4198, 4205, color=GREEN, alpha=0.08)
        a.text(4201.5, 0.97, "Strike Team\nhealthy range", transform=a.get_xaxis_transform(), ha="center", va="top",
               fontsize=7.5, color=GREEN)
        a.set(yscale="log", xlabel="Lake level held constant (ft, NGVD29 gauge datum)",
              ylabel=f"per year ({unit})", title=("Dust" if col == "R_kg" else "Arsenic") + " reaching the valley, 2022 weather")
        a.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    save(fig, "scenarios")


def fig_uncertainty(r):
    rows = [("Emission", "emission"), ("Chemistry", "chemistry"), ("All combined", "joint")]
    fig, ax = plt.subplots(1, 2, figsize=(10, 2.6))
    for a, col, scale, unit, color in ((ax[0], "R_kg", 1e3, "t", NAVY), (ax[1], "R_As_kg", 1, "kg", RUST)):
        for i, (name, g) in enumerate(rows):
            s = r[(r.group == g) & (r.scen == "observed_2022")][col] / scale
            lo, md, hi = s.quantile([0.05, 0.5, 0.95])
            a.plot([lo, hi], [i, i], color=color, lw=6, alpha=0.35, solid_capstyle="round")
            a.scatter([md], [i], color=color, s=30, zorder=3)
            label = (f"no effect on dust mass ({md:,.0f} {unit})" if hi / lo < 1.001 else
                     f"5–95%: {lo:.2g} – {hi:,.0f} {unit}" if hi > 10 else f"5–95%: {lo:.2g} – {hi:.2g} {unit}")
            a.text(hi * 1.3, i, label, va="center", fontsize=7.5)
        a.set(xscale="log", yticks=range(3), yticklabels=[n for n, _ in rows], ylim=(-0.6, 2.6),
              title=("Dust" if col == "R_kg" else "Arsenic") + f" reaching the valley, 2022 ({unit}); 200 members each")
        a.invert_yaxis()
        a.set_xlim(right=a.get_xlim()[1] * 30)
    fig.tight_layout()
    save(fig, "uncertainty")


def fig_sensitivity(r):
    em = r[(r.group == "emission") & (r.scen == "observed_2022")]
    names = {"p_u_star_t": "u*t threshold friction velocity", "p_flux_ref": "flux_ref reference flux",
             "p_z0_playa": "z0 playa roughness", "p_u_star_ref": "u*ref of the reference flux", "p_p": "p wind exponent",
             "p_g_areal": "G replenishment", "p_s0_areal": "S(0) initial reservoir", "p_phi_scale": "φ survey scale",
             "p_wet_depth_mm": "rain that wets the surface", "p_dry_tau_days": "surface drying time",
             "p_fringe_m": "capillary fringe height", "p_snow_m": "snow cutoff"}
    rho = pd.Series({v: spearmanr(em[k], np.log(em.R_kg))[0] for k, v in names.items()}).sort_values(key=np.abs)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    noise = 1.96 / np.sqrt(len(em) - 1)
    ax.axvspan(-noise, noise, color=LIGHT, alpha=0.5, label=f"within chance (|ρ| < {noise:.2f})")
    ax.barh(rho.index, rho.values, color=[NAVY if abs(v) > noise else GREY for v in rho.values])
    for i, v in enumerate(rho.values):
        ax.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=7.5)
    ax.set(xlim=(-1, 1), xlabel="Rank correlation with log(dust reaching the valley)",
           title="Which emission inputs drive the result (200 emission-ensemble members)")
    ax.legend(fontsize=7.5, loc="lower right")
    save(fig, "sensitivity")
    return rho


def fig_validation():
    d = json.loads((ROOT / "validation" / "heldout_design.json").read_text())
    c = pd.read_csv(TABLES / "episode_model_conc.csv")
    held = set(d["heldout_episode_ids"])
    rows = []
    for ep in d["episodes"]:
        cc = c[c.episode == ep["id"]]
        rows.append({"id": ep["id"], "date": pd.Timestamp(ep["start"]), "obs": ep["peak_pm10"],
                     "mod": cc.conc_ug_m3.max() if len(cc) else 0.0, "held": ep["id"] in held})
    v = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    x = np.arange(len(v))
    ax.bar(x, v.obs, color=[LIGHT if not h else "#e8c9a9" for h in v.held], width=0.7, label="observed PM10 peak (all sources)")
    ax.scatter(x, v["mod"].clip(lower=0.05), color=NAVY, s=24, zorder=3, label="modeled Great Salt Lake dust, peak at any monitor")
    ax.set(yscale="log", ylim=(0.05, 2000), xticks=x, ylabel="µg/m³ (log scale)",
           title="Stage 4: the 30 PM10 episodes of 2022 (orange bars = held-out events)")
    ax.set_xticklabels([f"{t:%b %d}" for t in v.date], rotation=70, fontsize=7)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.text(0.005, 0.02, "dots on the bottom edge: essentially no modeled lake dust", transform=ax.transAxes, fontsize=7,
            color=GREY)
    save(fig, "validation")

    m = pd.read_csv(TABLES / "episode_model_vs_monitor.csv", parse_dates=["time"])
    jun = m[m.episode.isin([12, 13, 14])]
    site = jun.groupby("site_id").conc_ug_m3.max().idxmax()
    w = jun[jun.site_id == site].sort_values("time").drop_duplicates("time")
    held = " (held-out monitor)" if site in set(d["heldout_monitors"]) else ""
    county = {"49-035": "Salt Lake County", "49-049": "Utah County", "49-057": "Weber County"}[site[:6]]
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.plot(w.time, w.pm10, color=GREY, lw=1.4, label=f"observed PM10, all sources")
    ax.plot(w.time, w.conc_ug_m3.clip(lower=0.01), color=NAVY, lw=1.8, label="modeled Great Salt Lake dust (HYSPLIT)")
    ax.set(yscale="log", ylim=(0.01, 2000), ylabel="µg/m³ (log scale)",
           title=f"June 16–19, 2022 windstorms at monitor {site}, {county}{held}")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d %Hh"))
    ax.legend(fontsize=7.5, loc="upper left")
    save(fig, "validation_timeseries")
    return v


README_FIGURES = ["scenarios", "seasonal", "source_maps", "transport_checks", "arsenic", "validation", "sensitivity"]


def copy_for_readme():
    """The README shows a few figures; results/ is not in git, so keep copies in docs/figures."""
    dest = ROOT / "docs" / "figures"
    dest.mkdir(exist_ok=True)
    for name in README_FIGURES:
        shutil.copy(FIGURES / f"report_{name}.png", dest / f"{name}.png")
    shutil.copy(FIGURES / "phi_surface_map.png", dest / "erodibility_map.png")


def main():
    r = runs()
    fig_arsenic(); fig_transport_checks(); fig_superposition()
    arrived, emitted = fig_source_maps()
    total, monthly = fig_seasonal()
    fig_scenarios(r); fig_uncertainty(r)
    rho = fig_sensitivity(r)
    v = fig_validation()
    print(f"source map check: arrived {arrived / 1e3:.0f} t of {emitted / 1e3:.0f} t emitted; hourly total {total:.0f} t")
    print((monthly / monthly.sum() * 100).round(1).T.to_string())
    print(rho.round(2).to_string())
    print(v.round(2).to_string())
    copy_for_readme()


if __name__ == "__main__":
    main()
