"""Run the model for the simulation period: central run, lake-level scenarios
under identical weather (Stage 5), and uncertainty ensembles reported separately
for emission, chemistry and transport (guide). Transport is HYSPLIT (Stage 3.2,
unit-source runs from run_hysplit_transfer.py) or the ODE prototype (Stage 3.1).
Requires stage1_exposure.py outputs and HRRR inputs."""
import argparse
import dataclasses
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union

from gsl_dust import (boundaries, chemistry, decisions, hysplit_run, hysplit_transport, lake_levels, model,
                      params, surface_survey)
from gsl_dust.config import CELL_M, DATA_INTERIM, TABLES
from gsl_dust.datums import navd88_ft_to_m, ngvd29_ft_to_navd88_ft
from gsl_dust.exposure import ANALYSIS_RES_M, CellHypsometry
from gsl_dust.grid import LATTICE
from gsl_dust.transfer import valley_geometry

GROUPS = {
    "emission": ["z0_playa", "u_star_t", "p", "flux_ref", "u_star_ref", "s0_areal", "g_areal",
                 "phi_scale", "wet_depth_mm", "dry_tau_days", "fringe_m", "snow_m"],
    "chemistry": ["eta"],
    "transport": ["v_dep", "mix_depth"],
}
SETUP_PATH = DATA_INTERIM / "setup.npz"


def build_setup():
    hyps = CellHypsometry.load(DATA_INTERIM / "cell_hypsometry.npz")
    X, Y = LATTICE.centers()
    cx, cy = X.ravel()[hyps.cells], Y.ravel()[hyps.cells]
    phi_s = surface_survey.erodible_fraction(cx, cy)
    _, zone_of_cell = hysplit_run.zones()
    extra = dict(phi_surface=phi_s, zone_of_cell=zone_of_cell, n_zones=int(zone_of_cell.max()) + 1)
    if SETUP_PATH.exists():
        z = np.load(SETUP_PATH)
        return model.Setup(hyps, cx, cy, z["dist"], z["start"], z["width"], z["c_as"], **extra), dict(z)
    lakebed_poly = unary_union([box(x - CELL_M / 2, y - CELL_M / 2, x + CELL_M / 2, y + CELL_M / 2)
                                for x, y in zip(cx, cy)])
    valley = boundaries.valley_receptor_utm(lakebed_poly)
    dist, start, width = valley_geometry(cx, cy, valley)

    terrain = np.load(DATA_INTERIM / f"terrain_{int(ANALYSIS_RES_M)}m.npy")
    sites = chemistry.site_values(chemistry.load_samples())
    from pyproj import Transformer
    from gsl_dust.config import CRS
    sx, sy = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(sites.longitude.to_numpy(), sites.latitude.to_numpy())
    col = ((sx - LATTICE.x0) / ANALYSIS_RES_M).astype(int)
    row = ((LATTICE.y1 - sy) / ANALYSIS_RES_M).astype(int)
    inside = (row >= 0) & (row < terrain.shape[0]) & (col >= 0) & (col < terrain.shape[1])
    elev = np.full(len(sites), np.nan)
    elev[inside] = terrain[row[inside], col[inside]]
    sites["elev_ft_navd88"] = elev / 0.3048
    sites["on_lakebed"] = sites["elev_ft_navd88"] <= decisions.LAKEBED_TOP_FT_NAVD88
    sites.to_csv(TABLES / "arsenic_sites.csv", index=False)
    lake = sites[sites["on_lakebed"]]
    c_med, c_lo, c_hi, kernel = chemistry.krige(lake, cx, cy)
    np.savez(SETUP_PATH, dist=dist, start=start, width=width, c_as=c_med, c_as_lo=c_lo, c_as_hi=c_hi)
    (TABLES / "arsenic_kriging.json").write_text(json.dumps({
        "n_lakebed_sites": int(len(lake)), "kernel": str(kernel),
        "c_as_ppm_median_range": [float(c_med.min() * 1e6), float(c_med.max() * 1e6)]}, indent=1))
    z = np.load(SETUP_PATH)
    return model.Setup(hyps, cx, cy, dist, start, width, c_med, **extra), dict(z)


def levels_observed(hours):
    return model.levels_by_arm_ft(lake_levels.hourly_levels_navd88_m(hours))


def levels_constant(hours, ngvd29_ft):
    h = float(ngvd29_ft_to_navd88_ft(ngvd29_ft))
    return pd.DataFrame({"north_arm": h, "south_arm": h}, index=hours)


def draw(rng, names):
    p = params.central()
    for k in names:
        q = params.P[k]
        if q.log and q.low > 0:
            p[k] = float(np.exp(rng.uniform(np.log(q.low), np.log(q.high))))
        else:
            p[k] = float(rng.uniform(q.low, q.high))
    return p


def summarize(res, label, extra=None):
    out = {"run": label, "R_kg": float(res.R[-1]), "R_As_kg": float(res.R_As[-1]),
           "peak_rate_kg_h": float(res.r_rate.max()), "peak_hour": str(res.hours[int(res.r_rate.argmax())]),
           "emitted_kg": float(res.per_cell["emitted"].sum()),
           "R_ode_kg": float(res.diagnostics["r_rate_ode"].sum()),
           "mean_exposed_km2": float(res.diagnostics["exposed_km2"].mean()),
           "mean_emitting_km2": float(res.diagnostics["emitting_km2"].mean())}
    out.update(extra or {})
    return out


_W = {}


def _init_worker(transport):
    setup, z = build_setup()
    start, end = pd.Timestamp(decisions.SIM_START), pd.Timestamp(decisions.SIM_END)
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    obs = levels_observed(hours)
    _W.update(setup=setup, start=start, end=end, sd_log=(np.log(z["c_as_hi"]) - np.log(z["c_as_lo"])) / (2 * 1.645),
              hys=hysplit_transport.Transfer(hours, setup.n_zones) if transport == "hysplit" else None,
              levels={"observed_2022": obs,
                      **{k: levels_constant(hours, v) for k, v in decisions.SCENARIO_LEVELS_NGVD29_FT.items()}})


def _member(task):
    group, m, name, p, c_shift, extra = task
    s = _W["setup"]
    if c_shift:
        s = dataclasses.replace(s, c_as=np.exp(np.log(s.c_as) + c_shift * _W["sd_log"]))
    res = model.run(s, _W["levels"][name], p, _W["start"], _W["end"], hysplit=_W["hys"])
    return summarize(res, f"{group}/{name}", {"member": m, **extra})


def main(n_ens, transport, workers=8):
    setup, z = build_setup()
    start, end = pd.Timestamp(decisions.SIM_START), pd.Timestamp(decisions.SIM_END)
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    hys = None
    if transport == "hysplit":
        hys = hysplit_transport.Transfer(hours, setup.n_zones)
        print(f"HYSPLIT coverage of zone-hours: {hys.coverage():.1%}", flush=True)
    obs = levels_observed(hours)
    p0 = params.central()
    rows = []
    run = lambda s, lv, p: model.run(s, lv, p, start, end, hysplit=hys)

    central = run(setup, obs, p0)
    rows.append(summarize(central, "central/observed"))
    pd.DataFrame({"R_rate_kg_h": central.r_rate, "R_As_rate_kg_h": central.ras_rate,
                  "R_rate_ode_kg_h": central.diagnostics["r_rate_ode"],
                  "exposed_km2": central.diagnostics["exposed_km2"],
                  "emitting_km2": central.diagnostics["emitting_km2"]}, index=central.hours) \
        .to_csv(TABLES / "central_hourly.csv")
    np.savez(DATA_INTERIM / "central_per_cell.npz", cells=setup.hyps.cells, **central.per_cell)
    np.save(DATA_INTERIM / "central_e_zone.npy", central.diagnostics["e_zone"])

    scen_levels = {"observed_2022": obs, **{k: levels_constant(hours, v) for k, v in decisions.SCENARIO_LEVELS_NGVD29_FT.items()}}
    for name, lv in scen_levels.items():
        if name != "observed_2022":
            rows.append(summarize(run(setup, lv, p0), f"central/{name}"))
        print(rows[-1], flush=True)

    rng = np.random.default_rng(7)
    sd_log = (np.log(z["c_as_hi"]) - np.log(z["c_as_lo"])) / (2 * 1.645)
    groups = dict(GROUPS)
    if transport == "hysplit":
        groups.pop("transport")  # HYSPLIT transport uncertainty is reported from its own runs
    tasks = []
    for group, names in list(groups.items()) + [("joint", sum(groups.values(), []))]:
        for m in range(n_ens):
            p = draw(rng, names)
            c_shift = float(rng.standard_normal()) if group in ("chemistry", "joint") else 0.0  # z-score
            for name in (list(scen_levels) if group == "joint" else ["observed_2022"]):
                tasks.append((group, m, name, p, c_shift, {f"p_{k}": p[k] for k in names}))
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(transport,)) as ex:
        for n, row in enumerate(ex.map(_member, tasks, chunksize=4), 1):
            rows.append(row)
            if n % 50 == 0:
                print(f"{n}/{len(tasks)} ensemble runs", flush=True)
    pd.DataFrame(rows).to_csv(TABLES / "runs.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", type=int, default=200)
    ap.add_argument("--transport", choices=["hysplit", "ode"], default="hysplit")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    main(a.ensemble, a.transport, a.workers)
