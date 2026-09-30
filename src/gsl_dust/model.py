"""The model chain in the guide's order, streamed month by month:

    lake levels + terrain -> X_i(t)                       (Stage 1)
    HRRR winds, rain, snow -> u*_i(t), phi_i(t)
    threshold law, S_i(0), G_i -> beta_i(t)               (Stage 2)
    ODE prototype: S -> I -> R, R_As with kV, kL          (Stage 3.1)

Lake-level scenarios (Stage 5) call `run` with the same weather and a different
level series. Parameters come from params.P or an ensemble draw."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gsl_dust import emission, erodibility, hrrr, ode, transfer
from gsl_dust.config import CELL_M
from gsl_dust.datums import navd88_m_to_ft
from gsl_dust.grid import LATTICE

HOUR = 3600.0


@dataclass
class Setup:
    """Everything that doesn't change with weather or parameters."""
    hyps: object              # exposure.CellHypsometry
    cell_x: np.ndarray        # UTM cell centers of lakebed cells
    cell_y: np.ndarray
    dist_valley: np.ndarray
    sector_start: np.ndarray
    sector_width: np.ndarray
    c_as: np.ndarray          # kg/kg per cell (kriged median or an ensemble draw)

    @property
    def lakebed_area_m2(self):
        return self.hyps.lakebed_fraction * CELL_M ** 2

    @property
    def flat_index(self):
        return self.hyps.cells


@dataclass
class Result:
    hours: pd.DatetimeIndex
    r_rate: np.ndarray        # kg/h into the valley
    ras_rate: np.ndarray      # kg arsenic/h into the valley
    per_cell: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    @property
    def R(self):
        return np.cumsum(self.r_rate)

    @property
    def R_As(self):
        return np.cumsum(self.ras_rate)


def _weather(var, start, end, idx):
    a = hrrr.load(var, start, end).reshape(-1, LATTICE.ny * LATTICE.nx)
    return a[:, idx]


def run(setup: Setup, levels_ft: pd.DataFrame, p: dict, start, end, eta=None) -> Result:
    """levels_ft: hourly NAVD88 ft, columns north_arm and south_arm (south system)."""
    idx = setup.flat_index
    area = setup.lakebed_area_m2
    S0 = p["s0_areal"] * area
    G = p["g_areal"] * area / emission.SECONDS_PER_YEAR
    K = emission.calibrate_K(p["flux_ref"], p["u_star_ref"], p["u_star_t"], p["p"], np.maximum(S0, 1e-12), CELL_M ** 2)
    eta = p["eta"] if eta is None else eta

    months = pd.date_range(pd.Timestamp(start).normalize().replace(day=1), end, freq="MS", inclusive="left")
    S, I = S0.astype(float).copy(), np.zeros(len(idx))
    wet = None
    all_hours, r_parts, ras_parts = [], [], []
    tot = ode.Totals(*(np.zeros(len(idx)) for _ in range(4)))
    exposed_km2, emitting_km2 = [], []
    missing_hours = 0
    for m0 in months:
        m1 = min(m0 + pd.offsets.MonthBegin(1), pd.Timestamp(end))
        a, b = max(m0, pd.Timestamp(start)), m1
        hours = pd.date_range(a, b, freq="h", inclusive="left")
        u10, v10 = _weather("u10", a, b, idx), _weather("v10", a, b, idx)
        precip, snow = _weather("precip", a, b, idx), _weather("snow_depth", a, b, idx)
        # missing HRRR hours: calm, dry, no snow (flagged in diagnostics)
        missing = np.isnan(u10).all(axis=1)
        u10, v10, precip, snow = (np.nan_to_num(x) for x in (u10, v10, precip, snow))

        lv = levels_ft.loc[hours]
        hn, hs = lv["north_arm"].to_numpy(), lv["south_arm"].to_numpy()
        X = setup.hyps.exposure(hn, hs)                                     # (T, n)
        fringe = erodibility.above_fringe(setup.hyps, hn, hs, p["fringe_m"])
        wet_t, wet = erodibility.rain_memory(precip, p["dry_tau_days"] * 24.0, wet)
        phi = erodibility.physics_phi(p["phi_surface"], wet_t, p["wet_depth_mm"], snow, p["snow_m"], fringe)
        u_star = emission.u_star_log_law(np.hypot(u10, v10), p["z0_playa"])
        beta = emission.beta(K, X, phi, u_star, p["u_star_t"], p["p"])
        kV, kL = transfer.rates(u10, v10, setup.dist_valley, setup.sector_start, setup.sector_width,
                                p["v_dep"], p["mix_depth"])

        S, I, t, r_rate, ras_rate = ode.run_from(S, I, G, beta, kV, kL, setup.c_as, eta, HOUR)
        for k in ("emitted", "arrived", "removed", "arrived_as"):
            getattr(tot, k)[:] += getattr(t, k)
        all_hours.append(hours)
        r_parts.append(r_rate)
        ras_parts.append(ras_rate)
        exposed_km2.append((X * CELL_M ** 2).sum(1) / 1e6)
        emitting_km2.append((X * phi * CELL_M ** 2).sum(1) / 1e6)
        missing_hours += int(missing.sum())
    hours_all = all_hours[0].append(all_hours[1:]) if len(all_hours) > 1 else all_hours[0]
    res = Result(hours_all, np.concatenate(r_parts), np.concatenate(ras_parts))
    res.per_cell = {"emitted": tot.emitted, "arrived": tot.arrived, "removed": tot.removed,
                    "arrived_as": tot.arrived_as, "S_final": S, "I_final": I, "S0": S0}
    res.diagnostics = {"exposed_km2": np.concatenate(exposed_km2), "emitting_km2": np.concatenate(emitting_km2),
                       "supplied_kg": float(S0.sum() + G.sum() * len(res.hours) * HOUR),
                       "missing_hrrr_hours": missing_hours}
    return res


def levels_by_arm_ft(levels_m: pd.DataFrame) -> pd.DataFrame:
    """lake_levels output (NAVD88 m by arm) -> NAVD88 ft for north arm and south system."""
    return pd.DataFrame({"north_arm": navd88_m_to_ft(levels_m["north_arm"]),
                         "south_arm": navd88_m_to_ft(levels_m["south_arm"])}, index=levels_m.index)
