"""The model chain in the guide's order, streamed month by month:

    lake levels + terrain -> X_i(t)                       (Stage 1)
    HRRR winds, rain, snow, surface survey -> u*_i(t), phi_i(t)
    threshold law, S_i(0), G_i -> beta_i(t)               (Stage 2)
    ODE prototype: S -> I -> R, R_As with kV, kL          (Stage 3.1)
    HYSPLIT (unit-source runs) applied to beta*S          (Stage 3.2)

Lake-level scenarios (Stage 5) call `run` with the same weather and a different
level series. Parameters come from params.P or an ensemble draw."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gsl_dust import decisions, emission, erodibility, hrrr, ode, transfer
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
    phi_surface: np.ndarray = None  # per-cell erodible surface fraction from the field survey
    zone_of_cell: np.ndarray = None  # HYSPLIT source zone of each cell
    n_zones: int = 0

    @property
    def lakebed_area_m2(self):
        return self.hyps.lakebed_fraction * CELL_M ** 2

    @property
    def flat_index(self):
        return self.hyps.cells


@dataclass
class Result:
    hours: pd.DatetimeIndex
    r_rate: np.ndarray        # kg/h into the valley (HYSPLIT when given, else the ODE prototype)
    ras_rate: np.ndarray      # kg arsenic/h into the valley
    per_cell: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    @property
    def R(self):
        return np.cumsum(self.r_rate)

    @property
    def R_As(self):
        return np.cumsum(self.ras_rate)


_WEATHER_CACHE = {}


def _weather(var, start, end, idx):
    """Hourly HRRR field at lakebed cells. Cached: scenarios and ensemble members
    reuse identical weather, so it is read from disk once per process."""
    key = (var, pd.Timestamp(start).value, pd.Timestamp(end).value, idx.tobytes().__hash__())
    if key not in _WEATHER_CACHE:
        a = hrrr.load(var, start, end).reshape(-1, LATTICE.ny * LATTICE.nx)
        _WEATHER_CACHE[key] = np.ascontiguousarray(a[:, idx])
    return _WEATHER_CACHE[key]


def run(setup: Setup, levels_ft: pd.DataFrame, p: dict, start, end, eta=None, hysplit=None) -> Result:
    """levels_ft: hourly NAVD88 ft, columns north_arm and south_arm (south system).
    hysplit: a hysplit_transport.Transfer; when given, R and R_As come from HYSPLIT
    and the ODE prototype's values are kept in diagnostics for comparison."""
    idx = setup.flat_index
    eta = p["eta"] if eta is None else eta
    phi_surface = setup.phi_surface * p["phi_scale"] if setup.phi_surface is not None else p["phi_surface"]
    months = pd.date_range(pd.Timestamp(start).normalize().replace(day=1), end, freq="MS", inclusive="left")
    lv0 = levels_ft.loc[[pd.Timestamp(start)]]
    X0 = setup.hyps.exposure(lv0["north_arm"].to_numpy(), lv0["south_arm"].to_numpy())[0]
    # Initial emittable sediment and replenishment (decisions.REPLENISH_ON): on the
    # exposed lakebed, or on the whole lakebed as in the first prototype.
    area0 = X0 * CELL_M ** 2 if decisions.REPLENISH_ON == "exposed" else setup.lakebed_area_m2
    S0 = p["s0_areal"] * area0
    K = emission.calibrate_K(p["flux_ref"], p["u_star_ref"], p["u_star_t"], p["p"],
                             np.maximum(p["s0_areal"] * CELL_M ** 2, 1e-12), CELL_M ** 2)

    S, I = S0.astype(float).copy(), np.zeros(len(idx))
    wet = None
    all_hours, r_parts, ras_parts, e_parts, eas_parts = [], [], [], [], []
    tot = ode.Totals(*(np.zeros(len(idx)) for _ in range(4)))
    exposed_km2, emitting_km2 = [], []
    missing_hours, supplied = 0, float(S0.sum())
    zones = setup.zone_of_cell if setup.zone_of_cell is not None else np.zeros(len(idx), int)
    nz = max(setup.n_zones, 1)
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
        phi = erodibility.physics_phi(phi_surface, wet_t, p["wet_depth_mm"], snow, p["snow_m"], fringe)
        u_star = emission.u_star_log_law(np.hypot(u10, v10), p["z0_playa"])
        beta = emission.beta(K, X, phi, u_star, p["u_star_t"], p["p"])
        kV, kL = transfer.rates(u10, v10, setup.dist_valley, setup.sector_start, setup.sector_width,
                                p["v_dep"], p["mix_depth"])
        rep_area = X * CELL_M ** 2 if decisions.REPLENISH_ON == "exposed" else np.broadcast_to(
            setup.lakebed_area_m2, X.shape)
        G = p["g_areal"] * rep_area / emission.SECONDS_PER_YEAR            # (T, n) kg/s
        supplied += float(G.sum() * HOUR)

        S, I, t, r_rate, ras_rate, e_z, eas_z = ode.run_from(S, I, G, beta, kV, kL, setup.c_as, eta, HOUR,
                                                             zone_of_cell=zones, n_zones=nz)
        for k in ("emitted", "arrived", "removed", "arrived_as"):
            getattr(tot, k)[:] += getattr(t, k)
        all_hours.append(hours)
        r_parts.append(r_rate)
        ras_parts.append(ras_rate)
        e_parts.append(e_z)
        eas_parts.append(eas_z)
        exposed_km2.append((X * CELL_M ** 2).sum(1) / 1e6)
        emitting_km2.append((X * phi * CELL_M ** 2).sum(1) / 1e6)
        missing_hours += int(missing.sum())
    hours_all = all_hours[0].append(all_hours[1:]) if len(all_hours) > 1 else all_hours[0]
    r_ode, ras_ode = np.concatenate(r_parts), np.concatenate(ras_parts)
    e_zone, eas_zone = np.concatenate(e_parts), np.concatenate(eas_parts)
    diag = {"exposed_km2": np.concatenate(exposed_km2), "emitting_km2": np.concatenate(emitting_km2),
            "supplied_kg": supplied, "missing_hrrr_hours": missing_hours,
            "r_rate_ode": r_ode, "ras_rate_ode": ras_ode, "e_zone": e_zone}
    if hysplit is not None:
        r, ras = hysplit.arrival(e_zone), hysplit.arrival(eas_zone)
        diag["hysplit_coverage"] = hysplit.coverage()
    else:
        r, ras = r_ode, ras_ode
    res = Result(hours_all, r, ras)
    res.per_cell = {"emitted": tot.emitted, "arrived_ode": tot.arrived, "removed_ode": tot.removed,
                    "arrived_as_ode": tot.arrived_as, "S_final": S, "I_final": I, "S0": S0}
    res.diagnostics = diag
    return res


def levels_by_arm_ft(levels_m: pd.DataFrame) -> pd.DataFrame:
    """lake_levels output (NAVD88 m by arm) -> NAVD88 ft for north arm and south system."""
    return pd.DataFrame({"north_arm": navd88_m_to_ft(levels_m["north_arm"]),
                         "south_arm": navd88_m_to_ft(levels_m["south_arm"])}, index=levels_m.index)
