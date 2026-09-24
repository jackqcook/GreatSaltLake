"""Physics-informed state-space model of GSL dust contribution to PM10 at a
monitor, per Section 3 of the modeling direction memo.

State equation (discrete, dt = 1 hour):

    P[t+1] = P[t] + dt * { k_v(t) [P_bg(t) - P[t]] + E_GSL(t-tau)/H(t)
                             + E_local - lambda * P[t] }

Lake source term:

    E_GSL(t) = beta * A_eff(t) * R(t) * max(U(t) - U_c, 0)^n * D_i(t)

  - A_eff(t): exposed playa area (km^2), from USGS hypsometry at the observed
    lake elevation (usgs_lake.exposed_playa_area).
  - R(t): surface-wetness suppression factor, R(t) = exp(-precip_48h / s),
    s = fitted decay scale (mm). Recent precipitation crusts/wets the playa
    and suppresses emission for a period afterward.
  - U(t): wind speed (m/s); U_c: fitted threshold wind speed; n: fitted
    emission exponent.
  - D_i(t): directional transport factor, 1 when wind blows exactly from the
    lake source toward the monitor, 0 when wind is perpendicular or from the
    opposite direction: D_i = clip(cos(wind_dir - bearing_lake_to_monitor), 0, 1).
  - tau: advective travel time from source to monitor, tau = d / U(t) (h),
    d = config.DIST_HAWTHORNE_TO_SOUTH_SHORE_KM. E_GSL is looked up at t-tau
    (nearest hour) so dust isn't seen at the monitor before it could have
    physically arrived.
  - k_v(t) = k_v0 + k_v1 * U(t): ventilation/advection rate increases with
    wind speed (Section 3: "faster atmospheric exchange pulls local
    concentration toward the upwind background state").
  - E_local: constant local/non-lake source term. Deliberately NOT given
    weekday/hour structure here -- that pattern is what the hybrid residual
    model (hybrid.py) is for for (Section 1: physics model should not simply
    learn traffic patterns).

Free parameters theta = [beta, U_c, n, lambda_, k_v0, k_v1, E_local, precip_scale]
fit by nonlinear least squares on one-step-ahead prediction error (using the
OBSERVED P[t] as the starting state each hour), which is the standard way to
calibrate a state-space model of this kind without letting simulation drift
dominate the loss. A separate `simulate` method free-runs the recursion for
multi-step forecasts and counterfactual scenarios.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from gsl_model.config import (
    DIST_HAWTHORNE_TO_SOUTH_SHORE_KM,
    WIND_DIR_FROM_LAKE_TO_HAWTHORNE_DEG,
    U_C_BOUNDS_MPH,
    N_BOUNDS,
)

MS_PER_MPH = 0.44704

PARAM_NAMES = ["beta", "U_c", "n", "lambda_", "k_v0", "k_v1", "E_local", "precip_scale"]

# theta bounds, in order of PARAM_NAMES.
LOWER_BOUNDS = np.array([
    1e-6,                              # beta > 0
    U_C_BOUNDS_MPH[0] * MS_PER_MPH,    # U_c
    N_BOUNDS[0],                       # n
    1e-4,                              # lambda_ (removal rate, /h)
    1e-4,                              # k_v0 (/h)
    0.0,                               # k_v1 (/h per m/s)
    0.0,                               # E_local (ug/m3/h)
    1.0,                               # precip_scale (mm)
])
UPPER_BOUNDS = np.array([
    3.0,                               # beta
    U_C_BOUNDS_MPH[1] * MS_PER_MPH,
    N_BOUNDS[1],
    2.0,                               # lambda_
    2.0,                               # k_v0
    0.5,                               # k_v1
    20.0,                              # E_local
    50.0,                              # precip_scale
])

INITIAL_GUESS = np.array([
    0.01,                              # beta
    15.0 * MS_PER_MPH,                 # U_c ~ 15 mph rough threshold guess
    1.5,                               # n
    0.15,                              # lambda_
    0.10,                              # k_v0
    0.02,                              # k_v1
    2.0,                               # E_local
    5.0,                               # precip_scale
])

MAX_LAG_HOURS = 12


@dataclass
class PhysicsModelFeatures:
    pm10: np.ndarray
    pm10_bg: np.ndarray
    wind_speed_ms: np.ndarray
    wind_dir_deg: np.ndarray
    exposed_playa_km2: np.ndarray
    mixing_height_m: np.ndarray
    precip_48h_mm: np.ndarray
    n: int = field(init=False)

    def __post_init__(self):
        self.n = len(self.pm10)


def _directional_factor(wind_dir_deg: np.ndarray) -> np.ndarray:
    theta = np.radians(wind_dir_deg - WIND_DIR_FROM_LAKE_TO_HAWTHORNE_DEG)
    return np.clip(np.cos(theta), 0.0, 1.0)


def build_features(df: pd.DataFrame) -> PhysicsModelFeatures:
    return PhysicsModelFeatures(
        pm10=df["pm10_ugm3"].to_numpy(dtype=float),
        pm10_bg=df["pm10_background"].to_numpy(dtype=float),
        wind_speed_ms=df["wind_speed_ms"].to_numpy(dtype=float),
        wind_dir_deg=df["wind_dir_deg"].to_numpy(dtype=float),
        exposed_playa_km2=df["exposed_playa_km2"].to_numpy(dtype=float),
        mixing_height_m=df["mixing_height_m"].to_numpy(dtype=float),
        precip_48h_mm=df["precip_48h_mm"].to_numpy(dtype=float),
    )


def _raw_emission(theta: np.ndarray, feat: PhysicsModelFeatures) -> np.ndarray:
    beta, U_c, n, *_ = theta
    precip_scale = theta[7]
    U = feat.wind_speed_ms
    D_i = _directional_factor(feat.wind_dir_deg)
    R = np.exp(-feat.precip_48h_mm / precip_scale)
    wind_term = np.power(np.clip(U - U_c, 0.0, None), n)
    raw = beta * feat.exposed_playa_km2 * R * wind_term * D_i
    # Missing weather at hour j must not poison OTHER hours' lag-shifted
    # lookups; treat a missing-weather hour as contributing zero lake emission.
    return np.nan_to_num(raw, nan=0.0)


def _lagged_emission(raw_emission: np.ndarray, wind_speed_ms: np.ndarray) -> np.ndarray:
    """Shift each hour's raw emission back by its own advective travel time."""
    with np.errstate(divide="ignore", invalid="ignore"):
        tau_h = DIST_HAWTHORNE_TO_SOUTH_SHORE_KM / (wind_speed_ms * 3.6)
    tau_h = np.nan_to_num(tau_h, nan=MAX_LAG_HOURS, posinf=MAX_LAG_HOURS)
    lag = np.clip(np.round(tau_h).astype(int), 0, MAX_LAG_HOURS)

    n = len(raw_emission)
    idx = np.arange(n) - lag
    valid = idx >= 0
    out = np.zeros(n)
    out[valid] = raw_emission[idx[valid]]
    return out


def one_step_prediction(theta: np.ndarray, feat: PhysicsModelFeatures, dt: float = 1.0) -> np.ndarray:
    """Predict P[t+1] from observed P[t] for every t (vectorized, one-step-ahead)."""
    _, _, _, lambda_, k_v0, k_v1, E_local, _ = theta

    raw_e = _raw_emission(theta, feat)
    e_gsl_lagged = _lagged_emission(raw_e, feat.wind_speed_ms)

    k_v = k_v0 + k_v1 * feat.wind_speed_ms
    P = feat.pm10
    dPdt = (
        k_v * (feat.pm10_bg - P)
        + e_gsl_lagged / np.clip(feat.mixing_height_m, 50.0, None)
        + E_local
        - lambda_ * P
    )
    return P + dt * dPdt


def _residuals(
    theta: np.ndarray, feat: PhysicsModelFeatures, y_next: np.ndarray, mask: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    pred = one_step_prediction(theta, feat)
    r = pred[mask] - y_next[mask]
    if weights is not None:
        r = r * weights[mask]
    return r


@dataclass
class FittedPhysicsModel:
    theta: np.ndarray
    param_names: list
    cost: float
    success: bool
    message: str

    def as_dict(self) -> dict:
        return dict(zip(self.param_names, self.theta))


def fit(
    df_train: pd.DataFrame, x0: np.ndarray | None = None, verbose: int = 0,
    event_weight_power: float = 1.0,
) -> FittedPhysicsModel:
    """Fit by (weighted) nonlinear least squares.

    Plain (unweighted) least squares on hourly data is dominated by the ~98%
    of hours that are unremarkable background -- squared error barely notices
    whether the model gets rare dust spikes right, so the optimizer can
    satisfy the objective by pushing the wind threshold U_c up until the lake
    term almost never fires and quietly relying on the mean-reversion/
    background terms alone. That defeats the purpose of the model. We
    therefore weight each hour's squared error by its observed concentration
    (weight = y_next ** event_weight_power on the SQUARED error, i.e. the
    residual itself is multiplied by y_next ** (event_weight_power / 2)), so
    dust-event hours are not drowned out by clean-air hours. Set
    event_weight_power=0 to recover plain OLS.
    """
    feat = build_features(df_train)
    y = feat.pm10
    y_next = np.roll(y, -1)
    mask = (
        ~np.isnan(y) & ~np.isnan(y_next) & ~np.isnan(feat.pm10_bg)
        & ~np.isnan(feat.wind_speed_ms) & ~np.isnan(feat.wind_dir_deg)
    )
    mask[-1] = False  # last row has no "next"

    if event_weight_power > 0:
        weights = np.clip(y_next, 1.0, None) ** (event_weight_power / 2.0)
        weights = weights / np.nanmean(weights[mask])  # keep loss scale comparable to OLS
    else:
        weights = None

    x0 = INITIAL_GUESS if x0 is None else x0
    result = least_squares(
        _residuals, x0, bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
        args=(feat, y_next, mask, weights), verbose=verbose, method="trf",
        x_scale=np.maximum(np.abs(x0), 1e-3),
    )
    return FittedPhysicsModel(
        theta=result.x, param_names=PARAM_NAMES, cost=result.cost,
        success=result.success, message=result.message,
    )


def predict(df: pd.DataFrame, theta: np.ndarray) -> np.ndarray:
    feat = build_features(df)
    return one_step_prediction(theta, feat)


def lake_contribution(
    df: pd.DataFrame, theta: np.ndarray, exposed_playa_km2_override: np.ndarray | None = None,
) -> np.ndarray:
    """The E_GSL(t-tau)/H(t) term alone: the model's estimate, in ug/m3/h added
    to the state equation, of how much of the dynamics is attributable to
    Great Salt Lake dust at each hour (used for source-attribution plots,
    the Section 6 physical-sanity checks, and lake-level counterfactuals)."""
    feat = build_features(df)
    if exposed_playa_km2_override is not None:
        feat.exposed_playa_km2 = exposed_playa_km2_override
    raw_e = _raw_emission(theta, feat)
    e_gsl_lagged = _lagged_emission(raw_e, feat.wind_speed_ms)
    return e_gsl_lagged / np.clip(feat.mixing_height_m, 50.0, None)


def simulate(
    df: pd.DataFrame,
    theta: np.ndarray,
    p0: float | None = None,
    exposed_playa_km2_override: np.ndarray | None = None,
    dt: float = 1.0,
) -> np.ndarray:
    """Free-running multi-step simulation: P[t+1] depends on the model's OWN
    previous prediction, not the observation. Used for held-out storm
    forecasts and lake-level counterfactuals (Section 8, "Run lake-level
    counterfactuals"), where by construction there is no "observed" trajectory
    under the hypothetical scenario.
    """
    feat = build_features(df)
    if exposed_playa_km2_override is not None:
        feat.exposed_playa_km2 = exposed_playa_km2_override

    _, _, _, lambda_, k_v0, k_v1, E_local, _ = theta
    raw_e = _raw_emission(theta, feat)
    e_gsl_lagged = _lagged_emission(raw_e, feat.wind_speed_ms)
    k_v = k_v0 + k_v1 * feat.wind_speed_ms
    H = np.clip(feat.mixing_height_m, 50.0, None)

    n = feat.n
    P_sim = np.empty(n)
    P_sim[0] = feat.pm10[0] if p0 is None else p0
    for t in range(n - 1):
        dPdt = (
            k_v[t] * (feat.pm10_bg[t] - P_sim[t])
            + e_gsl_lagged[t] / H[t]
            + E_local
            - lambda_ * P_sim[t]
        )
        P_sim[t + 1] = max(0.0, P_sim[t] + dt * dPdt)
    return P_sim
