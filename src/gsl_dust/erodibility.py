"""Stage 2.1 erodibility phi_i(t), physics branch. The guide: RF/XGBoost with
field labels, "otherwise physics". No GSL field labels are in hand yet, so:

    phi_i(t) = phi_surface * dry(t) * no_snow(t) * above_fringe_i(t)

- phi_surface: fraction of exposed lakebed that emits (crust absent/disturbed).
- dry(t): 0 while recent rain keeps the surface above the ~4% soil-moisture
  shutoff found on GSL playa; recent rain = exponentially decaying rain memory.
- no_snow(t): 0 under snow.
- above_fringe_i(t): share of exposed area more than `fringe_m` above the water
  line, i.e. outside the capillary fringe that keeps shoreline sediment moist.
  Computed from the per-cell terrain curves, so it is cell- and level-specific.

When PI-SWERL field labels are available, `ml_phi` replaces this function; the
rest of the model does not change."""
import numpy as np

from gsl_dust.datums import navd88_m_to_ft


def rain_memory(precip_mm, tau_hours, w0=None):
    """w_t = w_{t-1} exp(-1/tau) + P_t, over axis 0 (hours). Returns (w, w_last)."""
    decay = np.exp(-1.0 / tau_hours)
    w = np.empty_like(precip_mm, dtype=np.float64)
    prev = np.zeros(precip_mm.shape[1:]) if w0 is None else w0
    for t in range(precip_mm.shape[0]):
        prev = prev * decay + np.nan_to_num(precip_mm[t])
        w[t] = prev
    return w, prev


def above_fringe(hyps, h_north_ft, h_south_ft, fringe_m):
    """Fraction of each cell's exposed lakebed lying above the capillary fringe."""
    df = navd88_m_to_ft(fringe_m)
    x = hyps.exposure(h_north_ft, h_south_ft)
    x_dry = hyps.exposure(np.asarray(h_north_ft) + df, np.asarray(h_south_ft) + df)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(x > 0, x_dry / np.where(x > 0, x, 1.0), 0.0)


def physics_phi(phi_surface, wetness_mm, wet_depth_mm, snow_depth_m, snow_m, fringe_frac):
    dry = (wetness_mm < wet_depth_mm).astype(np.float32)
    no_snow = (np.nan_to_num(snow_depth_m) < snow_m).astype(np.float32)
    return phi_surface * dry * no_snow * fringe_frac
