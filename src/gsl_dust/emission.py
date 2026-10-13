"""Stage 2: emission beta_i(t) and the sediment reservoir S_i, per the guide:

    beta_i(t) = K_i X_i(t) phi_i(t) [max(u*_i(t) - u*t, 0)]^p
    dS_i/dt   = G_i - beta_i S_i

K_i is calibrated so that, with a full reservoir, the flux from emitting ground
at u*_ref equals the field flux flux_ref ("calibrate K accordingly"):

    K_i = flux_ref * A_cell / ((u*_ref - u*t)^p * S_i(0))
"""
import numpy as np

KAPPA = 0.4
Z_WIND = 10.0
SECONDS_PER_YEAR = 365.25 * 86400


def u_star_log_law(u10, z0):
    """Neutral log-law friction velocity over the lakebed from 10 m wind speed."""
    return KAPPA * np.asarray(u10) / np.log(Z_WIND / z0)


def calibrate_K(flux_ref, u_star_ref, u_star_t, p, s0_cell, cell_area_m2):
    return flux_ref * cell_area_m2 / ((u_star_ref - u_star_t) ** p * s0_cell)


def beta(K, X, phi, u_star, u_star_t, p):
    """Emission rate coefficient (1/s)."""
    return K * X * phi * np.maximum(u_star - u_star_t, 0.0) ** p


def fit_threshold_law(u_star, flux, p_bounds=(1.0, 6.0)):
    """Fit flux = a * max(u* - u*t, 0)^p to field measurements (e.g. a PI-SWERL
    run table). Returns (a, u*t, p). Used once GSL field fluxes are available."""
    from scipy.optimize import least_squares
    u = np.asarray(u_star, float)
    f = np.asarray(flux, float)

    def resid(th):
        a, ut, p = th
        return np.log1p(a * np.maximum(u - ut, 0.0) ** p / f.max()) - np.log1p(f / f.max())

    x0 = [f.max() / max(u.max() - 0.3, 0.1) ** 3, min(0.3, u.min() * 0.9), 3.0]
    lo = [0.0, 0.0, p_bounds[0]]
    hi = [np.inf, u.max(), p_bounds[1]]
    r = least_squares(resid, x0, bounds=(lo, hi))
    return tuple(r.x)
