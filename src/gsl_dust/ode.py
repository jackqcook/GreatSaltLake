"""Stage 3.1 ODE prototype, the guide's equations:

    dS_i/dt   = G_i - beta_i S_i
    dI_i/dt   = beta_i S_i - (kV_i + kL_i) I_i
    dR/dt     = sum_i kV_i I_i
    dR_As/dt  = sum_i cAs_i eta_i kV_i I_i

Coefficients are held constant within each hour and the linear system is solved
exactly over the hour, so mass is conserved to rounding error and the scheme
stays stable however large beta or k get. Mass leaving I in an hour is split
between valley arrival and removal in proportion kV : kL; dust counted into R
has left I, so each kg enters the valley once."""
from dataclasses import dataclass

import numpy as np


def _phi1(x, t):
    """(1 - exp(-x t)) / x, -> t as x -> 0."""
    x = np.asarray(x, float)
    small = np.abs(x * t) < 1e-8
    safe = np.where(small, 1.0, x)
    return np.where(small, t * (1 - 0.5 * x * t), -np.expm1(-safe * t) / safe)


def _dexp(b, k, t):
    """(exp(-b t) - exp(-k t)) / (k - b), -> t exp(-b t) as k -> b."""
    d = k - b
    close = np.abs(d * t) < 1e-6
    safe = np.where(close, 1.0, d)
    return np.where(close, t * np.exp(-b * t) * (1 - 0.5 * d * t),
                    (np.exp(-b * t) - np.exp(-k * t)) / safe)


@dataclass
class Totals:
    emitted: np.ndarray   # kg per cell, cumulative
    arrived: np.ndarray   # kg per cell reaching the valley (R = sum)
    removed: np.ndarray   # kg per cell deposited / transported elsewhere
    arrived_as: np.ndarray  # kg arsenic per cell reaching the valley


def step(S, I, G, beta, kV, kL, dt=3600.0):
    """Advance one interval. All arrays per cell; G in kg/s, rates in 1/s.
    Returns (S_new, I_new, emitted, to_valley, removed) in kg."""
    k = kV + kL
    pos = beta > 0
    b_safe = np.where(pos, beta, 1.0)
    S_eq = np.where(pos, G / b_safe, 0.0)
    S_new = np.where(pos, S_eq + (S - S_eq) * np.exp(-beta * dt), S + G * dt)
    emitted = np.where(pos, G * dt - (S_new - S), 0.0)
    a = beta * S - G  # beta*S(t) = G + a exp(-beta t)
    I_new = I * np.exp(-k * dt) + G * _phi1(k, dt) + a * _dexp(beta, k, dt)
    left = emitted - (I_new - I)
    k_safe = np.where(k > 0, k, 1.0)
    to_valley = np.where(k > 0, left * kV / k_safe, 0.0)
    removed = np.where(k > 0, left * kL / k_safe, 0.0)
    return S_new, I_new, emitted, to_valley, removed


def run(S0, G, beta_t, kV_t, kL_t, c_as, eta, dt=3600.0, record=None):
    """Integrate over T hours from S = S0, I = 0. beta_t, kV_t, kL_t: (T, n).
    Returns final S, I, cumulative per-cell Totals, and hourly valley-arrival
    series R_rate (T,), R_As_rate (T,) in kg per interval."""
    return run_from(S0, np.zeros(S0.shape[0]), G, beta_t, kV_t, kL_t, c_as, eta, dt, record)


def run_from(S0, I0, G, beta_t, kV_t, kL_t, c_as, eta, dt=3600.0, record=None):
    """As run(), continuing from a given state (S0, I0)."""
    n = S0.shape[0]
    S, I = S0.astype(float).copy(), I0.astype(float).copy()
    tot = Totals(*(np.zeros(n) for _ in range(4)))
    T = beta_t.shape[0]
    r_rate, ras_rate = np.zeros(T), np.zeros(T)
    for t in range(T):
        S, I, em, tv, rm = step(S, I, G, beta_t[t], kV_t[t], kL_t[t], dt)
        tot.emitted += em
        tot.arrived += tv
        tot.removed += rm
        tot.arrived_as += c_as * eta * tv
        r_rate[t] = tv.sum()
        ras_rate[t] = (c_as * eta * tv).sum()
        if record is not None:
            record.append((S.copy(), I.copy()))
    return S, I, tot, r_rate, ras_rate
