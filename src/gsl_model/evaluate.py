"""Shared train/val/test splitting, metrics, and physical sanity checks
(Section 7 of the modeling direction memo).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EVENT_PM10_THRESHOLD = 40.0   # ug/m3, well above typical Hawthorne background
EVENT_MIN_GAP_HOURS = 6
EVENT_MIN_DURATION_HOURS = 2


@dataclass
class EventBlock:
    start_idx: int
    end_idx: int   # inclusive
    peak_idx: int
    peak_value: float


def detect_events(
    df: pd.DataFrame,
    threshold: float = EVENT_PM10_THRESHOLD,
    min_gap_hours: int = EVENT_MIN_GAP_HOURS,
    min_duration_hours: int = EVENT_MIN_DURATION_HOURS,
) -> list[EventBlock]:
    pm10 = df["pm10_ugm3"].to_numpy()
    above = np.nan_to_num(pm10, nan=-1) >= threshold

    blocks = []
    i = 0
    n = len(above)
    while i < n:
        if above[i]:
            j = i
            while j < n:
                # extend through the current run and any gap <= min_gap_hours
                # that is itself followed by more above-threshold hours
                if above[j]:
                    j += 1
                    continue
                # look ahead for a re-crossing within min_gap_hours
                k = j
                while k < n and k - j <= min_gap_hours and not above[k]:
                    k += 1
                if k < n and above[k] and (k - j) <= min_gap_hours:
                    j = k
                    continue
                break
            end = j - 1
            if end - i + 1 >= min_duration_hours:
                seg = pm10[i:end + 1]
                peak_offset = int(np.nanargmax(seg))
                blocks.append(EventBlock(
                    start_idx=i, end_idx=end,
                    peak_idx=i + peak_offset, peak_value=float(seg[peak_offset]),
                ))
            i = end + 1
        else:
            i += 1
    return blocks


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def time_based_event_aware_split(
    df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15,
) -> Split:
    """Chronological split, snapped so that no detected dust event straddles a
    split boundary (Section 7: "hold out entire dust events, not only
    individual hours")."""
    n = len(df)
    events = detect_events(df)

    def snap(target_idx: int) -> int:
        for ev in events:
            if ev.start_idx <= target_idx <= ev.end_idx:
                # push the boundary to just before this event starts
                return ev.start_idx
        return target_idx

    train_end = snap(int(n * train_frac))
    val_end = snap(int(n * (train_frac + val_frac)))
    val_end = max(val_end, train_end + 1)

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n)

    return Split(
        train=df.iloc[train_idx].reset_index(drop=True),
        val=df.iloc[val_idx].reset_index(drop=True),
        test=df.iloc[test_idx].reset_index(drop=True),
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
    )


# --- Metrics -----------------------------------------------------------

def mae(pred: np.ndarray, obs: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(obs)
    return float(np.mean(np.abs(pred[mask] - obs[mask])))


def rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(obs)
    return float(np.sqrt(np.mean((pred[mask] - obs[mask]) ** 2)))


def bias(pred: np.ndarray, obs: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(obs)
    return float(np.mean(pred[mask] - obs[mask]))


def event_recall(pred: np.ndarray, obs: np.ndarray, threshold: float = EVENT_PM10_THRESHOLD) -> float:
    """Fraction of observed high-PM10 hours that the model also predicts above threshold."""
    mask = np.isfinite(pred) & np.isfinite(obs)
    obs_hi = obs[mask] >= threshold
    if obs_hi.sum() == 0:
        return float("nan")
    pred_hi = pred[mask] >= threshold
    return float(np.mean(pred_hi[obs_hi]))


def event_precision(pred: np.ndarray, obs: np.ndarray, threshold: float = EVENT_PM10_THRESHOLD) -> float:
    mask = np.isfinite(pred) & np.isfinite(obs)
    pred_hi = pred[mask] >= threshold
    if pred_hi.sum() == 0:
        return float("nan")
    obs_hi = obs[mask] >= threshold
    return float(np.mean(obs_hi[pred_hi]))


def peak_error_and_timing(
    pred: np.ndarray, obs: np.ndarray, events: list[EventBlock], max_timing_search_h: int = 6,
) -> pd.DataFrame:
    """For each held-out event: error in predicted peak magnitude, and timing
    offset (hours) between predicted and observed peak."""
    rows = []
    n = len(obs)
    for ev in events:
        lo = max(0, ev.start_idx - 2)
        hi = min(n, ev.end_idx + 3)
        obs_seg = obs[lo:hi]
        pred_seg = pred[lo:hi]
        if not np.isfinite(obs_seg).any() or not np.isfinite(pred_seg).any():
            continue
        obs_peak_offset = int(np.nanargmax(obs_seg))
        pred_peak_offset = int(np.nanargmax(np.nan_to_num(pred_seg, nan=-1)))
        rows.append({
            "start_idx": ev.start_idx, "end_idx": ev.end_idx,
            "obs_peak": float(np.nanmax(obs_seg)),
            "pred_peak": float(np.nanmax(pred_seg)),
            "peak_error": float(np.nanmax(pred_seg) - np.nanmax(obs_seg)),
            "timing_offset_h": pred_peak_offset - obs_peak_offset,
        })
    return pd.DataFrame(rows)


# --- Physical sanity checks (Section 7) --------------------------------

def physical_sanity_checks(df: pd.DataFrame, lake_contrib: np.ndarray) -> dict:
    """Checks that the inferred lake contribution behaves the way physics
    says it should, independent of predictive accuracy."""
    out = {}
    lc = np.asarray(lake_contrib)

    from gsl_model.config import WIND_DIR_FROM_LAKE_TO_HAWTHORNE_DEG
    wind_dir = df["wind_dir_deg"].to_numpy()
    angle_off = np.abs(((wind_dir - WIND_DIR_FROM_LAKE_TO_HAWTHORNE_DEG + 180) % 360) - 180)
    not_upwind = angle_off > 90
    out["mean_contribution_when_not_upwind"] = float(np.nanmean(lc[not_upwind])) if not_upwind.any() else float("nan")
    out["mean_contribution_when_upwind"] = float(np.nanmean(lc[~not_upwind])) if (~not_upwind).any() else float("nan")

    wet = df["precip_48h_mm"].to_numpy() > 5.0
    out["mean_contribution_after_rain"] = float(np.nanmean(lc[wet])) if wet.any() else float("nan")
    out["mean_contribution_when_dry"] = float(np.nanmean(lc[~wet])) if (~wet).any() else float("nan")

    from gsl_model.physics_model import INITIAL_GUESS  # bounds context only
    calm = df["wind_speed_ms"].to_numpy() < 4.0
    out["mean_contribution_when_calm"] = float(np.nanmean(lc[calm])) if calm.any() else float("nan")
    out["mean_contribution_when_windy"] = float(np.nanmean(lc[~calm])) if (~calm).any() else float("nan")

    out["passed_upwind_check"] = out["mean_contribution_when_upwind"] > out["mean_contribution_when_not_upwind"]
    out["passed_precip_check"] = out["mean_contribution_when_dry"] > out["mean_contribution_after_rain"]
    out["passed_calm_check"] = out["mean_contribution_when_windy"] > out["mean_contribution_when_calm"]
    return out


def flag_boundary_parameters(theta: np.ndarray, tol: float = 0.02) -> list[str]:
    """Flag any fitted parameter sitting within `tol` (fractional) of its
    bound -- a sign the data doesn't identify that parameter well and the
    point estimate should not be over-interpreted (Section 7)."""
    from gsl_model.physics_model import PARAM_NAMES, LOWER_BOUNDS, UPPER_BOUNDS
    flags = []
    for name, val, lo, hi in zip(PARAM_NAMES, theta, LOWER_BOUNDS, UPPER_BOUNDS):
        span = hi - lo
        if span <= 0:
            continue
        if (val - lo) / span < tol:
            flags.append(f"{name} is at its LOWER bound ({val:.4g})")
        elif (hi - val) / span < tol:
            flags.append(f"{name} is at its UPPER bound ({val:.4g})")
    return flags
