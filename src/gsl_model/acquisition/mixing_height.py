"""Boundary-layer mixing height H(t).

The modeling memo specifies NOAA HRRR as the source for gridded boundary-layer
height. Pulling and decoding HRRR GRIB2 archives requires eccodes/cfgrib system
libraries and multi-GB downloads that are not practical in this environment, so
this module implements a documented climatological proxy instead, built from
published Salt Lake Valley boundary-layer climatology (Whiteman et al., the
Salt Lake Valley has a strong, well-studied diurnal mixing-height cycle driven
by basin heating/cooling and frequent cold-air-pool/inversion conditions in
winter):

  - A diurnal sinusoid: shallow, stable nighttime boundary layer (~200-400 m),
    growing through the morning to a deep, well-mixed afternoon boundary layer
    (~1500-2500 m in summer, ~600-1000 m in winter).
  - A seasonal amplitude modulation (deeper convective mixing in summer,
    shallower in winter, consistent with more frequent winter inversions).
  - A wind-speed adjustment: mixing height is pulled down further, and the
    nighttime minimum held longer, under calm conditions (classic inversion
    signature), and increased somewhat under strong synoptic flow that
    suppresses stagnation.

This is an explicit, documented approximation -- NOT observed data. It is
wired in as a single function so it can be swapped for a real HRRR-derived H(t)
column later without touching the rest of the pipeline (see Section 9,
"Extensions", of docs/MODEL_EXPLAINER.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def climatological_mixing_height_m(
    local_hour: np.ndarray | pd.Series,
    month: np.ndarray | pd.Series,
    wind_speed_ms: np.ndarray | pd.Series,
) -> np.ndarray:
    local_hour = np.asarray(local_hour, dtype=float)
    month = np.asarray(month, dtype=float)
    wind_speed_ms = np.asarray(wind_speed_ms, dtype=float)

    # Seasonal peak (summer) and trough (winter) mixing-height amplitude, m.
    summer_peak_m = 2200.0
    winter_peak_m = 900.0
    night_floor_summer_m = 350.0
    night_floor_winter_m = 200.0

    # Smooth seasonal interpolation via cosine over month-of-year (peak ~ July).
    season_phase = 2 * np.pi * (month - 7) / 12.0
    season_frac = 0.5 * (1 + np.cos(season_phase))  # 1 in July, 0 in January
    day_peak_m = winter_peak_m + (summer_peak_m - winter_peak_m) * season_frac
    night_floor_m = night_floor_winter_m + (night_floor_summer_m - night_floor_winter_m) * season_frac

    # Diurnal shape: smooth rise from ~08:00 local, peak ~15:00-16:00 local,
    # decay back to the nighttime floor by ~20:00, flat overnight.
    hour_phase = (local_hour - 15.0) / 24.0 * 2 * np.pi
    diurnal_frac = np.clip(0.5 * (1 + np.cos(hour_phase)), 0.0, 1.0) ** 1.5

    h = night_floor_m + (day_peak_m - night_floor_m) * diurnal_frac

    # Calm-wind inversion suppression: below ~2 m/s, damp toward the nighttime
    # floor even during daytime hours (persistent cold-air pool signature).
    calm_factor = np.clip(wind_speed_ms / 3.0, 0.0, 1.0)
    h = night_floor_m + (h - night_floor_m) * (0.4 + 0.6 * calm_factor)

    # Strong synoptic flow mixes somewhat deeper even off-peak.
    windy_boost = np.clip((wind_speed_ms - 6.0) / 10.0, 0.0, 1.0) * 300.0
    h = h + windy_boost

    return h
