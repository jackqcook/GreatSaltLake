"""Build the hourly analysis table: merge PM10, weather, lake state, and derived
features into one clean, hourly-indexed DataFrame per Section 8 of the modeling
direction memo ("Build an hourly analysis table").
"""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from gsl_model.acquisition import aqs_pm10, asos_weather, usgs_lake
from gsl_model.acquisition.mixing_height import climatological_mixing_height_m
from gsl_model.config import (
    DATA_PROCESSED, PRIMARY_SITE, START_YEAR, END_YEAR,
)

MOUNTAIN_TZ = ZoneInfo("America/Denver")

MS_PER_KT = 0.514444
MS_PER_MPH = 0.44704


def build_analysis_table(force_refresh: bool = False) -> pd.DataFrame:
    pm10 = aqs_pm10.fetch_range(START_YEAR, END_YEAR, PRIMARY_SITE, force=force_refresh)
    wx_raw = asos_weather.fetch_range(PRIMARY_SITE.asos_station, START_YEAR, END_YEAR, force=force_refresh)
    wx = asos_weather.to_hourly(wx_raw)
    lake = usgs_lake.fetch_elevation(START_YEAR, END_YEAR, force=force_refresh)

    # --- Build a complete hourly UTC index spanning the data ---
    start = min(pm10["datetime_utc"].min(), wx["datetime_utc"].min())
    end = max(pm10["datetime_utc"].max(), wx["datetime_utc"].max())
    idx = pd.date_range(start, end, freq="1h", tz="UTC")
    df = pd.DataFrame({"datetime_utc": idx})

    df = df.merge(pm10[["datetime_utc", "pm10_ugm3"]], on="datetime_utc", how="left")
    df = df.merge(wx, on="datetime_utc", how="left")

    # --- Local time features ---
    df["datetime_local"] = df["datetime_utc"].dt.tz_convert(MOUNTAIN_TZ)
    df["local_hour"] = df["datetime_local"].dt.hour + df["datetime_local"].dt.minute / 60.0
    df["month"] = df["datetime_local"].dt.month
    df["date_local"] = df["datetime_local"].dt.tz_localize(None).dt.normalize()
    df["dow"] = df["datetime_local"].dt.dayofweek  # 0=Mon
    df["is_weekend"] = df["dow"] >= 5

    # --- Lake state: daily -> hourly forward fill ---
    lake = lake.rename(columns={"date": "date_local"})
    df = df.merge(lake, on="date_local", how="left")
    df["lake_elev_ft_navd88"] = df["lake_elev_ft_navd88"].ffill().bfill()
    df["exposed_playa_km2"] = usgs_lake.exposed_playa_area(df["lake_elev_ft_navd88"].values)

    # --- Units: convert wind to SI ---
    df["wind_speed_ms"] = df["wind_speed_kt"] * MS_PER_KT
    df["wind_gust_ms"] = df["wind_gust_kt"] * MS_PER_KT
    df["wind_dir_deg"] = df["wind_dir_deg"]
    df["precip_mm"] = df["precip_in"] * 25.4
    df["temp_c"] = (df["temp_f"] - 32) * 5 / 9

    # --- Mixing height proxy ---
    df["mixing_height_m"] = climatological_mixing_height_m(
        df["local_hour"].values, df["month"].values, df["wind_speed_ms"].fillna(0).values
    )

    # --- PM10 cleaning: continuous PM10 monitors can report small negative
    # values near zero due to instrument noise; clip at 0 but keep NaN as NaN. ---
    df["pm10_ugm3"] = df["pm10_ugm3"].clip(lower=0)

    # --- Lag features ---
    df["pm10_lag1"] = df["pm10_ugm3"].shift(1)
    df["pm10_lag24"] = df["pm10_ugm3"].shift(24)

    # --- Rolling background estimate: trailing 30-day (720h) 10th percentile,
    # a simple robust proxy for "background" particulate not driven by local
    # lake-dust events (Section 3, P_bg(t)). ---
    df["pm10_background"] = (
        df["pm10_ugm3"].rolling(window=24 * 30, min_periods=24 * 7).quantile(0.10)
    )
    df["pm10_background"] = df["pm10_background"].bfill()

    # --- Recent precipitation (wets/crusts the playa surface, suppresses
    # emission for a period after rain). ---
    df["precip_48h_mm"] = df["precip_mm"].rolling(window=48, min_periods=1).sum()

    df = df.sort_values("datetime_utc").reset_index(drop=True)

    out_path = DATA_PROCESSED / "hourly_analysis_table.parquet"
    df.to_parquet(out_path, index=False)
    return df


if __name__ == "__main__":
    df = build_analysis_table()
    print(df.shape)
    print(df.columns.tolist())
    print(df[["datetime_utc", "pm10_ugm3", "wind_speed_ms", "wind_dir_deg",
              "lake_elev_ft_navd88", "exposed_playa_km2", "mixing_height_m"]].tail(10))
    print("\nMissing data summary:")
    print(df.isna().mean().sort_values(ascending=False).head(15))
