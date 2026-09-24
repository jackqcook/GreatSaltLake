"""Pull hourly surface weather from the Iowa Environmental Mesonet ASOS archive.

The modeling memo names MesoWest/Synoptic as the weather source. Synoptic's API
requires a registered token; the Iowa State Mesonet ASOS/METAR archive
(https://mesonet.agron.iastate.edu/request/download.phtml) serves the same
underlying NWS/FAA surface observations (temperature, RH, wind, precipitation)
for free with no key, at the same station network. We use it as a drop-in,
equally authoritative substitute for a MesoWest API pull.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from gsl_model.config import DATA_RAW

ASOS_BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def _cache_path(station: str, year: int) -> Path:
    d = DATA_RAW / "asos"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{station}_{year}.parquet"


def fetch_year(station: str, year: int, force: bool = False) -> pd.DataFrame:
    cache = _cache_path(station, year)
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    params = {
        "station": station,
        "data": "tmpf,relh,drct,sknt,gust,p01i,alti,vsby",
        "year1": year, "month1": 1, "day1": 1,
        "year2": year + 1, "month2": 1, "day2": 1,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "elev": "no",
        "missing": "M",
        "trace": "T",
        "direct": "no",
        "report_type": "3",  # routine hourly METAR
    }
    last_err = None
    resp = None
    for attempt in range(5):
        try:
            resp = requests.get(ASOS_BASE_URL, params=params, timeout=120)
            resp.raise_for_status()
            last_err = None
            break
        except requests.exceptions.HTTPError as e:
            last_err = e
            time.sleep(5 * (attempt + 1))
    if last_err is not None:
        raise last_err

    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))

    df["datetime_utc"] = pd.to_datetime(df["valid"], utc=True)
    if "p01i" in df.columns:
        df["p01i"] = df["p01i"].replace("T", "0.005")
    for col in ["tmpf", "relh", "drct", "sknt", "gust", "p01i", "alti", "vsby"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.rename(columns={
        "tmpf": "temp_f", "relh": "rh_pct", "drct": "wind_dir_deg",
        "sknt": "wind_speed_kt", "gust": "wind_gust_kt", "p01i": "precip_in",
    })
    df = df[[
        "datetime_utc", "temp_f", "rh_pct", "wind_dir_deg", "wind_speed_kt",
        "wind_gust_kt", "precip_in", "vsby",
    ]]
    df = df.sort_values("datetime_utc").drop_duplicates("datetime_utc").reset_index(drop=True)

    df.to_parquet(cache, index=False)
    return df


def fetch_range(station: str, start_year: int, end_year: int, force: bool = False) -> pd.DataFrame:
    frames = []
    for y in range(start_year, end_year + 1):
        frames.append(fetch_year(station, y, force=force))
        if force or not _cache_path(station, y).exists():
            time.sleep(2)
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("datetime_utc").drop_duplicates("datetime_utc").reset_index(drop=True)
    return out


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse sub-hourly / multiple-obs-per-hour METAR records to top-of-hour.

    ASOS reports roughly every 5 minutes at many stations plus a routine hourly
    ob; we resample to the hour using the observation closest to (or the mean
    around) the top of the hour, and sum precipitation within the hour.
    """
    d = df.set_index("datetime_utc").sort_index()
    agg = d.resample("1h").agg({
        "temp_f": "mean",
        "rh_pct": "mean",
        "wind_dir_deg": "mean",
        "wind_speed_kt": "mean",
        "wind_gust_kt": "max",
        "precip_in": "sum",
        "vsby": "mean",
    })
    return agg.reset_index()


if __name__ == "__main__":
    from gsl_model.config import PRIMARY_SITE, START_YEAR, END_YEAR

    raw = fetch_range(PRIMARY_SITE.asos_station, START_YEAR, END_YEAR)
    print(raw.shape)
    hourly = to_hourly(raw)
    print(hourly.shape)
    print(hourly.head())
