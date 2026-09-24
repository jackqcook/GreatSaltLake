"""Pull hourly PM10 for a monitor from EPA AQS pre-generated bulk files.

We use the "AirData" pre-generated per-parameter/per-year files
(https://aqs.epa.gov/aqsweb/airdata/download_files.html) rather than the AQS API,
because the bulk files need no API key/registration and are a straight CSV
download. Each file covers ALL monitors nationwide for one year, so we download,
filter down to our site, and discard the rest.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

from gsl_model.config import DATA_RAW, AQS_PARAM_PM10, MonitorSite

AQS_BASE_URL = "https://aqs.epa.gov/aqsweb/airdata"


def _cache_path(year: int) -> Path:
    d = DATA_RAW / "aqs_pm10"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"hawthorne_pm10_{year}.parquet"


def fetch_year(year: int, site: MonitorSite, force: bool = False) -> pd.DataFrame:
    """Download one year of national hourly PM10 and return only rows for `site`."""
    cache = _cache_path(year)
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    url = f"{AQS_BASE_URL}/hourly_{AQS_PARAM_PM10}_{year}.zip"
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    csv_name = zf.namelist()[0]

    usecols = [
        "State Code", "County Code", "Site Num", "Date Local", "Time Local",
        "Date GMT", "Time GMT", "Sample Measurement", "Units of Measure",
        "MDL", "Uncertainty", "Qualifier", "Method Type", "State Name",
        "County Name",
    ]
    with zf.open(csv_name) as f:
        df = pd.read_csv(
            f, usecols=lambda c: c in usecols,
            dtype={"State Code": str, "County Code": str, "Site Num": str},
            low_memory=False,
        )

    mask = (
        (df["State Code"] == site.aqs_state)
        & (df["County Code"] == site.aqs_county)
        & (df["Site Num"] == site.aqs_site)
    )
    site_df = df.loc[mask].copy()
    site_df["datetime_utc"] = pd.to_datetime(
        site_df["Date GMT"] + " " + site_df["Time GMT"], utc=True
    )
    site_df = site_df.rename(columns={"Sample Measurement": "pm10_ugm3"})
    site_df = site_df[["datetime_utc", "pm10_ugm3", "Units of Measure", "MDL", "Qualifier"]]
    site_df = site_df.sort_values("datetime_utc").drop_duplicates("datetime_utc")

    site_df.to_parquet(cache, index=False)
    return site_df


def fetch_range(start_year: int, end_year: int, site: MonitorSite, force: bool = False) -> pd.DataFrame:
    frames = [fetch_year(y, site, force=force) for y in range(start_year, end_year + 1)]
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("datetime_utc").drop_duplicates("datetime_utc").reset_index(drop=True)
    return out


if __name__ == "__main__":
    from gsl_model.config import PRIMARY_SITE, START_YEAR, END_YEAR

    df = fetch_range(START_YEAR, END_YEAR, PRIMARY_SITE)
    print(df.shape)
    print(df.head())
    print(df["datetime_utc"].min(), df["datetime_utc"].max())
