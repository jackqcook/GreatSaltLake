"""Great Salt Lake water-surface elevation (USGS NWIS) and hypsometry (area/volume
vs. elevation, USGS topobathymetric model) -> exposed-playa area over time.

Gauge: USGS 10010000, "Great Salt Lake at Saltair Boat Harbor, UT" — the
standard south-arm lake-elevation gauge used in state reporting. Daily values,
parameter 62614 (lake/reservoir surface elevation, NAVD88, ft).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from gsl_model.config import DATA_RAW, REFERENCE_HEALTHY_ELEV_FT_NAVD88

USGS_SITE = "10010000"
USGS_PARAM_ELEV_NAVD88 = "62614"
NWIS_URL = "https://waterservices.usgs.gov/nwis/dv/"

HYPSOMETRY_FILES = {
    "total": "Great_Salt_Lake_2023_ElevAreaVolume_total.csv",
    "south_arm": "Great_Salt_Lake_2023_ElevAreaVolume_south_arm.csv",
    "north_arm": "Great_Salt_Lake_2023_ElevAreaVolume_north_arm.csv",
    "farmington_bay": "Great_Salt_Lake_2023_ElevAreaVolume_farmington_bay.csv",
    "bear_river_bay": "Great_Salt_Lake_2023_ElevAreaVolume_bear_river_bay.csv",
}


def _cache_path(start_year: int, end_year: int) -> Path:
    d = DATA_RAW / "usgs_lake"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"elevation_{start_year}_{end_year}.parquet"


def fetch_elevation(start_year: int, end_year: int, force: bool = False) -> pd.DataFrame:
    """Daily lake surface elevation (ft, NAVD88) at the Saltair gauge."""
    cache = _cache_path(start_year, end_year)
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    params = {
        "sites": USGS_SITE,
        "format": "json",
        "startDT": f"{start_year}-01-01",
        "endDT": f"{end_year}-12-31",
        "parameterCd": USGS_PARAM_ELEV_NAVD88,
    }
    resp = requests.get(NWIS_URL, params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.json()

    ts = payload["value"]["timeSeries"]
    if not ts:
        raise RuntimeError("USGS NWIS returned no time series for GSL elevation")
    values = ts[0]["values"][0]["value"]
    df = pd.DataFrame(values)
    df["date"] = pd.to_datetime(df["dateTime"]).dt.tz_localize(None).dt.normalize()
    df["lake_elev_ft_navd88"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["date", "lake_elev_ft_navd88"]].dropna().sort_values("date").reset_index(drop=True)

    df.to_parquet(cache, index=False)
    return df


def load_hypsometry(basin: str = "total") -> pd.DataFrame:
    """USGS elevation-area-volume table; 0.01 ft elevation resolution."""
    path = DATA_RAW / "lake_hypsometry" / HYPSOMETRY_FILES[basin]
    df = pd.read_csv(path)
    df = df.rename(columns={"elev_ft_NAVD88": "elev_ft", "area_km2": "area_km2"})
    return df[["elev_ft", "area_km2"]].sort_values("elev_ft").reset_index(drop=True)


def area_at_elevation(elev_ft: np.ndarray | pd.Series, basin: str = "total") -> np.ndarray:
    hyp = load_hypsometry(basin)
    return np.interp(elev_ft, hyp["elev_ft"], hyp["area_km2"])


def exposed_playa_area(
    elev_ft: np.ndarray | pd.Series,
    basin: str = "total",
    reference_elev_ft: float = REFERENCE_HEALTHY_ELEV_FT_NAVD88,
) -> np.ndarray:
    """Exposed lakebed area (km^2) = area at the reference healthy elevation
    minus area at the current elevation, floored at 0.

    This treats "healthy full pool" as the reference shoreline and measures
    newly-exposed lakebed relative to it -- a modeling choice (see config.py),
    not a claim about a specific historical high-water year.
    """
    hyp = load_hypsometry(basin)
    ref_area = np.interp(reference_elev_ft, hyp["elev_ft"], hyp["area_km2"])
    cur_area = np.interp(elev_ft, hyp["elev_ft"], hyp["area_km2"])
    return np.clip(ref_area - cur_area, a_min=0.0, a_max=None)


if __name__ == "__main__":
    from gsl_model.config import START_YEAR, END_YEAR

    elev = fetch_elevation(START_YEAR, END_YEAR)
    print(elev.shape, elev["date"].min(), elev["date"].max())
    print(elev.tail())

    hyp = load_hypsometry("total")
    print(hyp.head())
    print(hyp.tail())

    exposed = exposed_playa_area(elev["lake_elev_ft_navd88"].values)
    print("exposed playa km2 range:", exposed.min(), exposed.max())
