"""Lake levels by arm (guide: "use lake levels by arm"), converted from the
gauges' NGVD29 feet to the terrain's NAVD88 meters."""
import pandas as pd
import requests

from gsl_dust import decisions
from gsl_dust.config import (DATA_RAW, GAUGE_NORTH_ARM, GAUGE_SOUTH_ARM,
                             LAKE_LEVEL_PARAM)
from gsl_dust.datums import ngvd29_ft_to_navd88_m

NWIS_DV = "https://waterservices.usgs.gov/nwis/dv/"
LOCAL_TZ = "America/Denver"

ARM_GAUGE = {
    "north_arm": GAUGE_NORTH_ARM,
    "south_arm": GAUGE_SOUTH_ARM,
    "farmington_bay": GAUGE_SOUTH_ARM if decisions.BAYS_USE_GAUGE == "south_arm" else None,
    "bear_river_bay": GAUGE_SOUTH_ARM if decisions.BAYS_USE_GAUGE == "south_arm" else None,
}


def fetch_daily(site: str, start: str, end: str) -> pd.Series:
    """Daily mean lake level, NGVD29 feet, indexed by local calendar date."""
    cache = DATA_RAW / "gauges" / f"{site}_{start}_{end}.csv"
    if cache.exists():
        s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
        return s.rename(site)
    r = requests.get(NWIS_DV, timeout=60, params={
        "sites": site, "parameterCd": LAKE_LEVEL_PARAM, "startDT": start, "endDT": end, "format": "json"})
    r.raise_for_status()
    series = r.json()["value"]["timeSeries"]
    if not series:
        raise RuntimeError(f"No lake-level data for {site}")
    name = series[0]["variable"]["variableName"]
    if "NGVD 1929" not in name:
        raise RuntimeError(f"Unexpected datum for {site}: {name}")
    vals = series[0]["values"][0]["value"]
    s = pd.Series({pd.Timestamp(v["dateTime"][:10]): float(v["value"]) for v in vals}, name=site).sort_index()
    cache.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(cache)
    return s


def hourly_levels_navd88_m(hours_utc: pd.DatetimeIndex) -> pd.DataFrame:
    """Level for each arm at each UTC hour, NAVD88 meters. Each hour takes its
    local day's daily mean; missing days are linearly interpolated and flagged."""
    local_days = hours_utc.tz_convert(LOCAL_TZ).tz_localize(None).normalize()
    start = (local_days.min() - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    end = (local_days.max() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    out, flags = {}, {}
    for site in sorted(set(ARM_GAUGE.values())):
        daily = fetch_daily(site, start, end)
        full = daily.reindex(pd.date_range(start, end, freq="D"))
        flags[site] = full.isna()
        full = full.interpolate(limit_area="inside")
        out[site] = ngvd29_ft_to_navd88_m(full.reindex(local_days).to_numpy())
        flags[site] = flags[site].reindex(local_days).to_numpy()
    df = pd.DataFrame({arm: out[g] for arm, g in ARM_GAUGE.items()}, index=hours_utc)
    df.attrs["interpolated"] = {g: int(f.sum()) for g, f in flags.items()}
    return df
