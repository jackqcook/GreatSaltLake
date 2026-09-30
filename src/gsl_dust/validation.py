"""Stage 4 inputs: EPA AQS hourly PM10 at Wasatch Front monitors, dust-episode
detection, and the held-out design (guide: held-out events and locations).

The held-out design is written once to validation/heldout_design.json and
committed, so it is fixed before any model output is compared with a monitor.
Comparisons themselves use HYSPLIT concentrations (Stage 3.2), never R."""
import io
import json
import zipfile

import numpy as np
import pandas as pd
import requests

from gsl_dust.config import DATA_RAW, ROOT

AQS_HOURLY = "https://aqs.epa.gov/aqsweb/airdata/hourly_{param}_{year}.zip"
PM10 = "81102"
UTAH = "49"
WASATCH_COUNTIES = {"003": "Box Elder", "011": "Davis", "035": "Salt Lake", "045": "Tooele",
                    "049": "Utah", "057": "Weber"}
DESIGN_PATH = ROOT / "validation" / "heldout_design.json"

EVENT_HOURLY_UG = 150.0      # an hour at any monitor at or above this starts an episode
EVENT_GAP_HOURS = 6          # exceedances closer than this merge into one episode
EVENT_MIN_MONITORS = 2       # regional episodes only; single-monitor spikes are local sources


def fetch_pm10(year: int) -> pd.DataFrame:
    cache = DATA_RAW / "aqs" / f"pm10_{year}_wasatch.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    r = requests.get(AQS_HOURLY.format(param=PM10, year=year), timeout=300)
    r.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    cols = ["State Code", "County Code", "Site Num", "POC", "Latitude", "Longitude", "Date GMT", "Time GMT",
            "Sample Measurement", "Method Code"]
    with zf.open(zf.namelist()[0]) as f:
        df = pd.read_csv(f, usecols=cols, dtype={"State Code": str, "County Code": str, "Site Num": str}, low_memory=False)
    df = df[(df["State Code"] == UTAH) & df["County Code"].isin(WASATCH_COUNTIES)]
    df["site_id"] = df["State Code"] + "-" + df["County Code"] + "-" + df["Site Num"]
    df["time"] = pd.to_datetime(df["Date GMT"] + " " + df["Time GMT"], utc=True)
    df = df.rename(columns={"Sample Measurement": "pm10", "Latitude": "lat", "Longitude": "lon"})
    df = (df.groupby(["site_id", "time"], as_index=False)
            .agg(pm10=("pm10", "mean"), lat=("lat", "first"), lon=("lon", "first")))
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def detect_episodes(pm: pd.DataFrame) -> pd.DataFrame:
    """Network episodes: any monitor at or above EVENT_HOURLY_UG, merged across gaps."""
    hot = pm[pm["pm10"] >= EVENT_HOURLY_UG].sort_values("time")
    if hot.empty:
        return pd.DataFrame(columns=["start", "end", "peak_pm10", "n_monitors"])
    times = hot["time"].drop_duplicates().sort_values().reset_index(drop=True)
    group = (times.diff() > pd.Timedelta(hours=EVENT_GAP_HOURS)).cumsum()
    rows = []
    for g, t in times.groupby(group):
        w = hot[(hot["time"] >= t.min()) & (hot["time"] <= t.max())]
        rows.append({"start": t.min(), "end": t.max(), "peak_pm10": float(w["pm10"].max()),
                     "n_monitors": int(w["site_id"].nunique())})
    ep = pd.DataFrame(rows)
    return ep[ep["n_monitors"] >= EVENT_MIN_MONITORS].reset_index(drop=True)


def make_design(pm: pd.DataFrame, episodes: pd.DataFrame, frac=1 / 3, seed=2026) -> dict:
    """Randomly (seeded) hold out a third of episodes and a third of monitors."""
    rng = np.random.default_rng(seed)
    sites = sorted(pm["site_id"].unique())
    n_sites = max(1, int(round(len(sites) * frac)))
    held_sites = sorted(rng.choice(sites, n_sites, replace=False).tolist())
    ep_ids = list(range(len(episodes)))
    n_ep = max(1, int(round(len(ep_ids) * frac)))
    held_eps = sorted(int(i) for i in rng.choice(ep_ids, n_ep, replace=False))
    return {
        "note": "Fixed before any model-monitor comparison. Do not edit after comparisons begin.",
        "seed": seed, "event_rule": {"hourly_ug_m3": EVENT_HOURLY_UG, "merge_gap_hours": EVENT_GAP_HOURS,
                                     "min_monitors": EVENT_MIN_MONITORS},
        "monitors": sites, "heldout_monitors": held_sites,
        "episodes": [{"id": i, "start": e.start.isoformat(), "end": e.end.isoformat(),
                      "peak_pm10": e.peak_pm10, "n_monitors": e.n_monitors}
                     for i, e in enumerate(episodes.itertuples())],
        "heldout_episode_ids": held_eps,
    }


def write_design_once(design: dict):
    if DESIGN_PATH.exists():
        raise FileExistsError(f"{DESIGN_PATH} already fixed; not overwriting")
    DESIGN_PATH.parent.mkdir(parents=True, exist_ok=True)
    DESIGN_PATH.write_text(json.dumps(design, indent=1))
