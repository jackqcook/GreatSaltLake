"""Hourly HRRR inputs on the 1 km lattice (guide: "NOAA HRRR winds and mixing").

Each hour uses the 1-hour forecast (f01) from the previous hour's cycle, so
every field, including the 0-1 h precipitation accumulation, shares one valid
time. Single GRIB2 fields are fetched by byte range from NOAA's public AWS
archive using the published .idx inventories, sampled bilinearly at lattice cell
centers, and the raw bytes discarded."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
from rasterio.io import MemoryFile
from scipy.ndimage import map_coordinates

from gsl_dust.config import CRS, DATA_INTERIM, HRRR_BUCKET
from gsl_dust.grid import LATTICE

HRRR_DIR = DATA_INTERIM / "hrrr"

# output name -> (.idx match, units)
FIELDS = {
    "u_star": (":FRICV:surface:", "m/s"),
    "precip": (":APCP:surface:0-1 hour acc", "mm/h"),
    "moisture_avail": (":MSTAV:0 m underground:", "%"),
    "snow_depth": (":SNOD:surface:", "m"),
    "u10_grid": (":UGRD:10 m above ground:", "m/s"),
    "v10_grid": (":VGRD:10 m above ground:", "m/s"),
    "rh2m": (":RH:2 m above ground:", "%"),
}
OUTPUTS = ["u_star", "precip", "moisture_avail", "snow_depth", "u10", "v10", "rh2m"]

# HRRR Lambert conformal parameters (tangent at 38.5N, orientation 97.5W).
_CONE = np.sin(np.radians(38.5))
_LOV = -97.5

_session = threading.local()


def _http():
    if not hasattr(_session, "s"):
        _session.s = requests.Session()
    return _session.s


def file_url(valid: pd.Timestamp) -> str:
    init = valid - timedelta(hours=1)
    return f"{HRRR_BUCKET}/hrrr.{init:%Y%m%d}/conus/hrrr.t{init:%H}z.wrfsfcf01.grib2"


class _Sampler:
    """Maps lattice cell centers to fractional HRRR (row, col) once, from the
    projection GDAL reports for the GRIB field itself."""

    def __init__(self, ds):
        lon, lat = LATTICE.centers_lonlat()
        to_hrrr = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        hx, hy = to_hrrr.transform(lon, lat)
        col, row = ~ds.transform * (hx, hy)
        self.coords = np.stack([row - 0.5, col - 0.5])
        alpha = np.radians(_CONE * (lon - _LOV))
        self.cos, self.sin = np.cos(alpha), np.sin(alpha)

    def sample(self, arr):
        return map_coordinates(arr.astype(np.float64), self.coords, order=1, mode="nearest").astype(np.float32)


_sampler = None
_sampler_lock = threading.Lock()


def _read_field(content):
    global _sampler
    with MemoryFile(content) as mf, mf.open() as ds:
        arr = ds.read(1)
        with _sampler_lock:
            if _sampler is None:
                _sampler = _Sampler(ds)
    return _sampler.sample(arr)


def fetch_hour(valid: pd.Timestamp) -> dict:
    url = file_url(valid)
    r = _http().get(url + ".idx", timeout=60)
    r.raise_for_status()
    lines = r.text.strip().splitlines()
    offsets = [int(l.split(":")[1]) for l in lines]
    out = {}
    for name, (match, _) in FIELDS.items():
        k = next(i for i, l in enumerate(lines) if match in l)
        end = offsets[k + 1] - 1 if k + 1 < len(offsets) else ""
        g = _http().get(url, headers={"Range": f"bytes={offsets[k]}-{end}"}, timeout=120)
        g.raise_for_status()
        out[name] = _read_field(g.content)
    # grid-relative -> earth-relative winds
    ug, vg = out.pop("u10_grid"), out.pop("v10_grid")
    out["u10"] = (_sampler.cos * ug + _sampler.sin * vg).astype(np.float32)
    out["v10"] = (-_sampler.sin * ug + _sampler.cos * vg).astype(np.float32)
    return out


def _month_path(var, month):
    return HRRR_DIR / var / f"{month}.npy"


def _open_month(var, month, n_hours):
    p = _month_path(var, month)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        return np.load(p, mmap_mode="r+")
    arr = np.lib.format.open_memmap(p, mode="w+", dtype=np.float32, shape=(n_hours, *LATTICE.shape))
    arr[:] = np.nan
    return arr


def download_month(month: str, workers: int = 12, log=print):
    """Fill every hour of one UTC month. Resumable via a per-month ledger."""
    start = pd.Timestamp(month + "-01", tz="UTC")
    hours = pd.date_range(start, start + pd.offsets.MonthBegin(1), freq="h", inclusive="left")
    ledger_path = HRRR_DIR / "ledger" / f"{month}.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    done = set(json.loads(ledger_path.read_text())) if ledger_path.exists() else set()
    failed = {}
    arrays = {v: _open_month(v, month, len(hours)) for v in OUTPUTS}
    todo = [(i, h) for i, h in enumerate(hours) if h.isoformat() not in done]
    lock = threading.Lock()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch_hour, h): (i, h) for i, h in todo}
        for n, f in enumerate(as_completed(futs), 1):
            i, h = futs[f]
            try:
                fields = f.result()
            except Exception as e:  # missing hour in the archive, network error
                failed[h.isoformat()] = repr(e)[:200]
                continue
            with lock:
                for v in OUTPUTS:
                    arrays[v][i] = fields[v]
                done.add(h.isoformat())
            if n % 100 == 0:
                log(f"{month}: {n}/{len(todo)} hours")
    for a in arrays.values():
        a.flush()
    ledger_path.write_text(json.dumps(sorted(done)))
    (HRRR_DIR / "ledger" / f"{month}_failed.json").write_text(json.dumps(failed, indent=1))
    log(f"{month}: done {len(done)}/{len(hours)}, failed {len(failed)}")
    return failed


def load(var: str, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    """Hourly field (time, ny, nx) for [start, end)."""
    months = pd.period_range(start.tz_convert(None), (end - pd.Timedelta(hours=1)).tz_convert(None), freq="M")
    parts = []
    for m in months:
        arr = np.load(_month_path(var, str(m)), mmap_mode="r")
        mstart = pd.Timestamp(str(m) + "-01", tz="UTC")
        i0 = max(0, int((start - mstart) / pd.Timedelta(hours=1)))
        i1 = min(arr.shape[0], int((end - mstart) / pd.Timedelta(hours=1)))
        parts.append(arr[i0:i1])
    return np.concatenate(parts)
