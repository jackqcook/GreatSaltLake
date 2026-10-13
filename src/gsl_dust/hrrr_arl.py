"""HRRR weather for HYSPLIT (guide: "Replace transfer with HYSPLIT driven by HRRR").

For each hour, the needed HRRR GRIB2 fields are fetched by byte range from NOAA's
public AWS archive, cropped to a northern-Utah box on HRRR's own 3 km Lambert
grid (no regridding), and written in HYSPLIT's ARL format. Raw GRIB bytes are
never written to disk. Winds stay grid-relative, as ARL expects.

Like hrrr.py, each hour uses the 1-hour forecast from the previous cycle so the
0-1 h precipitation shares the valid time. Upper air comes from the pressure-level
file (25 hPa spacing); surface fields from the surface file."""
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import threading
import time

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
import rasterio
from rasterio.io import MemoryFile

from gsl_dust import arl
from gsl_dust.config import DATA_INTERIM, HRRR_BUCKET

ARL_DIR = DATA_INTERIM / "hysplit_met"

# Crop: NX x NY HRRR points centred here (covers the lakebed, the Salt Lake
# Valley and ~200 km beyond in every direction).
CENTER_LAT, CENTER_LON = 41.0, -112.4
NX = NY = 150
SYNC_LAT, SYNC_LON = 41.00, -112.40

PRESSURES = list(range(925, 499, -25))  # hPa, 18 levels; Utah's surface is ~870 hPa or less
UPPER = {  # ARL name: (HRRR name, scale)
    "HGTS": ("HGT", 1.0), "TEMP": ("TMP", 1.0), "UWND": ("UGRD", 1.0), "VWND": ("VGRD", 1.0),
    "WWND": ("VVEL", 0.01), "SPHU": ("SPFH", 1.0)}
SURFACE = {  # ARL name: (.idx match, scale)
    "PRSS": (":PRES:surface:", 0.01), "SHGT": (":HGT:surface:", 1.0), "T02M": (":TMP:2 m above ground:", 1.0),
    "U10M": (":UGRD:10 m above ground:", 1.0), "V10M": (":VGRD:10 m above ground:", 1.0),
    "RH2M": (":RH:2 m above ground:", 1.0), "PBLH": (":HPBL:surface:", 1.0), "USTR": (":FRICV:surface:", 1.0),
    "SHTF": (":SHTFL:surface:", 1.0), "LTHF": (":LHTFL:surface:", 1.0), "DSWF": (":DSWRF:surface:", 1.0),
    "TPP1": (":APCP:surface:0-1 hour acc", 0.001)}

_local = threading.local()


def _http():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


# Forecast used for each valid hour, in order of preference: the 1-h forecast of
# the previous cycle (as hrrr.py), else the 2-h forecast of the cycle before, else
# the analysis (which has no 0-1 h precipitation; TPP1 is then set to 0).
SOURCES = [(1, 1, "0-1 hour acc"), (2, 2, "1-2 hour acc"), (0, 0, None)]


def _urls(valid, back=1, fhr=1):
    init = valid - timedelta(hours=back)
    base = f"{HRRR_BUCKET}/hrrr.{init:%Y%m%d}/conus/hrrr.t{init:%H}z"
    return f"{base}.wrfsfcf{fhr:02d}.grib2", f"{base}.wrfprsf{fhr:02d}.grib2"


def _get(url, headers=None, timeout=120, tries=6):
    """GET with retries: S3 occasionally drops connections."""
    for k in range(tries):
        try:
            r = _http().get(url, headers=headers, timeout=timeout)
            if r.status_code == 404:
                return r
            r.raise_for_status()
            return r
        except requests.RequestException:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)
            _local.s = requests.Session()


def _ranges(url, matches):
    r = _get(url + ".idx", timeout=60)
    r.raise_for_status()
    lines = r.text.strip().splitlines()
    offs = [int(l.split(":")[1]) for l in lines]
    out = {}
    for key, m in matches.items():
        k = next(i for i, l in enumerate(lines) if m in l)
        out[key] = (url, offs[k], offs[k + 1] - 1 if k + 1 < len(offs) else None)
    return out


class Crop:
    """Pixel window of the crop in GDAL's north-up array, and the ARL grid."""

    def __init__(self, ds):
        to_hrrr = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        inv = ~ds.transform

        def pix(lat, lon):
            col, row = inv * to_hrrr.transform(lon, lat)
            return col - 0.5, row - 0.5  # pixel-centre coordinates

        cc, rc = pix(CENTER_LAT, CENTER_LON)
        self.c0, self.r0 = int(round(cc - NX / 2)), int(round(rc - NY / 2))
        sc, sr = pix(SYNC_LAT, SYNC_LON)
        # ARL x is 1-based from the west; ARL y is 1-based from the south (array is north-up)
        self.grid = arl.LambertGrid(
            tangent_lat=38.5, orient_lon=262.5, grid_km=3.0,
            sync_x=sc - self.c0 + 1.0, sync_y=(self.r0 + NY - 1 - sr) + 1.0,
            sync_lat=SYNC_LAT, sync_lon=SYNC_LON % 360.0, nx=NX, ny=NY)

    def cut(self, arr):
        return arr[self.r0:self.r0 + NY, self.c0:self.c0 + NX][::-1].astype(np.float64)


_crop = None
_crop_lock = threading.Lock()


def _fetch(item):
    url, a, b = item
    rng = f"bytes={a}-{b}" if b is not None else f"bytes={a}-"
    g = _get(url, headers={"Range": rng})
    g.raise_for_status()
    global _crop
    # GDAL converts GRIB temperatures to Celsius unless told not to; ARL wants K.
    with rasterio.Env(GRIB_NORMALIZE_UNITS="NO"), MemoryFile(g.content) as mf, mf.open() as ds:
        arr = ds.read(1)
        with _crop_lock:
            if _crop is None:
                _crop = Crop(ds)
    return _crop.cut(arr), len(g.content)


def fetch_hour(valid, pool):
    """All ARL fields for one valid hour. Returns (levels, fields, bytes_downloaded)."""
    for back, fhr, acc in SOURCES:
        sfc_url, prs_url = _urls(valid, back, fhr)
        if _get(sfc_url + ".idx", timeout=60).status_code == 200 and _get(prs_url + ".idx", timeout=60).status_code == 200:
            break
    else:
        raise RuntimeError(f"no HRRR forecast available for {valid}")
    sfc = {k: m for k, (m, _) in SURFACE.items()}
    if acc is None:
        sfc.pop("TPP1")
    else:
        sfc["TPP1"] = f":APCP:surface:{acc}"
    items = {}
    items.update({("sfc", k): v for k, v in _ranges(sfc_url, sfc).items()})
    upper_matches = {(p, k): f":{h}:{p} mb:" for p in PRESSURES for k, (h, _) in UPPER.items()}
    items.update({("prs",) + k: v for k, v in _ranges(prs_url, upper_matches).items()})
    keys = list(items)
    results = list(pool.map(_fetch, [items[k] for k in keys]))
    nbytes = sum(n for _, n in results)
    fields = {}
    for key, (arr, _) in zip(keys, results):
        if key[0] == "sfc":
            fields[(0, key[1])] = arr * SURFACE[key[1]][1]
        else:
            p, name = key[1], key[2]
            fields[(1 + PRESSURES.index(p), name)] = arr * UPPER[name][1]
    if acc is None:
        fields[(0, "TPP1")] = np.zeros((NY, NX))
    levels = [arl.Level(0.0, list(SURFACE))] + [arl.Level(float(p), list(UPPER)) for p in PRESSURES]
    return levels, fields, nbytes


def write_day(day: pd.Timestamp, workers=16, log=print):
    """One ARL file per UTC day (24 periods). Skips days already written."""
    ARL_DIR.mkdir(parents=True, exist_ok=True)
    path = ARL_DIR / f"hrrr_utah_{day:%Y%m%d}.arl"
    if path.exists():
        return path, 0
    tmp = path.with_suffix(".part")
    total = 0
    with ThreadPoolExecutor(workers) as pool, open(tmp, "wb") as fh:
        for h in range(24):
            valid = day + pd.Timedelta(hours=h)
            levels, fields, n = fetch_hour(valid, pool)
            total += n
            arl.write_period(fh, valid, _crop.grid, levels, fields, forecast=1)
    tmp.rename(path)
    log(f"{day:%Y-%m-%d}: {total / 1e9:.2f} GB fetched, {path.stat().st_size / 1e6:.0f} MB kept")
    return path, total
