"""Exposure X_i(t): the exposed fraction of each lakebed cell, from lake level
(by arm) and terrain. Guide method: geometry; no ML.

Each lattice cell keeps its own hypsometric curve per arm: the count of lakebed
terrain pixels at or below each elevation, on the same 0.01 ft steps as the USGS
elevation-area tables. Exposure is then a table lookup:

    X_i(t) = sum over arms a of [N_i,a - C_i,a(h_a(t))] * pixel_area / cell_area

where N_i,a is the number of lakebed pixels of arm a in cell i, C_i,a(h) the
number at or below level h, and h_a(t) that arm's level (NAVD88). Pixels between
h_a(t) and the lakebed top are exposed lakebed."""
import io
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import rasterio
import requests
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from scipy import ndimage

from gsl_dust import decisions
from gsl_dust.config import (CELL_M, CRS, DATA_INTERIM, DATA_RAW, FT_TO_M, GAUGE_LOCATIONS,
                             GAUGE_NORTH_ARM, GAUGE_SOUTH_ARM, HYPSOMETRY_ITEM,
                             TBDEM_HEIGHT, TBDEM_OVR, TBDEM_TFW, TBDEM_WIDTH)
from gsl_dust.datums import navd88_ft_to_m
from gsl_dust.grid import LATTICE

ANALYSIS_RES_M = 20.0          # terrain pixels used inside each 1 km cell (50 x 50)
BIN_FT = 0.01                  # matches the USGS tables
BIN_LO_FT = 4170.0
ARMS = ("north_arm", "south_system")  # south system = south arm + both bays (decision 2)
NORTH, SOUTH = 1, 2


# ---------------------------------------------------------------- terrain ---

def _base_transform() -> Affine:
    """Full-resolution transform from the world file (pixel-center origin)."""
    a, d, b, e, c, f = [float(v) for v in TBDEM_TFW.read_text().split()]
    return Affine(a, b, c - a / 2 - b / 2, d, e, f - d / 2 - e / 2)


def list_overview_pages():
    """(page index, width, height, pixel size m) for every page of the .ovr."""
    pages, n = [], 0
    while True:
        try:
            with rasterio.open(f"GTIFF_DIR:{n + 1}:{TBDEM_OVR}") as ds:
                pages.append((n + 1, ds.width, ds.height, TBDEM_WIDTH / ds.width * abs(_base_transform().a)))
        except rasterio.errors.RasterioIOError:
            return pages
        n += 1


def load_terrain(max_page_res_m: float = 17.0) -> np.ndarray:
    """Terrain (NAVD88 m) resampled by averaging onto the lattice at
    ANALYSIS_RES_M, read from the finest overview page no finer than needed."""
    cache = DATA_INTERIM / f"terrain_{int(ANALYSIS_RES_M)}m.npy"
    if cache.exists():
        return np.load(cache)
    pages = [p for p in list_overview_pages() if p[3] <= max_page_res_m]
    page, w, h, res = max(pages, key=lambda p: p[3])
    base = _base_transform()
    src_transform = base * Affine.scale(TBDEM_WIDTH / w, TBDEM_HEIGHT / h)
    f = int(round(CELL_M / ANALYSIS_RES_M))
    dst = np.full((LATTICE.ny * f, LATTICE.nx * f), np.nan, dtype=np.float32)
    dst_transform = LATTICE.transform * Affine.scale(1 / f)
    with rasterio.open(f"GTIFF_DIR:{page}:{TBDEM_OVR}") as ds:
        nodata = ds.nodata
        src = ds.read(1).astype(np.float32)
    if nodata is not None:
        src[src == nodata] = np.nan
    src[(src == 0) | (src < -1e30)] = np.nan  # the .ovr pages declare nodata = 0
    # The metadata says meters, but the values are NAVD88 feet (the minimum,
    # 4170.09, matches the 4,170 ft floor of the USGS area tables). Convert once
    # here so everything downstream is NAVD88 meters.
    src *= FT_TO_M
    reproject(src, dst, src_transform=src_transform, src_crs=CRS, src_nodata=np.nan,
              dst_transform=dst_transform, dst_crs=CRS, dst_nodata=np.nan,
              resampling=Resampling.average)
    np.save(cache, dst)
    (DATA_INTERIM / "terrain_meta.json").write_text(json.dumps(
        {"overview_page": page, "page_res_m": res, "analysis_res_m": ANALYSIS_RES_M}))
    return dst


# ------------------------------------------------------------------- arms ---

def _pixel_rc(lat, lon, transform):
    from pyproj import Transformer
    x, y = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(lon, lat)
    col, row = ~transform * (x, y)
    return int(row), int(col)


def label_arms(lakebed: np.ndarray, transform: Affine, barrier) -> np.ndarray:
    """0 = not lakebed, 1 = north arm, 2 = south system. The causeway line is
    burned into the lakebed mask as a barrier; the component holding the north
    gauge is the north arm, the one holding the south gauge the south system,
    and any isolated pocket joins whichever of the two is nearer."""
    cut = rasterize([(barrier.buffer(transform.a * 1.5), 1)], out_shape=lakebed.shape,
                    transform=transform, fill=0, dtype=np.uint8).astype(bool)
    labels, _ = ndimage.label(lakebed & ~cut)
    rn, cn = _pixel_rc(*GAUGE_LOCATIONS[GAUGE_NORTH_ARM], transform)
    rs, cs = _pixel_rc(*GAUGE_LOCATIONS[GAUGE_SOUTH_ARM], transform)
    ln, ls = _nearest_label(labels, rn, cn), _nearest_label(labels, rs, cs)
    if ln == ls:
        raise RuntimeError("Causeway barrier does not separate the arms")
    arm = np.zeros(lakebed.shape, dtype=np.uint8)
    dn = ndimage.distance_transform_edt(labels != ln)
    ds = ndimage.distance_transform_edt(labels != ls)
    arm[lakebed] = np.where(dn[lakebed] <= ds[lakebed], NORTH, SOUTH)
    arm[labels == ln] = NORTH
    arm[labels == ls] = SOUTH
    return arm


def _nearest_label(labels, r, c, search=200):
    win = labels[max(r - search, 0):r + search, max(c - search, 0):c + search]
    if labels[r, c]:
        return labels[r, c]
    vals, counts = np.unique(win[win > 0], return_counts=True)
    return vals[np.argmax(counts)]


# ----------------------------------------------------- per-cell hypsometry ---

@dataclass
class CellHypsometry:
    cells: np.ndarray          # (n,) flat lattice indices of lakebed cells
    n_lakebed: np.ndarray      # (n, 2) lakebed pixel counts per arm
    cum: np.ndarray            # (n, 2, n_bins) pixels at or below each bin's upper edge
    bin_top_ft: np.ndarray     # (n_bins,) NAVD88 ft
    pixel_area_m2: float

    @property
    def lakebed_fraction(self):
        return self.n_lakebed.sum(1) * self.pixel_area_m2 / CELL_M ** 2

    def save(self, path):
        np.savez_compressed(path, cells=self.cells, n_lakebed=self.n_lakebed, cum=self.cum,
                            bin_top_ft=self.bin_top_ft, pixel_area_m2=self.pixel_area_m2)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["cells"], z["n_lakebed"], z["cum"], z["bin_top_ft"], float(z["pixel_area_m2"]))

    def _bin(self, h_ft):
        return np.clip(np.searchsorted(self.bin_top_ft, np.asarray(h_ft), side="right") - 1, -1, len(self.bin_top_ft) - 1)

    def submerged_pixels(self, h_north_ft, h_south_ft):
        """Submerged lakebed pixels per cell: (n,) for scalar levels, (T, n) for
        level series (NAVD88 ft)."""
        scalar = np.ndim(h_north_ft) == 0
        out = 0
        for a, h in enumerate((np.atleast_1d(h_north_ft), np.atleast_1d(h_south_ft))):
            b = self._bin(h)
            vals = self.cum[:, a, np.maximum(b, 0)].T.astype(np.int64)
            vals[b < 0] = 0
            out = out + vals
        return out[0] if scalar else out

    def exposure(self, h_north_ft, h_south_ft):
        """X_i for each cell (exposed lakebed area / cell area)."""
        exposed = self.n_lakebed.sum(1) - self.submerged_pixels(h_north_ft, h_south_ft)
        return exposed * self.pixel_area_m2 / CELL_M ** 2


def build_cell_hypsometry(terrain_m: np.ndarray, arm: np.ndarray, f: int) -> CellHypsometry:
    """terrain on a lattice-aligned raster with f x f pixels per cell."""
    top_ft = decisions.LAKEBED_TOP_FT_NAVD88
    edges = np.round(np.arange(BIN_LO_FT, top_ft + BIN_FT / 2, BIN_FT), 2)
    bin_top = edges[1:]
    z_ft = terrain_m / 0.3048
    lakebed = arm > 0
    ny, nx = LATTICE.shape
    cell_idx = (np.arange(ny * f)[:, None] // f) * nx + (np.arange(nx * f)[None, :] // f)
    cid = cell_idx[lakebed]
    zb = np.clip(z_ft[lakebed], BIN_LO_FT, top_ft)
    ab = arm[lakebed].astype(np.int64) - 1
    b = np.clip(np.ceil((zb - BIN_LO_FT) / BIN_FT - 1e-9).astype(np.int64) - 1, 0, len(bin_top) - 1)
    cells = np.unique(cid)
    pos = np.searchsorted(cells, cid)
    counts = np.zeros((len(cells), 2, len(bin_top)), dtype=np.uint32)
    np.add.at(counts, (pos, ab, b), 1)
    cum = np.cumsum(counts, axis=2).astype(np.uint16)
    n_lakebed = counts.sum(axis=2).astype(np.uint16)
    return CellHypsometry(cells, n_lakebed, cum, bin_top, (CELL_M / f) ** 2)


# -------------------------------------------------- USGS validation tables ---

USGS_TABLES = {
    "total": "Great_Salt_Lake_2023_ElevAreaVolume_total.csv",
    "south_arm": "Great_Salt_Lake_2023_ElevAreaVolume_south_arm.csv",
    "north_arm": "Great_Salt_Lake_2023_ElevAreaVolume_north_arm.csv",
    "farmington_bay": "Great_Salt_Lake_2023_ElevAreaVolume_farmington_bay.csv",
    "bear_river_bay": "Great_Salt_Lake_2023_ElevAreaVolume_bear_river_bay.csv",
}


def usgs_area_tables() -> pd.DataFrame:
    """Wet area (km2) by basin at each 0.01 ft NAVD88 level."""
    d = DATA_RAW / "usgs_hypsometry"
    d.mkdir(parents=True, exist_ok=True)
    urls = None
    frames = {}
    for basin, name in USGS_TABLES.items():
        p = d / name
        if not p.exists():
            if urls is None:
                item = requests.get(f"https://www.sciencebase.gov/catalog/item/{HYPSOMETRY_ITEM}?format=json", timeout=60).json()
                urls = {f["name"]: f["url"] for f in item["files"]}
            p.write_bytes(requests.get(urls[name], timeout=120).content)
        t = pd.read_csv(p)
        frames[basin] = t.set_index(t["elev_ft_NAVD88"].round(2))["area_km2"]
    return pd.DataFrame(frames)


def validate_against_usgs(h: CellHypsometry, levels_ft=None) -> pd.DataFrame:
    """Compare modeled wet area per arm with the USGS tables."""
    usgs = usgs_area_tables()
    if levels_ft is None:
        levels_ft = np.round(np.arange(4175.0, 4215.0 + 1e-9, 0.5), 2)
    rows = []
    for lv in levels_ft:
        b = h._bin(lv)
        wet = h.cum[:, :, b].sum(axis=0) * h.pixel_area_m2 / 1e6
        usgs_south = usgs.loc[lv, ["south_arm", "farmington_bay", "bear_river_bay"]].sum()
        rows.append({"level_ft_navd88": lv,
                     "model_north_km2": wet[0], "usgs_north_km2": usgs.loc[lv, "north_arm"],
                     "model_south_km2": wet[1], "usgs_south_km2": usgs_south,
                     "model_total_km2": wet.sum(), "usgs_total_km2": usgs.loc[lv, "total"]})
    df = pd.DataFrame(rows)
    for k in ("north", "south", "total"):
        df[f"rel_err_{k}"] = (df[f"model_{k}_km2"] - df[f"usgs_{k}_km2"]) / df[f"usgs_{k}_km2"].replace(0, np.nan)
    return df
