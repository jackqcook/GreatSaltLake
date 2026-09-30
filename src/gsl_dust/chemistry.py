"""Stage 2.4 arsenic: c_As,i by GP/kriging and eta measured-or-bounded (guide).

Inputs are the USGS Great Salt Lake dust and sediment releases (2018-19 and
2022). Only sediment samples on the lakebed (terrain at or below the lakebed top)
are kriged. Each site contributes the mean of its finest available size
fraction (<63 um preferred); samples are the <63 um or coarser fractions, never
PM10-matched, which is why eta stays bounded."""
import io
import json
import urllib.request

import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from gsl_dust.config import CRS, DATA_RAW

RELEASES = {
    "usgs_2018_19": ("62ebed2ad34eacf539724ce2", "SLVDust_DataRelease.csv"),
    "usgs_2022": ("64a6d315d34ef77fcb0639cd", "NUtahDust_DataRelease.csv"),
}
PPM_TO_KG_PER_KG = 1e-6


def load_samples() -> pd.DataFrame:
    """All sediment samples with arsenic, from both releases."""
    d = DATA_RAW / "chemistry"
    d.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, (item, fname) in RELEASES.items():
        p = d / fname
        if not p.exists():
            it = json.load(urllib.request.urlopen(f"https://www.sciencebase.gov/catalog/item/{item}?format=json", timeout=60))
            url = next(f["url"] for f in it["files"] if f["name"] == fname)
            p.write_bytes(urllib.request.urlopen(url, timeout=120).read())
        df = pd.read_csv(p, encoding="utf-8-sig", encoding_errors="replace", low_memory=False)
        df.columns = [c.strip().lstrip("﻿").replace("ï»¿", "") for c in df.columns]
        df = df[(df["Sample_Type"] == "sediments") & df["As"].notna()].copy()
        df["release"] = name
        num = pd.to_numeric(df["Size_Max"], errors="coerce")
        df["Size_Max"] = np.where(num.notna(), num.fillna(0).astype(int).astype(str), df["Size_Max"].astype(str))
        frames.append(df[["release", "Site_Name", "Latitude", "Longitude", "Size_Min", "Size_Max", "As"]])
    out = pd.concat(frames, ignore_index=True)
    out["As"] = pd.to_numeric(out["As"], errors="coerce")
    return out.dropna(subset=["As"])


def site_values(samples: pd.DataFrame) -> pd.DataFrame:
    """One arsenic value (ppm) per site: mean of the finest available fraction."""
    s = samples.copy()
    s["fraction"] = s["Size_Min"].astype(float).astype(int).astype(str) + "-" + s["Size_Max"]
    rank = {"0-63": 0, "0-bulk": 1, "0-500": 2}
    s["rank"] = s["fraction"].map(rank).fillna(9)
    s["site"] = s["Site_Name"].str.strip() + " @" + s["Latitude"].round(3).astype(str) + "," + s["Longitude"].round(3).astype(str)
    best = s[s["rank"] == s.groupby("site")["rank"].transform("min")]
    return best.groupby("site").agg(latitude=("Latitude", "mean"), longitude=("Longitude", "mean"),
                                    as_ppm=("As", "mean"), n=("As", "size"), fraction=("fraction", "first"),
                                    release=("release", "first")).reset_index()


def enrichment_evidence(samples: pd.DataFrame) -> pd.DataFrame:
    """Per site: arsenic in the <63 um fraction relative to the coarser fractions
    (63-500 um) measured at the same site. Motivates eta's bounds."""
    s = samples.copy()
    s["lo"] = s["Size_Min"].astype(float)
    fine = s[(s["lo"] == 0) & (s["Size_Max"] == "63")].groupby("Site_Name")["As"].mean()
    coarse = s[s["lo"] >= 63].groupby("Site_Name")["As"].mean()
    both = pd.concat([fine.rename("fine_ppm"), coarse.rename("coarse_ppm")], axis=1).dropna()
    both["ratio"] = both["fine_ppm"] / both["coarse_ppm"]
    return both


def krige(sites: pd.DataFrame, x_cells: np.ndarray, y_cells: np.ndarray, seed=0):
    """GP (kriging) of log arsenic over the lakebed. Returns median and 5th/95th
    percentile c_As (kg/kg) per cell, plus the fitted kernel."""
    to_utm = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    sx, sy = to_utm.transform(sites["longitude"].to_numpy(), sites["latitude"].to_numpy())
    Xs = np.column_stack([sx, sy]) / 1000.0
    y = np.log(sites["as_ppm"].to_numpy())
    kernel = ConstantKernel(0.1, (1e-3, 10.0)) * Matern(length_scale=30.0, length_scale_bounds=(5.0, 300.0), nu=0.5) \
        + WhiteKernel(0.02, (1e-4, 1.0))
    gp = GaussianProcessRegressor(kernel, normalize_y=False, n_restarts_optimizer=10, random_state=seed)
    mu = y.mean()
    gp.fit(Xs, y - mu)
    m, sd = gp.predict(np.column_stack([x_cells, y_cells]) / 1000.0, return_std=True)
    m = m + mu
    to_kg = lambda v: np.exp(v) * PPM_TO_KG_PER_KG
    return to_kg(m), to_kg(m - 1.645 * sd), to_kg(m + 1.645 * sd), gp.kernel_
