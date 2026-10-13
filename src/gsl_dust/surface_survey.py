"""Stage 2.1 erodibility input from field observations of crust and texture.

Perry, Crosman & Hoch (2019), "Results of the Great Salt Lake Dust Plume Study
(2016-2018)", Appendix C: surface condition at ~5,250 lakebed points (vegetation,
crust type, visible fines when disturbed). Their dust "hot spot" is a point with
no vegetation (NV), visible fines (VF), and no crust or an erodible shallow crust
(NC/ESC); 8.5% of their points. These are the guide's crust and texture inputs
for physics phi ("otherwise physics"; no PI-SWERL labels exist for the ML branch).

Per lakebed cell, the erodible surface fraction is the Gaussian-kernel-weighted
hot-spot share of nearby points, shrunk toward the lakewide share where points are
sparse. The table was parsed from the PDF by character position; category counts
reproduce the report's within 5%, the hot-spot count is 18% below the report's
(366 vs 446), which phi_scale covers."""
import numpy as np
import pandas as pd
from pyproj import Transformer

from gsl_dust.config import CRS, DATA_RAW

SURVEY_CSV = DATA_RAW / "fieldflux" / "perry2019_appendixC_surface.csv"
REPORT_HOTSPOT_SHARE = 0.085  # report section 2.6: 446 of 5246 points


def load():
    df = pd.read_csv(SURVEY_CSV)
    df["hotspot"] = ((df.NV == 1) & (df.VF == 1) & ((df.NC == 1) | (df.ESC == 1))).astype(float)
    x, y = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(df.lon.values, df.lat.values)
    return df.assign(x=x, y=y)


def erodible_fraction(cell_x, cell_y, bandwidth_m=5_000.0, prior_weight=5.0):
    """Per-cell share of surface that is a dust hot spot (0-1)."""
    df = load()
    prior = df.hotspot.mean()
    out = np.empty(len(cell_x))
    for s in range(0, len(cell_x), 500):
        dx = cell_x[s:s + 500, None] - df.x.values[None]
        dy = cell_y[s:s + 500, None] - df.y.values[None]
        w = np.exp(-0.5 * (dx ** 2 + dy ** 2) / bandwidth_m ** 2)
        out[s:s + 500] = (w @ df.hotspot.values + prior_weight * prior) / (w.sum(1) + prior_weight)
    return out
