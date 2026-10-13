"""The 1 km lattice (NAD83 / UTM 12N). Lakebed cells i are the lattice cells that
contain lakebed terrain (exposure.py); forcing is stored on the whole lattice so
the lakebed subset can change without re-downloading anything."""
from dataclasses import dataclass

import numpy as np
from affine import Affine
from pyproj import Transformer

from gsl_dust.config import CELL_M, CRS

# Lakebed extent from USGS terrain metadata (lon -113.07..-111.92, lat
# 40.66..41.70) projected to UTM 12N, snapped to 1 km, plus a 5 km margin.
LATTICE_X0, LATTICE_X1 = 320_000.0, 430_000.0
LATTICE_Y0, LATTICE_Y1 = 4_496_000.0, 4_624_000.0


@dataclass(frozen=True)
class Lattice:
    x0: float = LATTICE_X0
    y1: float = LATTICE_Y1
    nx: int = int((LATTICE_X1 - LATTICE_X0) / CELL_M)
    ny: int = int((LATTICE_Y1 - LATTICE_Y0) / CELL_M)
    cell: float = CELL_M

    @property
    def transform(self) -> Affine:
        return Affine(self.cell, 0.0, self.x0, 0.0, -self.cell, self.y1)

    @property
    def shape(self):
        return (self.ny, self.nx)

    def centers(self):
        cols = self.x0 + (np.arange(self.nx) + 0.5) * self.cell
        rows = self.y1 - (np.arange(self.ny) + 0.5) * self.cell
        return np.meshgrid(cols, rows)

    def centers_lonlat(self):
        x, y = self.centers()
        to_ll = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
        return to_ll.transform(x, y)


LATTICE = Lattice()
