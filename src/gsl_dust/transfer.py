"""Stage 3.1 first-order transfer rates for the ODE prototype.

Documented physical basis (these are replaced by HYSPLIT in Stage 3.2, and are
never tuned to PM10, since R is arrival mass, not concentration):

- Airborne dust leaves cell i's well-mixed compartment by advection at rate
  U(t) / d_i, where d_i is the distance from the cell to the valley boundary.
- If the wind blows toward the valley (its downwind bearing falls inside the
  angular sector the valley occupies as seen from cell i), that advective loss
  is kV_i; otherwise it is transport elsewhere and joins kL_i.
- Dry deposition adds v_dep / mix_depth to kL_i in all hours."""
import numpy as np
from pyproj import Geod, Transformer
from shapely.geometry import Point

from gsl_dust.config import CRS

_geod = Geod(ellps="GRS80")


def valley_geometry(cell_x, cell_y, valley_poly_utm, n_boundary=720):
    """Per-cell distance to the valley (m) and the valley's sector as seen from the
    cell: (start bearing deg, width deg), true-north bearings clockwise."""
    boundary = valley_poly_utm.exterior if valley_poly_utm.geom_type == "Polygon" else \
        max(valley_poly_utm.geoms, key=lambda g: g.area).exterior
    pts = [boundary.interpolate(f, normalized=True) for f in np.linspace(0, 1, n_boundary, endpoint=False)]
    to_ll = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    blon, blat = to_ll.transform([p.x for p in pts], [p.y for p in pts])
    clon, clat = to_ll.transform(cell_x, cell_y)
    dist = np.array([valley_poly_utm.distance(Point(x, y)) for x, y in zip(cell_x, cell_y)])
    start = np.empty(len(cell_x))
    width = np.empty(len(cell_x))
    for i in range(len(cell_x)):
        az, _, _ = _geod.inv(np.full(len(blon), clon[i]), np.full(len(blat), clat[i]), blon, blat)
        az = np.sort(np.mod(az, 360.0))
        gaps = np.diff(np.concatenate([az, [az[0] + 360.0]]))
        j = np.argmax(gaps)  # the valley occupies the complement of the largest empty gap
        start[i] = az[(j + 1) % len(az)]
        width[i] = 360.0 - gaps[j]
    return dist, start, width


def rates(u10, v10, dist_m, sector_start, sector_width, v_dep, mix_depth):
    """kV, kL (1/s) for hourly earth-relative winds of shape (T, n)."""
    speed = np.hypot(u10, v10)
    downwind = np.mod(np.degrees(np.arctan2(u10, v10)), 360.0)
    toward = np.mod(downwind - sector_start, 360.0) <= sector_width
    adv = speed / np.maximum(dist_m, 500.0)
    kV = np.where(toward, adv, 0.0)
    kL = np.where(toward, 0.0, adv) + v_dep / mix_depth
    return kV, kL
