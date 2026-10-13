import numpy as np
from shapely.geometry import box

from gsl_dust import erodibility, transfer


def test_sector_and_rates_for_cell_west_of_valley():
    valley = box(410_000, 4_500_000, 430_000, 4_520_000)      # 20 km square
    cell_x, cell_y = np.array([380_000.0]), np.array([4_510_000.0])  # 30 km west, centered
    dist, start, width = transfer.valley_geometry(cell_x, cell_y, valley)
    assert np.isclose(dist[0], 30_000, atol=1)
    assert 30 < width[0] < 40          # 2*atan(10/30) ~ 36.9 deg as seen from the cell
    u = np.array([[5.0], [-5.0], [0.0]])  # blowing east (toward), west (away), calm-north
    v = np.array([[0.0], [0.0], [5.0]])
    kV, kL = transfer.rates(u, v, dist, start, width, v_dep=0.01, mix_depth=1000.0)
    assert np.isclose(kV[0, 0], 5.0 / 30_000) and np.isclose(kL[0, 0], 0.01 / 1000)
    assert kV[1, 0] == 0 and np.isclose(kL[1, 0], 5.0 / 30_000 + 1e-5)
    assert kV[2, 0] == 0


def test_rain_memory_and_phi():
    P = np.zeros((48, 1)); P[0] = 10.0
    w, last = erodibility.rain_memory(P, tau_hours=24.0)
    assert np.isclose(w[24, 0], 10 * np.exp(-1.0))
    phi = erodibility.physics_phi(0.1, w, wet_depth_mm=2.0, snow_depth_m=np.zeros_like(w),
                                  snow_m=0.01, fringe_frac=np.ones_like(w))
    wet_hours = int((phi == 0).sum())
    assert wet_hours == int(np.ceil(24 * np.log(10 / 2.0)))   # dries below 2 mm after ~39 h
    assert np.allclose(phi[phi > 0], 0.1)
