import numpy as np

from gsl_dust import exposure
from gsl_dust.exposure import CellHypsometry, build_cell_hypsometry


def _synthetic(monkeypatch, f=10):
    """2 x 2 cells of 10 x 10 pixels: a tilted plane from 4180 to 4220 ft, the
    top two rows of cells labelled north arm, the rest south."""
    from gsl_dust import grid

    class L:
        nx, ny = 2, 2
        shape = (2, 2)
    monkeypatch.setattr(exposure, "LATTICE", L)
    monkeypatch.setattr(grid, "LATTICE", L)
    ny, nx = 2 * f, 2 * f
    z_ft = np.linspace(4180, 4220, ny * nx).reshape(ny, nx)
    arm = np.where(z_ft <= 4215.0, np.where(np.arange(ny)[:, None] < f, 1, 2), 0).astype(np.uint8)
    return z_ft * 0.3048, arm, z_ft


def test_exposure_matches_direct_count(monkeypatch):
    f = 10
    terrain_m, arm, z_ft = _synthetic(monkeypatch, f)
    h = build_cell_hypsometry(terrain_m, arm, f)
    pix = h.pixel_area_m2 / exposure.CELL_M ** 2
    for hn, hs in [(4185.0, 4190.0), (4200.0, 4200.0), (4170.0, 4214.99)]:
        x = h.exposure(hn, hs)
        level = np.where(arm == 1, hn, hs)
        expected_pixels = ((arm > 0) & (z_ft > level)).reshape(2, f, 2, f).sum(axis=(1, 3)).ravel()
        got = np.zeros(4)
        got[h.cells] = x
        assert np.allclose(got, expected_pixels * pix, atol=2 * pix), (hn, hs)


def test_exposure_series_and_bounds(monkeypatch):
    f = 10
    terrain_m, arm, _ = _synthetic(monkeypatch, f)
    h = build_cell_hypsometry(terrain_m, arm, f)
    series = h.exposure(np.array([4180.0, 4200.0, 4215.0]), np.array([4180.0, 4200.0, 4215.0]))
    assert series.shape == (3, len(h.cells))
    assert np.all(np.diff(series.sum(1)) <= 0)          # higher lake -> less exposed
    assert np.allclose(series[-1], 0)                    # lake at lakebed top -> nothing exposed
    assert np.all(series[0] <= h.lakebed_fraction + 1e-12)


def test_round_trip(tmp_path, monkeypatch):
    terrain_m, arm, _ = _synthetic(monkeypatch)
    h = build_cell_hypsometry(terrain_m, arm, 10)
    h.save(tmp_path / "h.npz")
    h2 = CellHypsometry.load(tmp_path / "h.npz")
    assert np.array_equal(h.exposure(4190.0, 4195.0), h2.exposure(4190.0, 4195.0))
