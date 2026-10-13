import numpy as np
import pandas as pd

from gsl_dust import model, params
from gsl_dust.exposure import build_cell_hypsometry


def _setup(monkeypatch, f=10):
    from gsl_dust import exposure, grid

    class L:
        nx, ny, shape = 2, 2, (2, 2)
    monkeypatch.setattr(exposure, "LATTICE", L)
    monkeypatch.setattr(grid, "LATTICE", L)
    z_ft = np.linspace(4185, 4215, (2 * f) ** 2).reshape(2 * f, 2 * f)
    arm = np.where(np.arange(2 * f)[:, None] < f, 1, 2) * np.ones_like(z_ft, dtype=int)
    hyps = build_cell_hypsometry(z_ft * 0.3048, arm.astype(np.uint8), f)
    n = len(hyps.cells)
    return model.Setup(hyps=hyps, cell_x=np.zeros(n), cell_y=np.zeros(n),
                       dist_valley=np.full(n, 30_000.0), sector_start=np.full(n, 80.0),
                       sector_width=np.full(n, 20.0), c_as=np.full(n, 2e-5))


def _weather_factory(u, v):
    def fake(var, start, end, idx):
        T = len(pd.date_range(start, end, freq="h", inclusive="left"))
        base = {"u10": u, "v10": v, "precip": 0.0, "snow_depth": 0.0}[var]
        return np.full((T, len(idx)), base, dtype=np.float32)
    return fake


def _levels(start, end, h):
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    return pd.DataFrame({"north_arm": h, "south_arm": h}, index=hours)


def test_chain_conserves_mass_and_responds_to_wind_and_level(monkeypatch):
    setup = _setup(monkeypatch)
    start, end = pd.Timestamp("2022-04-01", tz="UTC"), pd.Timestamp("2022-04-03", tz="UTC")
    p = params.central()

    monkeypatch.setattr(model, "_weather", _weather_factory(15.0, 0.0))      # strong, toward valley (east)
    toward = model.run(setup, _levels(start, end, 4192.0), p, start, end)
    pc = toward.per_cell
    closes = pc["S_final"].sum() + pc["I_final"].sum() + pc["arrived_ode"].sum() + pc["removed_ode"].sum()
    assert np.isclose(closes, toward.diagnostics["supplied_kg"], rtol=1e-9)
    assert toward.R[-1] > 0 and np.all(np.diff(toward.R) >= 0)
    assert np.isclose(toward.R_As[-1], p["eta"] * 2e-5 * toward.R[-1])

    monkeypatch.setattr(model, "_weather", _weather_factory(-15.0, 0.0))     # same speed, away
    away = model.run(setup, _levels(start, end, 4192.0), p, start, end)
    assert away.R[-1] == 0 and away.per_cell["emitted"].sum() > 0

    monkeypatch.setattr(model, "_weather", _weather_factory(15.0, 0.0))
    higher = model.run(setup, _levels(start, end, 4205.0), p, start, end)    # higher lake, less exposed
    assert higher.R[-1] < toward.R[-1]

    monkeypatch.setattr(model, "_weather", _weather_factory(2.0, 0.0))      # below threshold
    calm = model.run(setup, _levels(start, end, 4192.0), p, start, end)
    assert calm.per_cell["emitted"].sum() == 0
