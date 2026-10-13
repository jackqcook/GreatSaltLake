"""Stage 3.2 plumbing: ARL packing, HYSPLIT superposition, exposed-only replenishment."""
import io

import numpy as np
import pandas as pd

from gsl_dust import arl, hysplit_transport


def test_arl_pack_roundtrip_within_half_step():
    rng = np.random.default_rng(1)
    for f in (rng.normal(0, 3, (40, 60)).cumsum(1), np.zeros((5, 7)), 850 + rng.normal(0, 1e-3, (9, 11))):
        data, nexp, prec, var1, ksum = arl.pack(f)
        back = arl.unpack(data, f.shape[1], f.shape[0], nexp, var1)
        assert np.abs(back - f).max() <= 2.0 ** nexp / 256 + 1e-12
        assert 0 <= ksum <= 255


def test_arl_period_layout(tmp_path):
    g = arl.LambertGrid(38.5, 262.5, 3.0, 10.5, 12.25, 41.0, 247.6, nx=20, ny=15)
    levels = [arl.Level(0.0, ["PRSS", "SHGT"]), arl.Level(900.0, ["TEMP"])]
    f = {(0, "PRSS"): np.full((15, 20), 850.0), (0, "SHGT"): np.arange(300.0).reshape(15, 20),
         (1, "TEMP"): np.full((15, 20), 280.0)}
    p = tmp_path / "x.arl"
    with open(p, "wb") as fh:
        arl.write_period(fh, pd.Timestamp("2022-05-03 04:00"), g, levels, f, forecast=1)
    raw = p.read_bytes()
    assert len(raw) == 4 * (50 + 20 * 15)          # index + 3 records, all the same length
    idx = raw[50:50 + 108].decode()
    assert idx.startswith("HRRR") and int(idx[93:96]) == 20 and int(idx[96:99]) == 15 and int(idx[99:102]) == 2
    got = {i["var"]: a for i, a in arl.read_file(p)}
    assert np.allclose(got["SHGT"], f[(0, "SHGT")], atol=1.0)


def test_superposition_is_linear_and_lagged():
    hours = pd.date_range("2022-05-01", periods=6, freq="h", tz="UTC")
    table = pd.DataFrame({"zone": [0, 0, 1], "release": hours[[0, 0, 1]], "entry": hours[[1, 2, 3]],
                          "mass": [0.5, 0.25, 0.1]})
    tr = hysplit_transport.Transfer(hours, 2, table)
    e = np.zeros((6, 2)); e[0, 0] = 10.0; e[1, 1] = 20.0
    r = tr.arrival(e)
    assert np.allclose(r, [0, 5.0, 2.5, 2.0, 0, 0])
    assert np.allclose(tr.arrival(3 * e), 3 * r)
    assert np.isclose(tr.frac[0, 0], 0.75) and np.isclose(tr.frac[1, 1], 0.1)
