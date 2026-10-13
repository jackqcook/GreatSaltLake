"""Read HYSPLIT particle dumps (PARDUMP).

Layout from the HYSPLIT User's Guide ("Particle Dump File Format"): big-endian
Fortran sequential records. Per dump time: a header record (number of particles,
pollutants per particle, year, month, day, hour, minute), then for each particle
three records: masses (one per pollutant), position/turbulence (lat, lon, height
agl, sigma-h, sigma-w, sigma-v), and integers (age in minutes, distribution,
pollutant, meteorological grid, sort index)."""
import numpy as np
import pandas as pd

_I4 = np.dtype(">i4")


def read_pardump(path):
    """Returns a DataFrame with one row per particle per dump time."""
    raw = np.fromfile(path, dtype=np.uint8)
    pos, frames = 0, []
    while pos < raw.size:
        n = int(raw[pos:pos + 4].view(_I4)[0])
        hdr = raw[pos + 4:pos + 4 + n].view(_I4)
        pos += n + 8
        npar, npol, yy, mm, dd, hh, mi = (int(x) for x in hdr[:7])
        rec_mass = 4 * npol
        stride = (8 + rec_mass) + (8 + 24) + (8 + 20)
        block = raw[pos:pos + npar * stride].reshape(npar, stride)
        pos += npar * stride
        o = 4
        mass = block[:, o:o + rec_mass].copy().view(">f4").reshape(npar, npol).sum(1)
        o += rec_mass + 8
        geo = block[:, o:o + 24].copy().view(">f4").reshape(npar, 6)
        o += 24 + 8
        ints = block[:, o:o + 20].copy().view(">i4").reshape(npar, 5)
        t = pd.Timestamp(year=2000 + yy if yy < 70 else 1900 + yy, month=mm, day=dd, hour=hh, minute=mi, tz="UTC")
        frames.append(pd.DataFrame({
            "time": t, "mass": mass.astype(np.float64), "lat": geo[:, 0], "lon": geo[:, 1], "agl": geo[:, 2],
            "age_min": ints[:, 0], "pollutant": ints[:, 2], "sort": ints[:, 4]}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
