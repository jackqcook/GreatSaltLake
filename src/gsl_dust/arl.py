"""Write and read HYSPLIT's ARL packed meteorological format.

Follows the format in the HYSPLIT User's Guide ("Meteorological Data / ARL packed
data format") and NOAA's own converter (data2arl/api2arl/arlmod.f, MAKNDX) for the
Lambert conformal grid parameters. Each time period is one index record (INDX)
followed by one record per variable per level; every record is 50 ASCII header
bytes plus nx*ny packed bytes. Values are packed as one-byte differences from the
previous reconstructed value in the row; the first value of each row is a
difference from the first value of the row below it (rows run south to north)."""
from dataclasses import dataclass, field

import numpy as np

HEADER_LEN = 50


@dataclass
class LambertGrid:
    """ARL grid definition. For Lambert conformal, NOAA's MAKNDX sets pole lat to
    the tangent latitude and pole lon / ref lon to the orientation longitude
    (0-360). The index stores floats as F7.2, so the sync point is a lat/lon with
    two decimals and its (1-based, fractional) grid position, which keeps the
    georeferencing to ~0.01 grid cell instead of ~0.01 degree."""
    tangent_lat: float
    orient_lon: float  # 0-360
    grid_km: float
    sync_x: float
    sync_y: float
    sync_lat: float
    sync_lon: float  # 0-360
    nx: int
    ny: int
    model: str = "HRRR"

    def grids(self):
        return [self.tangent_lat, self.orient_lon, self.tangent_lat, self.orient_lon, self.grid_km, 0.0,
                self.tangent_lat, self.sync_x, self.sync_y, self.sync_lat, self.sync_lon, 0.0]


@dataclass
class Level:
    height: float  # 0 for the surface; pressure in hPa above it
    variables: list = field(default_factory=list)


def pack(values: np.ndarray):
    """Pack a (ny, nx) float array (row 0 = south). Returns (bytes, nexp, prec,
    var1, checksum). Differences are taken against reconstructed values so the
    unpacking error does not accumulate along rows."""
    v = np.asarray(values, dtype=np.float64)
    ny, nx = v.shape
    # exponent from the largest difference HYSPLIT's unpacker will see
    d_row = np.abs(np.diff(v, axis=1)).max() if nx > 1 else 0.0
    d_col = np.abs(np.diff(v[:, 0])).max() if ny > 1 else 0.0
    rmax = max(d_row, d_col)
    nexp = 0
    if rmax > 0:
        sexp = np.log2(rmax)
        nexp = int(sexp)
        if sexp >= 0.0 or sexp % 1.0 == 0.0:
            nexp += 1
    while True:
        scale = 2.0 ** (7 - nexp)
        codes = np.empty((ny, nx), dtype=np.int64)
        recon = np.empty((ny, nx), dtype=np.float64)
        var1 = v[0, 0]
        prev = var1
        for j in range(ny):  # first column chains up the rows
            c = int(np.floor((v[j, 0] - prev) * scale + 127.5))
            codes[j, 0] = c
            prev = (c - 127.0) / scale + prev
            recon[j, 0] = prev
        for i in range(1, nx):  # remaining columns: all rows at once
            c = np.floor((v[:, i] - recon[:, i - 1]) * scale + 127.5).astype(np.int64)
            codes[:, i] = c
            recon[:, i] = (c - 127.0) / scale + recon[:, i - 1]
        if codes.min() >= 0 and codes.max() <= 254:
            break
        nexp += 1  # reconstruction drift pushed a difference out of range
    flat = codes.astype(np.uint8).ravel()
    # HYSPLIT's rotating checksum adds each byte and subtracts 255 whenever the
    # sum reaches 256; for a non-negative total S that is 0 if S == 0, else
    # (S - 1) % 255 + 1.
    total = int(flat.sum(dtype=np.int64))
    ksum = 0 if total == 0 else (total - 1) % 255 + 1
    prec = 2.0 ** nexp / 254.0
    return flat.tobytes(), nexp, prec, var1, ksum


def unpack(data: bytes, nx: int, ny: int, nexp: int, var1: float) -> np.ndarray:
    """Inverse of pack(), the same arithmetic as HYSPLIT's UNPACK."""
    c = np.frombuffer(data, dtype=np.uint8).reshape(ny, nx).astype(np.float64)
    scale = 2.0 ** (7 - nexp)
    d = (c - 127.0) / scale
    first = var1 + np.cumsum(d[:, 0])
    out = np.cumsum(d, axis=1)
    out += (first - d[:, 0])[:, None]
    return out


def _header(t, level, var, nexp=0, prec=0.0, var1=0.0, forecast=0):
    s = (f"{t.year % 100:2d}{t.month:2d}{t.day:2d}{t.hour:2d}{forecast:2d}{level:2d}{99:2d}"
         f"{var:<4.4s}{nexp:4d}{prec:14.7E}{var1:14.7E}")
    assert len(s) == HEADER_LEN, s
    return s.encode("ascii")


def write_period(fh, t, grid: LambertGrid, levels: list, fields: dict, forecast=0):
    """Append one time period. `fields[(level_index, var)]` is a (ny, nx) array,
    row 0 = south; `levels[k].variables` lists the variable names in record order."""
    reclen = HEADER_LEN + grid.nx * grid.ny
    records, sums = [], {}
    for k, lev in enumerate(levels):
        for var in lev.variables:
            data, nexp, prec, var1, ksum = pack(fields[(k, var)])
            records.append(_header(t, k, var, nexp, prec, var1, forecast) + data)
            sums[(k, var)] = ksum
    g = "".join(f"{x:7.2f}" for x in grid.grids())
    fixed = f"{grid.model:<4.4s}{forecast:3d}{0:2d}{g}{grid.nx:3d}{grid.ny:3d}{len(levels):3d}{2:2d}"
    assert len(fixed) == 104
    lev_text = []
    for k, lev in enumerate(levels):
        lev_text.append(f"{lev.height:6.1f}{len(lev.variables):2d}")
        lev_text.extend(f"{var:<4.4s}{sums[(k, var)]:3d} " for var in lev.variables)
    lev_text = "".join(lev_text)
    # (I4): length of the index record excluding the 50-byte header
    index_text = fixed + f"{108 + len(lev_text):4d}" + lev_text
    assert len(index_text) <= grid.nx * grid.ny, "grid too small for the index record"
    index = _header(t, 0, "INDX", forecast=forecast) + index_text.encode("ascii").ljust(grid.nx * grid.ny)
    fh.write(index)
    for r in records:
        assert len(r) == reclen
        fh.write(r)


def read_file(path):
    """Minimal reader for checks: yields (header dict, unpacked field)."""
    raw = open(path, "rb").read()
    first = raw[:HEADER_LEN].decode()
    idx = raw[HEADER_LEN:HEADER_LEN + 108].decode()
    nx, ny = int(idx[93:96]), int(idx[96:99])
    reclen = HEADER_LEN + nx * ny
    for off in range(0, len(raw), reclen):
        h = raw[off:off + HEADER_LEN].decode()
        var = h[14:18]
        if var == "INDX":
            continue
        info = {"yy": int(h[0:2]), "mm": int(h[2:4]), "dd": int(h[4:6]), "hh": int(h[6:8]),
                "level": int(h[10:12]), "var": var, "nexp": int(h[18:22]), "var1": float(h[36:50])}
        yield info, unpack(raw[off + HEADER_LEN:off + reclen], nx, ny, info["nexp"], info["var1"])
