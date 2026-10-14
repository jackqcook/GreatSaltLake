"""First valley entry from HYSPLIT particle dumps (guide: "count first valley
entry once").

HYSPLIT is run with a fixed 1-minute step and a particle dump every step. Particle
slots (sort index) are stable between clean-ups; a particle that leaves the
meteorological domain is zeroed (lat = lon = 0) and keeps its slot until the next
clean-up, which removes zeroed particles preserving the order of the rest; new
particles are appended. Each particle therefore keeps a global id, and its first
dump inside the valley receptor is its one and only entry: the mass it carries
then (after HYSPLIT's deposition along the way) is added to R. Every mapping is
checked against position continuity, so a change in HYSPLIT's clean-up rule
raises instead of silently mixing particles up."""
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import contains_xy, prepare

from gsl_dust.config import CRS

_TO_UTM = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
MAX_STEP_M = 5_000.0  # no particle moves 5 km in one minute (would need 83 m/s)
_I4 = np.dtype(">i4")


def read_pardump_arrays(path):
    """PARDUMP (see hysplit_io) as flat arrays plus per-dump row ranges."""
    raw = np.fromfile(path, dtype=np.uint8)
    pos, dumps, parts = 0, [], []
    while pos < raw.size:
        n = int(raw[pos:pos + 4].view(_I4)[0])
        hdr = raw[pos + 4:pos + 4 + n].view(_I4)
        pos += n + 8
        npar, npol, yy, mm, dd, hh, mi = (int(x) for x in hdr[:7])
        stride = (8 + 4 * npol) + (8 + 24) + (8 + 20)
        block = raw[pos:pos + npar * stride].reshape(npar, stride)
        pos += npar * stride
        o = 4
        mass = block[:, o:o + 4 * npol].copy().view(">f4").reshape(npar, npol).sum(1)
        o += 4 * npol + 8
        geo = block[:, o:o + 24].copy().view(">f4").reshape(npar, 6)
        o += 32
        ints = block[:, o:o + 20].copy().view(">i4").reshape(npar, 5)
        order = np.argsort(ints[:, 4], kind="stable")  # slot order
        parts.append((mass[order], geo[order, 0], geo[order, 1], geo[order, 2], ints[order, 0]))
        dumps.append(pd.Timestamp(year=2000 + yy, month=mm, day=dd, hour=hh, minute=mi, tz="UTC"))
    if not parts:
        return None
    counts = np.array([len(p[0]) for p in parts])
    cat = [np.concatenate([p[k] for p in parts]) for k in range(5)]
    return {"time": np.array(dumps), "start": np.concatenate([[0], np.cumsum(counts)[:-1]]), "count": counts,
            "mass": cat[0].astype(np.float64), "lat": cat[1], "lon": cat[2], "agl": cat[3], "age": cat[4]}


def assign_ids(a):
    """Persistent particle id for every row of read_pardump_arrays()."""
    x, y = _TO_UTM.transform(a["lon"].astype(np.float64), a["lat"].astype(np.float64))
    live = a["lat"] != 0
    pid = np.empty(a["mass"].size, dtype=np.int64)
    next_id = 0
    prev = None  # (ids, x, y, live) of the previous dump in slot order
    for k, (s, n) in enumerate(zip(a["start"], a["count"])):
        sl = slice(s, s + n)
        if prev is None:
            ids = np.arange(n)
        else:
            p_ids, p_x, p_y, p_live = prev
            ids = None
            for keep in ([np.ones(len(p_ids), bool)] + ([p_live] if not p_live.all() else [])):
                base = p_ids[keep]
                m = min(len(base), n)
                ok = keep.nonzero()[0][:m]
                both = p_live[ok] & live[sl][:m]
                if both.any() and np.hypot(x[sl][:m][both] - p_x[ok][both],
                                           y[sl][:m][both] - p_y[ok][both]).max() > MAX_STEP_M:
                    continue
                ids = np.concatenate([base[:m], np.arange(next_id, next_id + n - m)])
                break
            if ids is None:
                raise RuntimeError(f"particle slots at dump {k} ({a['time'][k]}) fit no known HYSPLIT clean-up rule")
        if n:
            next_id = max(next_id, int(ids.max()) + 1)
        pid[sl] = ids
        prev = (ids, x[sl], y[sl], live[sl])
    return pid, x, y


def first_entries(pardump_path, valley_utm) -> pd.DataFrame:
    """One row per particle that entered the valley: release hour, entry time,
    entry height (m agl), and the mass it carried at entry."""
    a = read_pardump_arrays(pardump_path)
    if a is None:
        return pd.DataFrame(columns=["release", "entry", "agl", "mass"])
    pid, x, y = assign_ids(a)
    prepare(valley_utm)
    inside = (a["lat"] != 0) & contains_xy(valley_utm, x, y)
    rows = inside.nonzero()[0]  # rows are in time order
    _, first = np.unique(pid[rows], return_index=True)
    rows = rows[first]
    dump_of_row = np.repeat(np.arange(len(a["count"])), a["count"])[rows]
    entry = pd.DatetimeIndex(a["time"][dump_of_row])
    release = (entry - pd.to_timedelta(a["age"][rows], unit="min")).floor("h")
    return pd.DataFrame({"release": release, "entry": entry, "agl": a["agl"][rows], "mass": a["mass"][rows]})


def iter_dumps(fh):
    """Yield one PARDUMP time period at a time from a binary stream (a file or the
    named pipe HYSPLIT writes into), so the dump never has to exist on disk."""
    def read_exact(n):
        buf = bytearray()
        while len(buf) < n:
            chunk = fh.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return bytes(buf)

    while True:
        head = read_exact(4)
        if head is None:
            return
        n = int(np.frombuffer(head, _I4)[0])
        hdr = np.frombuffer(read_exact(n + 4)[:n], _I4)
        npar, npol, yy, mm, dd, hh, mi = (int(x) for x in hdr[:7])
        stride = (8 + 4 * npol) + (8 + 24) + (8 + 20)
        block = np.frombuffer(read_exact(npar * stride), np.uint8).reshape(npar, stride) if npar else \
            np.zeros((0, stride), np.uint8)
        o = 4
        mass = block[:, o:o + 4 * npol].copy().view(">f4").reshape(npar, npol).sum(1)
        o += 4 * npol + 8
        geo = block[:, o:o + 24].copy().view(">f4").reshape(npar, 6)
        o += 32
        ints = block[:, o:o + 20].copy().view(">i4").reshape(npar, 5)
        order = np.argsort(ints[:, 4], kind="stable")
        yield (pd.Timestamp(year=2000 + yy, month=mm, day=dd, hour=hh, minute=mi, tz="UTC"),
               mass[order].astype(np.float64), geo[order, 0], geo[order, 1], geo[order, 2], ints[order, 0],
               ints[order, 2])


class StreamTracker:
    """assign_ids + first_entries, one dump at a time (constant memory). The
    clean-up also drops particles that reached KHMAX (age_limit_min) whether or
    not they were zeroed."""

    def __init__(self, valley_utm, age_limit_min=None):
        prepare(valley_utm)
        self.age_limit = age_limit_min
        self.valley = valley_utm
        self.prev = None
        self.next_id = 0
        self.entered = set()
        self.rows = []

    def feed(self, t, mass, lat, lon, agl, age, pollutant=None):
        n = len(mass)
        live = lat != 0
        x, y = _TO_UTM.transform(lon.astype(np.float64), lat.astype(np.float64))
        if self.prev is None:
            ids = np.arange(n)
        else:
            p_ids, p_x, p_y, p_live, p_age = self.prev
            ids = None
            cands = [np.ones(len(p_ids), bool), p_live]
            if self.age_limit is not None:
                cands.append(p_live & (p_age < self.age_limit))
            for keep in cands:
                base = p_ids[keep]
                m = min(len(base), n)
                ok = keep.nonzero()[0][:m]
                both = p_live[ok] & live[:m]
                if both.any() and np.hypot(x[:m][both] - p_x[ok][both], y[:m][both] - p_y[ok][both]).max() > MAX_STEP_M:
                    continue
                ids = np.concatenate([base[:m], np.arange(self.next_id, self.next_id + n - m)])
                break
            if ids is None:
                raise RuntimeError(f"particle slots at {t} fit no known HYSPLIT clean-up rule")
        if n:
            self.next_id = max(self.next_id, int(ids.max()) + 1)
        self.prev = (ids, x, y, live, age)
        inside = live & contains_xy(self.valley, x, y)
        for k in inside.nonzero()[0]:
            pid = int(ids[k])
            if pid not in self.entered:
                self.entered.add(pid)
                self.rows.append((t - pd.Timedelta(minutes=int(age[k])), t, float(agl[k]), float(mass[k]),
                                  int(pollutant[k]) if pollutant is not None else 1))

    def result(self):
        if not self.rows:  # no particle reached the valley: typed empty frame
            return pd.DataFrame({"release": pd.DatetimeIndex([], tz="UTC"), "entry": pd.DatetimeIndex([], tz="UTC"),
                                 "agl": pd.Series([], dtype=float), "mass": pd.Series([], dtype=float),
                                 "pollutant": pd.Series([], dtype=int)})
        df = pd.DataFrame(self.rows, columns=["release", "entry", "agl", "mass", "pollutant"])
        df["release"] = pd.DatetimeIndex(df.release).floor("h")
        return df
