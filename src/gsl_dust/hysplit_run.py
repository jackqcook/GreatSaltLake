"""Stage 3.2: HYSPLIT, driven by HRRR, as unit-source runs per lakebed zone.

HYSPLIT's transport and deposition are linear in emitted mass, so each zone is
run once with a unit emission (1 kg/h) and every particle's first valley entry is
recorded by release hour. Any emission series beta*S then gives arrival by
superposition: R(t) = sum over zones and release hours of E(zone, hour) x the mass
that entered the valley at t per unit released then. That is HYSPLIT using the
emissions beta*S, computed once instead of once per ensemble member or scenario.
Direct HYSPLIT runs with the actual hourly emissions over the validation episodes
check the superposition (see validation)."""
import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from gsl_dust import first_entry
from gsl_dust.config import CELL_M, CRS, DATA_INTERIM
from gsl_dust.grid import LATTICE

HYSPLIT = Path.home() / "hysplit"
EXEC = HYSPLIT / "exec" / "hycs_std"
MET_LINK = HYSPLIT / "gsl_met"  # symlink to data/interim/hysplit_met (HYSPLIT paths must not contain spaces)
RUNS = HYSPLIT / "runs" / "transfer"
OUT = DATA_INTERIM / "hysplit_transfer"

ZONE_M = 10_000.0
EMIT_HOURS = 96         # emission window per run
TAIL_HOURS = 24         # particles are followed this long (KHMAX); pilot entries all came within 6 h
PARTICLES_PER_HOUR = 50
RELEASE_AGL_M = 10.0

# PM10 dust as HYSPLIT particles. decisions.SIZE_TREATMENT = uncertainty bounds:
# three sizes ride in the same run as separate pollutants (each released at the
# unit rate), central 5 um with 2.5 and 10 um as the bounds.
DIAMETERS_UM = (2.5, 5.0, 10.0)
CENTRAL_DIAMETER_UM = 5.0
DIAMETER_UM, DENSITY_GCC, SHAPE = CENTRAL_DIAMETER_UM, 2.5, 1.0
WET = "0.0 8.0E-05 8.0E-05"  # HYSPLIT's suggested in-cloud / below-cloud scavenging


def zones(zone_m=ZONE_M):
    """Lakebed cells grouped into zone_m squares. Each zone releases from the
    centroid of its lakebed cells over an area equal to theirs."""
    cells = np.load(DATA_INTERIM / "central_per_cell.npz")["cells"]
    iy, ix = np.unravel_index(cells, LATTICE.shape)
    x = LATTICE.x0 + (ix + 0.5) * CELL_M
    y = LATTICE.y1 - (iy + 0.5) * CELL_M
    key = np.floor(x / zone_m).astype(int) * 100_000 + np.floor(y / zone_m).astype(int)
    to_ll = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    rows = []
    for z, k in enumerate(np.unique(key)):
        m = key == k
        lon, lat = to_ll.transform(x[m].mean(), y[m].mean())
        rows.append({"zone": z, "lat": lat, "lon": lon, "n_cells": int(m.sum()), "area_m2": m.sum() * CELL_M ** 2})
    zone_of_cell = np.searchsorted(np.unique(key), key)
    return pd.DataFrame(rows), zone_of_cell


def windy_hours(hours, zone_of_cell, n_zones):
    """(T, Z) mask of zone-hours in which ANY parameter set in params.P could emit:
    10 m wind at some cell of the zone above the lowest threshold the ranges allow
    (smallest u*_t with the largest z0 in the log law). Outside these hours the
    threshold law gives zero emission for every ensemble member, so HYSPLIT
    releases no particles there."""
    from gsl_dust import emission, hrrr
    from gsl_dust.params import P
    u10_min = P["u_star_t"].low * np.log(emission.Z_WIND / P["z0_playa"].high) / emission.KAPPA
    cells = np.load(DATA_INTERIM / "central_per_cell.npz")["cells"]
    u = hrrr.load("u10", hours[0], hours[-1] + pd.Timedelta(hours=1)).reshape(len(hours), -1)[:, cells]
    v = hrrr.load("v10", hours[0], hours[-1] + pd.Timedelta(hours=1)).reshape(len(hours), -1)[:, cells]
    speed = np.nan_to_num(np.hypot(u, v))
    out = np.zeros((len(hours), n_zones), bool)
    for z in range(n_zones):
        out[:, z] = speed[:, zone_of_cell == z].max(axis=1) > u10_min
    return out


def chunks(start, end):
    """Emission windows [t0, t1) covering the simulation period."""
    t = start
    while t < end:
        yield t, min(t + pd.Timedelta(hours=EMIT_HOURS), end)
        t += pd.Timedelta(hours=EMIT_HOURS)


def met_days(t0, t1):
    return list(pd.date_range(t0.floor("D"), (t1 + pd.Timedelta(hours=TAIL_HOURS) - pd.Timedelta(seconds=1)).floor("D"), freq="D"))


def _control(z, t0, t1, days):
    run_h = int((t1 - t0) / pd.Timedelta(hours=1)) + TAIL_HOURS
    emit_h = int((t1 - t0) / pd.Timedelta(hours=1))
    met = "".join(f"{MET_LINK}/\nhrrr_utah_{d:%Y%m%d}.arl\n" for d in days)
    # emissions come from EMITIMES (hourly unit rate in windy hours, zero otherwise)
    pols = "".join(f"D{int(d * 10):03d}\n0.0\n0.0\n00 00 00 00 00\n" for d in DIAMETERS_UM)
    dep = "".join(f"{d} {DENSITY_GCC} {SHAPE}\n0.0 1.0 0.0 0.0 0.0\n{WET}\n0.0\n0.0\n" for d in DIAMETERS_UM)
    return (f"{t0:%y %m %d %H}\n1\n{z.lat:.5f} {z.lon:.5f} {RELEASE_AGL_M} 0.0 {z.area_m2:.4E}\n{run_h}\n0\n10000.0\n"
            f"{len(days)}\n{met}{len(DIAMETERS_UM)}\n{pols}"
            "1\n41.0 -112.4\n1.0 1.0\n1.0 1.0\n./\ncdump\n1\n100\n00 00 00 00 00\n00 00 00 00 00\n00 24 00\n"
            f"{len(DIAMETERS_UM)}\n{dep}")


def _setup(t0, t1):
    emit_h = int((t1 - t0) / pd.Timedelta(hours=1))
    run_h = emit_h + TAIL_HOURS
    return (f"&SETUP\n initd = 0,\n numpar = {PARTICLES_PER_HOUR * len(DIAMETERS_UM)},\n maxpar = 150000,\n"
            " efile = 'EMITIMES',\n"
            f" delt = 1.0,\n khmax = {TAIL_HOURS},\n ndump = -{run_h},\n ncycl = 0,\n poutf = 'PARDUMP',\n/\n")


def _emitimes(z, t0, windy):
    """One emission cycle per hour: 1 kg/h over the zone in windy hours, else 0."""
    lines = ["YYYY MM DD HH DURATION(hhhh) #RECORDS",
             "YYYY MM DD HH MM DURATION(hhmm) LAT LON HGT(m) RATE(/h) AREA(m2) HEAT(w)"]
    for k, on in enumerate(windy):
        t = t0 + pd.Timedelta(hours=k)
        lines.append(f"{t:%Y %m %d %H} 0001 {len(DIAMETERS_UM)}")
        for _ in DIAMETERS_UM:  # one record per pollutant, in CONTROL order
            lines.append(f"{t:%Y %m %d %H} 00 0100 {z.lat:.5f} {z.lon:.5f} {RELEASE_AGL_M} {1.0 if on else 0.0:.1f} "
                         f"{z.area_m2:.4E} 0.0")
    return "\n".join(lines) + "\n"


def out_path(z, t0, out=OUT):
    return out / f"{t0:%Y%m%d%H}" / f"zone{z:03d}.parquet"


def run_zone(z, t0, t1, valley_utm, windy, out=OUT):
    """One unit-source HYSPLIT run (releases in the windy hours `windy`, a bool per
    hour of [t0, t1)); writes arrivals by (release hour, entry hour). Returns run
    statistics."""
    z.zone = int(z.zone)
    dest = out_path(z.zone, t0, out)
    if dest.exists():
        return None
    if not np.any(windy):  # no member can emit here in this window
        dest.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"diameter_um": [], "release": pd.DatetimeIndex([], tz="UTC"), "entry": pd.DatetimeIndex([], tz="UTC"),
                      "mass": [], "n": [], "agl_max": [], "zone": []}).to_parquet(dest, index=False)
        return {"zone": z.zone, "chunk": str(t0), "hysplit_s": 0.0, "dump_mb": 0.0, "arrived_frac": 0.0,
                "windy_hours": 0}
    days = met_days(t0, t1)
    work = RUNS / f"{out.name}_{t0:%Y%m%d%H}_z{z.zone:03d}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    (work / "CONTROL").write_text(_control(z, t0, t1, days))
    (work / "SETUP.CFG").write_text(_setup(t0, t1))
    (work / "EMITIMES").write_text(_emitimes(z, t0, windy))
    asc = (HYSPLIT / "bdyfiles" / "ASCDATA.CFG").read_text().replace("'../bdyfiles/'", f"'{HYSPLIT}/bdyfiles/'")
    (work / "ASCDATA.CFG").write_text(asc)
    # HYSPLIT writes its per-minute particle dump into a named pipe that the tracker
    # reads as it goes, so the dump (hundreds of MB per run) never touches disk.
    fifo = work / "PARDUMP"
    os.mkfifo(fifo)
    t = pd.Timestamp.now()
    log = open(work / "run.log", "w")
    proc = subprocess.Popen([str(EXEC)], cwd=work, stdout=log, stderr=subprocess.STDOUT)

    def unblock():  # if HYSPLIT dies before opening the pipe, open it ourselves so the reader sees EOF
        proc.wait()
        try:
            fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
            os.close(fd)
        except OSError:
            pass

    watchdog = threading.Thread(target=unblock, daemon=True)
    watchdog.start()
    tracker = first_entry.StreamTracker(valley_utm, age_limit_min=TAIL_HOURS * 60)
    n_dumps = 0
    with open(fifo, "rb") as fh:
        for dump in first_entry.iter_dumps(fh):
            tracker.feed(*dump)
            n_dumps += 1
    proc.wait()
    log.close()
    if proc.returncode != 0 or n_dumps == 0:
        raise RuntimeError(f"HYSPLIT failed for zone {z.zone} {t0}: {(work / 'run.log').read_text()[-800:]}")
    hysplit_s = (pd.Timestamp.now() - t).total_seconds()
    dump_mb = 0.0
    fe = tracker.result()
    fe["entry"] = fe["entry"].dt.floor("h")
    fe["diameter_um"] = np.asarray(DIAMETERS_UM, dtype=float)[fe.pollutant.to_numpy(int) - 1]
    agg = (fe.groupby(["diameter_um", "release", "entry"]).agg(mass=("mass", "sum"), n=("mass", "size"),
                                                                 agl_max=("agl", "max")).reset_index())
    agg["zone"] = z.zone
    dest.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(dest, index=False)
    shutil.rmtree(work)
    return {"zone": z.zone, "chunk": str(t0), "hysplit_s": hysplit_s, "dump_mb": dump_mb,
            "arrived_frac": float(agg[agg.diameter_um == CENTRAL_DIAMETER_UM].mass.sum() / max(int(np.sum(windy)), 1)),
            "windy_hours": int(np.sum(windy))}


def load_transfer(start=None, end=None, out=OUT, diameter_um=CENTRAL_DIAMETER_UM):
    """Arrivals for one particle size: columns zone, release, entry, mass (kg
    arriving per kg/h released)."""
    parts = [pd.read_parquet(f) for f in sorted(out.glob("*/zone*.parquet"))]
    df = pd.concat(parts, ignore_index=True)
    df = df[df.diameter_um == diameter_um]
    if start is not None:
        df = df[(df.release >= start) & (df.release < end)]
    return df
