"""Stage 4 direct HYSPLIT runs over the validation episodes.

For each episode in validation/heldout_design.json, HYSPLIT is run with the actual
hourly emissions beta*S of every zone (EMITIMES), from 24 h before the episode to
its end. Outputs:
- hourly air concentration (0-100 m layer) at every PM10 monitor, for the guide's
  comparison of monitors with HYSPLIT concentrations (never with R);
- valley arrival counted directly from the particles, to check the unit-source
  superposition used for R (hysplit_transport) against an ordinary HYSPLIT run."""
import json
import os
import shutil
import subprocess
import threading

import numpy as np
import pandas as pd

from gsl_dust import first_entry, hysplit_run
from gsl_dust.config import DATA_RAW, ROOT

OUT = ROOT / "data" / "interim" / "hysplit_episodes"
SPINUP_H = 24
PARTICLES_PER_SOURCE_HOUR = 40
KG_M3_TO_UG_M3 = 1e9


def monitors():
    d = pd.read_parquet(DATA_RAW / "aqs" / "pm10_2022_wasatch.parquet")
    return d.groupby("site_id")[["lat", "lon"]].first().reset_index()


def episodes():
    return json.loads((ROOT / "validation" / "heldout_design.json").read_text())


def _window(ep):
    t0 = pd.Timestamp(ep["start"]).floor("h") - pd.Timedelta(hours=SPINUP_H)
    t1 = min(pd.Timestamp(ep["end"]).floor("h") + pd.Timedelta(hours=2), pd.Timestamp(ep["end"]).floor("D") + pd.Timedelta(days=1))
    return t0, t1


def run_episode(ep, e_zone, hours, zones, valley_utm):
    """Returns (concentration DataFrame [time, site_id, conc_ug_m3], direct R per hour Series)."""
    t0, t1 = _window(ep)
    a0, a1 = hours.get_loc(t0), hours.get_loc(t1 - pd.Timedelta(hours=1)) + 1
    e = e_zone[a0:a1]
    src = np.nonzero(e.sum(0) > 0)[0]
    run_h = a1 - a0
    work = hysplit_run.HYSPLIT / "runs" / f"episode_{ep['id']:02d}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    mon = monitors()
    (work / "stations.txt").write_text("".join(f"{i + 1} {r.lat:.5f} {r.lon:.5f}\n" for i, r in enumerate(mon.itertuples())))
    if len(src) == 0:
        return pd.DataFrame(columns=["time", "site_id", "conc_ug_m3"]), pd.Series(dtype=float)
    days = list(pd.date_range(t0.floor("D"), (t1 - pd.Timedelta(seconds=1)).floor("D"), freq="D"))
    zs = zones.iloc[src]
    met = "".join(f"{hysplit_run.MET_LINK}/\nhrrr_utah_{d:%Y%m%d}.arl\n" for d in days)
    srcs = "".join(f"{z.lat:.5f} {z.lon:.5f} {hysplit_run.RELEASE_AGL_M} 0.0 {z.area_m2:.4E}\n" for z in zs.itertuples())
    (work / "CONTROL").write_text(
        f"{t0:%y %m %d %H}\n{len(src)}\n{srcs}{run_h}\n0\n10000.0\n{len(days)}\n{met}1\nDUST\n0.0\n0.0\n00 00 00 00 00\n"
        "1\n40.80 -111.90\n0.01 0.01\n1.3 1.3\n./\ncdump\n1\n100\n00 00 00 00 00\n00 00 00 00 00\n00 01 00\n"
        f"1\n{hysplit_run.DIAMETER_UM} {hysplit_run.DENSITY_GCC} {hysplit_run.SHAPE}\n0.0 1.0 0.0 0.0 0.0\n"
        f"{hysplit_run.WET}\n0.0\n0.0\n")
    lines = ["YYYY MM DD HH DURATION(hhhh) #RECORDS",
             "YYYY MM DD HH MM DURATION(hhmm) LAT LON HGT(m) RATE(/h) AREA(m2) HEAT(w)"]
    for k in range(run_h):
        t = t0 + pd.Timedelta(hours=k)
        lines.append(f"{t:%Y %m %d %H} 0001 {len(src)}")
        for j, z in zip(src, zs.itertuples()):
            lines.append(f"{t:%Y %m %d %H} 00 0100 {z.lat:.5f} {z.lon:.5f} {hysplit_run.RELEASE_AGL_M} "
                         f"{e[k, j]:.6E} {z.area_m2:.4E} 0.0")
    (work / "EMITIMES").write_text("\n".join(lines) + "\n")
    (work / "SETUP.CFG").write_text(
        f"&SETUP\n initd = 0,\n numpar = {PARTICLES_PER_SOURCE_HOUR * len(src)},\n maxpar = 400000,\n"
        f" efile = 'EMITIMES',\n delt = 1.0,\n khmax = {hysplit_run.TAIL_HOURS},\n ndump = -{run_h},\n ncycl = 0,\n"
        " poutf = 'PARDUMP',\n/\n")
    asc = (hysplit_run.HYSPLIT / "bdyfiles" / "ASCDATA.CFG").read_text().replace(
        "'../bdyfiles/'", f"'{hysplit_run.HYSPLIT}/bdyfiles/'")
    (work / "ASCDATA.CFG").write_text(asc)

    fifo = work / "PARDUMP"
    os.mkfifo(fifo)
    log = open(work / "run.log", "w")
    proc = subprocess.Popen([str(hysplit_run.EXEC)], cwd=work, stdout=log, stderr=subprocess.STDOUT)

    def unblock():
        proc.wait()
        try:
            os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
        except OSError:
            pass

    threading.Thread(target=unblock, daemon=True).start()
    tracker = first_entry.StreamTracker(valley_utm, age_limit_min=hysplit_run.TAIL_HOURS * 60)
    with open(fifo, "rb") as fh:
        for dump in first_entry.iter_dumps(fh):
            tracker.feed(*dump)
    proc.wait()
    log.close()
    if proc.returncode != 0:
        raise RuntimeError(f"episode {ep['id']}: {(work / 'run.log').read_text()[-800:]}")
    fe = tracker.result()
    r_direct = fe.groupby(fe.entry.dt.floor("h")).mass.sum() if len(fe) else pd.Series(dtype=float)

    subprocess.run([str(hysplit_run.HYSPLIT / "exec" / "con2stn"), "-icdump", "-sstations.txt", "-oconc.txt",
                    f"-c{KG_M3_TO_UG_M3:.0E}", "-r1", "-d1", "-xi"], cwd=work, capture_output=True, check=True)
    conc = _read_con2stn(work / "conc.txt", mon)
    shutil.rmtree(work)
    return conc, r_direct


def _read_con2stn(path, mon):
    """con2stn column output (-r1 -d1): date, YR MO DA1 HR1 MN1 DA2 HR2 MN2, then one
    value per station; -1 marks a station outside the grid. Times are the start of
    each hourly average, matching AQS hourly PM10."""
    recs = []
    for line in path.read_text().splitlines()[1:]:
        r = line.split()
        if len(r) < 9 + len(mon):
            continue
        yr, mo, da, hr = int(r[1]), int(r[2]), int(r[3]), int(r[4])
        t = pd.Timestamp(year=2000 + yr, month=mo, day=da, hour=hr, tz="UTC")
        for sid, v in zip(mon.site_id, r[9:9 + len(mon)]):
            v = float(v)
            recs.append({"time": t, "site_id": sid, "conc_ug_m3": np.nan if v < 0 else v})
    return pd.DataFrame(recs)
