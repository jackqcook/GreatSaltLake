"""Extract field data from Perry, Crosman & Hoch (2019), "Results of the Great
Salt Lake Dust Plume Study (2016-2018)", University of Utah final report
(data/raw/fieldflux/Perry_2019_GSL_Dust_Plumes_Final_Report.pdf).

- Appendix C: surface condition at ~5,250 lakebed points (vegetation, crust,
  visible fines...). Parsed by character position (pdfplumber) because the X marks
  only mean something by column. Checks against the report's own counts are
  printed (section 2.6: NV 4450, VF 1153, NC/ESC 903, hot spots 446 of 5246).
- Appendix G: PM10 elemental mass fractions (resuspended <74 um soil through a
  PM10 inlet) at 120 sub-units with coordinates; arsenic is the third element.

Writes data/raw/fieldflux/perry2019_appendixC_surface.csv and
perry2019_appendixG_pm10_arsenic.csv (the latter with eta evidence against the
USGS lakebed sediment arsenic)."""
import re
import subprocess

import numpy as np
import pandas as pd
import pdfplumber
from pyproj import Transformer

from gsl_dust import chemistry
from gsl_dust.config import DATA_RAW, TABLES

PDF = DATA_RAW / "fieldflux" / "Perry_2019_GSL_Dust_Plumes_Final_Report.pdf"
CODES = "V SV NV TC MC SC NC ETC EMC ESC VF COL COM COS SA B SND BM EBM BH HAL GYP HEX CIR MR".split()


def appendix_c():
    rows = []
    with pdfplumber.open(PDF) as pdf:
        for pno in range(95, min(245, len(pdf.pages))):
            words = pdf.pages[pno].extract_words()
            vf = [w["top"] for w in words if w["text"] == "VF"]
            if not vf or not any(w["text"] == "Longitude" for w in words):
                continue
            top_hdr = min(vf)
            hdr = {w["text"]: (w["x0"] + w["x1"]) / 2 for w in words if w["text"] in CODES and abs(w["top"] - top_hdr) < 2}
            names, cx = list(hdr), np.array(list(hdr.values()))
            lines = {}
            for w in words:
                if w["top"] > top_hdr + 2:
                    lines.setdefault(round(w["top"]), []).append(w)
            for _, ws in sorted(lines.items()):
                ws = sorted(ws, key=lambda w: w["x0"])
                if len(ws) < 4 or not re.match(r"DU\d+-(SU|DU)\d+", ws[0]["text"]):
                    continue
                try:
                    lon, lat = float(ws[1]["text"]), float(ws[2]["text"])
                except ValueError:
                    continue
                flags = dict.fromkeys(CODES, 0)
                for w in ws[4:]:
                    if w["text"] == "X":
                        xc = (w["x0"] + w["x1"]) / 2
                        j = int(np.argmin(np.abs(cx - xc)))
                        if abs(cx[j] - xc) < 8:
                            flags[names[j]] = 1
                rows.append({"su": ws[0]["text"].replace("-DU", "-SU"), "lon": lon, "lat": lat, "date": ws[3]["text"],
                             "page": pno + 1, **flags})
    df = pd.DataFrame(rows)
    hot = (df.NV == 1) & (df.VF == 1) & ((df.NC == 1) | (df.ESC == 1))
    print(f"Appendix C: {len(df)} points (report 5246); NV {int(df.NV.sum())} (4450), VF {int(df.VF.sum())} (1153), "
          f"NC/ESC {int(((df.NC == 1) | (df.ESC == 1)).sum())} (903), hot spots {int(hot.sum())} (446)")
    df.to_csv(DATA_RAW / "fieldflux" / "perry2019_appendixC_surface.csv", index=False)
    return df


def appendix_g():
    txt = subprocess.run(["pdftotext", "-layout", str(PDF), "-"], capture_output=True, text=True, check=True).stdout
    rows, section = [], ""
    for line in txt.splitlines():
        if "ANALYTICAL RESULTS SUMMARY" in line:
            section = line
        m = re.match(r"\s*(DU\d+-SU\d+[A-Za-z]?)\s+(\d+\.\d+)\s+(-\d+\.\d+)\s+(\d+/\d+/\d+)\s+(.*)", line)
        if m and "(A - C)" in section:
            rows.append({"site": m.group(1), "lat": float(m.group(2)), "lon": float(m.group(3)), "date": m.group(4),
                         "pm10_as_ppm": float(m.group(5).split()[2])})
    perry = pd.DataFrame(rows)
    # eta evidence: PM10 arsenic over lakebed sediment arsenic (USGS)
    lake = pd.read_csv(TABLES / "arsenic_sites.csv")
    lake = lake[lake.on_lakebed]
    tr = Transformer.from_crs("EPSG:4326", "EPSG:26912", always_xy=True)
    px, py = tr.transform(perry.lon.values, perry.lat.values)
    med, _, _, _ = chemistry.krige(lake, px, py)
    perry["sed_as_kriged_ppm"] = med * 1e6
    perry["eta_vs_kriged"] = perry.pm10_as_ppm / perry.sed_as_kriged_ppm
    lx, ly = tr.transform(lake.longitude.values, lake.latitude.values)
    d = np.hypot(px[:, None] - lx[None], py[:, None] - ly[None])
    perry["nearest_usgs_km"] = d.min(1) / 1000
    perry["nearest_usgs_as_ppm"] = lake.as_ppm.values[d.argmin(1)]
    perry["eta_pair"] = np.where(perry.nearest_usgs_km <= 5, perry.pm10_as_ppm / perry.nearest_usgs_as_ppm, np.nan)
    q = lambda s: [round(float(v), 2) for v in s.dropna().quantile([0.05, 0.5, 0.95])]
    print(f"Appendix G: {len(perry)} sites; PM10 As median {perry.pm10_as_ppm.median():.1f} mg/kg; "
          f"eta (pairs <=5 km, n={perry.eta_pair.notna().sum()}) p5/50/95 {q(perry.eta_pair)}; "
          f"eta (all vs kriged) {q(perry.eta_vs_kriged)}")
    perry.to_csv(DATA_RAW / "fieldflux" / "perry2019_appendixG_pm10_arsenic.csv", index=False)
    return perry


if __name__ == "__main__":
    appendix_c()
    appendix_g()
