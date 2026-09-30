"""Paths and fixed constants. Modeling choices that are the user's to make live
in decisions.py; everything here is either a fact about a data source or plumbing."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_OUT = ROOT / "data" / "processed"
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"

TBDEM_DIR = DATA_RAW / "tbdem"
TBDEM_OVR = TBDEM_DIR / "Great_Salt_Lake_TBDEM.tif.ovr"
TBDEM_TFW = TBDEM_DIR / "Great_Salt_Lake_TBDEM.tfw"
TBDEM_WIDTH = 248_919  # full-resolution columns (USGS metadata)
TBDEM_HEIGHT = 202_228  # full-resolution rows
TBDEM_RES_M = 0.5

# NAD83 / UTM zone 12N: the terrain's horizontal reference.
CRS = "EPSG:26912"
CELL_M = 1000.0

# Vertical datums. Gauges (USGS parameter 62614) are NGVD29 feet; the terrain is
# NAVD88 meters. USGS's VERTCON-derived offset at the Saltair gauge:
NAVD88_MINUS_NGVD29_FT = 3.47585958
FT_TO_M = 0.3048

# USGS NWIS gauges (daily, parameter 62614 = NGVD29 ft).
GAUGE_SOUTH_ARM = "10010000"  # Great Salt Lake at Saltair Boat Harbor
GAUGE_NORTH_ARM = "10010100"  # Great Salt Lake near Saline
GAUGE_LOCATIONS = {  # lat, lon (NWIS site metadata)
    GAUGE_SOUTH_ARM: (40.73133329, -112.2135569),
    GAUGE_NORTH_ARM: (41.25521065, -112.496903),
}
LAKE_LEVEL_PARAM = "62614"

# USGS elevation-area-volume tables (0.01 ft steps, 4,170-4,215 ft NAVD88),
# computed from the same terrain resampled to 5 m.
HYPSOMETRY_ITEM = "6467b42fd34ec11ae4a8afb1"
HYPSOMETRY_TOP_FT_NAVD88 = 4215.0

HRRR_BUCKET = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"

for _d in (DATA_RAW, DATA_INTERIM, DATA_OUT, FIGURES, TABLES, TBDEM_DIR):
    _d.mkdir(parents=True, exist_ok=True)
