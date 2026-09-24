"""Project-wide configuration: site definitions, paths, and physical constants.

Site choice: Hawthorne (AQS 49-035-3006) is the primary monitor. It has one of the
longest continuous hourly PM10 records in the Salt Lake Valley and sits in a
consistent transport path from the south shore of Great Salt Lake / Farmington Bay
under the northwesterly flow regime that typically precedes lake-dust events
(Utah DEQ, Mallia & Perry GSL dust exposure work). See docs/MODEL_EXPLAINER.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models"

for _d in (DATA_RAW, DATA_PROCESSED, FIGURES_DIR, TABLES_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class MonitorSite:
    name: str
    aqs_state: str      # 2-digit FIPS
    aqs_county: str     # 3-digit county code
    aqs_site: str       # 4-digit site number
    latitude: float
    longitude: float
    asos_station: str   # nearest ASOS/METAR station for weather


# Primary site: Hawthorne Elementary, Salt Lake City
HAWTHORNE = MonitorSite(
    name="Hawthorne",
    aqs_state="49",
    aqs_county="035",
    aqs_site="3006",
    latitude=40.736389,
    longitude=-111.872222,
    asos_station="SLC",
)

PRIMARY_SITE = HAWTHORNE

# AQS parameter code: PM10 Total 0-10um STP, hourly
AQS_PARAM_PM10 = "81102"

# Analysis period. Checked empirically (see etl.py / data audit): Hawthorne's
# CONTINUOUS hourly PM10 monitor record begins in 2022 -- years before that
# exist in AQS only as intermittent FRM 24-hour filter samples, which are not
# in the "hourly" bulk files and would leave >90% of hours missing if included.
# 2022-present is also, usefully, the record-low-lake-level period through
# partial recovery, which is exactly the regime the core question is about.
START_YEAR = 2022
END_YEAR = 2026

# Great Salt Lake source geometry, relative to Hawthorne.
# Approx. south-shore exposed-playa "center of mass" used for the simple
# distance/direction transport factor D_i (first-pass; Section 3 of the modeling
# direction memo). Refined later with bay-specific centroids.
GSL_SOUTH_SHORE_LATLON = (40.85, -112.25)   # near Saltair / south shore playa
GSL_FARMINGTON_BAY_LATLON = (41.00, -112.10)

import math


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


DIST_HAWTHORNE_TO_SOUTH_SHORE_KM = _haversine_km(
    HAWTHORNE.latitude, HAWTHORNE.longitude, *GSL_SOUTH_SHORE_LATLON
)

# Bearing FROM the lake source TO the monitor (i.e. the wind direction, "from",
# that would carry lake dust toward Hawthorne). Computed once, used by the
# directional transport factor D_i in physics_model.py.

def _bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing (deg, 0=N, cw) from point 1 to point 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlambda)
    theta = math.degrees(math.atan2(x, y))
    return (theta + 360) % 360


# Wind direction (meteorological "from") that transports air FROM the lake
# source TO Hawthorne = bearing from Hawthorne back to the lake source.
WIND_DIR_FROM_LAKE_TO_HAWTHORNE_DEG = _bearing_deg(
    HAWTHORNE.latitude, HAWTHORNE.longitude, *GSL_SOUTH_SHORE_LATLON
)

# Reference "healthy" lake elevation used to define exposed playa area.
# 4,198-4,205 ft NAVD88 is the range identified by the Great Salt Lake Strike
# Team (2023) as the minimum healthy range; we use the lower bound (4198 ft) as
# the reference "full" shoreline against which exposure is measured. This is a
# modeling choice, not a claim about a natural historical maximum.
REFERENCE_HEALTHY_ELEV_FT_NAVD88 = 4198.0

# Effective PM10 emission threshold parameters are fit, not fixed; see
# physics_model.py. Reasonable literature-informed starting bounds:
U_C_BOUNDS_MPH = (10.0, 30.0)   # threshold friction/erosion wind speed
N_BOUNDS = (1.0, 3.0)           # emission-wind exponent
