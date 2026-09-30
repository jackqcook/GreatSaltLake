"""Choices the guide leaves open. Every value here is PROVISIONAL: picked so the
build can proceed, recorded in one place so it can be changed without touching
model code. See docs/GSL_Dust_Model_Implementation_Plan.pdf, section 10."""
from datetime import datetime, timezone

# Decision 1 - Salt Lake Valley boundary (defines "entering the valley").
# Provisional: Salt Lake County (US Census TIGER) minus the lakebed.
VALLEY_BOUNDARY = "salt_lake_county_minus_lakebed"

# Decision 2 - lake-level source by arm. The two USGS gauges cover the north and
# south arms; Farmington Bay and Bear River Bay are connected to the south arm,
# so provisionally they take the south-arm (Saltair) gauge.
BAYS_USE_GAUGE = "south_arm"

# Decision 3 - size-dependent transport and enrichment: the guide allows
# particle bins or uncertainty bounds. Provisional: uncertainty bounds.
SIZE_TREATMENT = "uncertainty_bounds"

# Decision 5 - simulation period (UTC, end exclusive). 2022 contains the record
# low (Nov 2022) and has hourly PM10 at Wasatch Front monitors.
SIM_START = datetime(2022, 1, 1, tzinfo=timezone.utc)
SIM_END = datetime(2023, 1, 1, tzinfo=timezone.utc)

# Implementation definition of "lakebed": terrain at or below the top of the
# USGS elevation-area tables, 4,215 ft NAVD88 (about the 1986-87 historic high).
LAKEBED_TOP_FT_NAVD88 = 4215.0
