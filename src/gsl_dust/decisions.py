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

# Decision 6 - scenario lake levels (Stage 5), NGVD29 feet as the gauges report
# them, applied to both arms. 4,188.5 = Nov 2022 record low; 4,198-4,205 =
# Great Salt Lake Strike Team healthy range (datum as reported by the state:
# assumed gauge datum, NGVD29 - confirm).
SCENARIO_LEVELS_NGVD29_FT = {
    "record_low_4188.5": 4188.5,
    "4192": 4192.0,
    "4195": 4195.0,
    "healthy_low_4198": 4198.0,
    "healthy_high_4205": 4205.0,
}

# Implementation definition of "lakebed": terrain at or below the top of the
# USGS elevation-area tables, 4,215 ft NAVD88 (about the 1986-87 historic high).
LAKEBED_TOP_FT_NAVD88 = 4215.0
