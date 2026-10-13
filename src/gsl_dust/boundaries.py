"""Vector boundaries: the Salt Lake Valley receptor (decision 1) and the railroad
causeway that separates the north arm from the south arm (lake levels by arm)."""
import json

import requests
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, shape
from shapely.ops import linemerge, transform

from gsl_dust import decisions
from gsl_dust.config import CRS, DATA_RAW

BOUNDARY_DIR = DATA_RAW / "boundaries"
COUNTY_GEOJSON = BOUNDARY_DIR / "salt_lake_county.geojson"
OSM_RAIL_JSON = BOUNDARY_DIR / "osm_rail.json"
TIGERWEB_COUNTY = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/1/query"
OVERPASS = "https://overpass-api.de/api/interpreter"

# The lake crossing of the Union Pacific Lakeside Subdivision (Lucin Cutoff).
CAUSEWAY_NAME = "UP Lakeside Subdivision"
CAUSEWAY_LON_RANGE = (-112.90, -112.40)

_to_utm = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform


def _fetch_county():
    BOUNDARY_DIR.mkdir(parents=True, exist_ok=True)
    r = requests.get(TIGERWEB_COUNTY, timeout=60, params={
        "where": "GEOID='49035'", "outFields": "GEOID,NAME", "outSR": "4326", "f": "geojson"})
    r.raise_for_status()
    COUNTY_GEOJSON.write_text(r.text)


def _fetch_rail():
    BOUNDARY_DIR.mkdir(parents=True, exist_ok=True)
    q = '[out:json][timeout:90];way["railway"](41.10,-112.95,41.35,-112.40);out geom tags;'
    r = requests.post(OVERPASS, data={"data": q}, timeout=180, headers={"User-Agent": "gsl-dust-model"})
    r.raise_for_status()
    OSM_RAIL_JSON.write_text(r.text)


def salt_lake_county_utm():
    if not COUNTY_GEOJSON.exists():
        _fetch_county()
    geom = shape(json.loads(COUNTY_GEOJSON.read_text())["features"][0]["geometry"])
    return transform(_to_utm, geom)


def valley_receptor_utm(lakebed_utm=None):
    """Salt Lake Valley receptor polygon. With a lakebed polygon supplied, the
    lakebed is removed so no source cell sits inside the receptor."""
    if decisions.VALLEY_BOUNDARY != "salt_lake_county_minus_lakebed":
        raise NotImplementedError(decisions.VALLEY_BOUNDARY)
    county = salt_lake_county_utm()
    return county.difference(lakebed_utm) if lakebed_utm is not None else county


def causeway_utm():
    if not OSM_RAIL_JSON.exists():
        _fetch_rail()
    lines = []
    for el in json.loads(OSM_RAIL_JSON.read_text())["elements"]:
        if el.get("tags", {}).get("name") != CAUSEWAY_NAME:
            continue
        coords = [(p["lon"], p["lat"]) for p in el["geometry"]]
        lons = [c[0] for c in coords]
        if max(lons) < CAUSEWAY_LON_RANGE[0] or min(lons) > CAUSEWAY_LON_RANGE[1]:
            continue
        lines.append(LineString(coords))
    merged = linemerge(MultiLineString(lines))
    return transform(_to_utm, merged)
