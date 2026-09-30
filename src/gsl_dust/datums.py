"""Vertical datum and unit conversions. The guide: "Match elevation units and
vertical datums." Gauges are NGVD29 feet; the terrain is NAVD88 meters."""
import numpy as np

from gsl_dust.config import FT_TO_M, NAVD88_MINUS_NGVD29_FT


def ngvd29_ft_to_navd88_ft(h):
    return np.asarray(h, dtype=float) + NAVD88_MINUS_NGVD29_FT


def navd88_ft_to_m(h):
    return np.asarray(h, dtype=float) * FT_TO_M


def ngvd29_ft_to_navd88_m(h):
    return navd88_ft_to_m(ngvd29_ft_to_navd88_ft(h))


def navd88_m_to_ft(z):
    return np.asarray(z, dtype=float) / FT_TO_M
