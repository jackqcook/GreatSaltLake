import numpy as np

from gsl_dust.datums import ngvd29_ft_to_navd88_ft, ngvd29_ft_to_navd88_m, navd88_m_to_ft


def test_offset_matches_usgs_table():
    # USGS elevation-area table rows pair NAVD88 4170.00 ft with NGVD29 4166.52 ft.
    assert np.isclose(ngvd29_ft_to_navd88_ft(4166.52), 4170.0, atol=0.005)


def test_meters_round_trip():
    z = ngvd29_ft_to_navd88_m(4188.5)
    assert np.isclose(z, (4188.5 + 3.47585958) * 0.3048)
    assert np.isclose(navd88_m_to_ft(z), 4188.5 + 3.47585958)
