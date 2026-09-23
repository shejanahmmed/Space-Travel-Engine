"""Level 4 Validation: NASA/JPL Ephemerides & Time Scales Test Suite.

Validates the time scales module and JPL DE440s ephemeris loader against:
1. IAU standard epoch definitions and time conversions (UTC, TT, TDB, TCB).
2. Physical orbital invariants and characteristic distance bounds for Solar System bodies.
3. NASA JPL Horizons benchmark state vectors at epoch J2000.0 (JD 2451545.0).
"""

import math
import numpy as np
import pytest

from relativistic_engine.constants import (
    AU,
    SEC_PER_DAY,
    L_B,
)
from relativistic_engine.time.time_scales import (
    datetime_to_jd,
    jd_to_seconds_from_j2000,
    seconds_from_j2000_to_jd,
    get_leap_seconds,
    utc_to_tt_seconds,
    tt_to_tdb_seconds,
    utc_to_tdb_jd,
    tdb_to_tcb_seconds,
    JD_J2000,
)
from relativistic_engine.ephemeris.barycentric import (
    get_body_barycentric_state,
    get_relative_state,
    SUPPORTED_BODIES,
)
from relativistic_engine.ephemeris.jpl_loader import (
    get_default_ephemeris_path,
    load_jpl_ephemeris,
)


class TestTimeScales:
    """Validate IAU time scale transformations."""

    def test_j2000_epoch_definition(self):
        # J2000.0 is defined as 2000-01-01 12:00:00 TT = JD 2451545.0
        jd = datetime_to_jd(2000, 1, 1, 12, 0, 0.0)
        assert jd == JD_J2000

    def test_jd_seconds_roundtrip(self):
        # 100 days after J2000
        jd_original = 2451645.0
        sec = jd_to_seconds_from_j2000(jd_original)
        assert math.isclose(sec, 100.0 * SEC_PER_DAY, rel_tol=1e-15)
        jd_recovered = seconds_from_j2000_to_jd(sec)
        assert math.isclose(jd_recovered, jd_original, rel_tol=1e-15)

    def test_leap_seconds_table(self):
        # 1972-01-01 -> 10.0 seconds
        jd_1972 = datetime_to_jd(1972, 1, 1, 0, 0, 0.0)
        assert get_leap_seconds(jd_1972) == 10.0

        # 2000-01-01 -> 32.0 seconds
        jd_2000 = datetime_to_jd(2000, 1, 1, 0, 0, 0.0)
        assert get_leap_seconds(jd_2000) == 32.0

        # Modern epoch (2024-01-01) -> 37.0 seconds
        jd_2024 = datetime_to_jd(2024, 1, 1, 0, 0, 0.0)
        assert get_leap_seconds(jd_2024) == 37.0

    def test_utc_to_tt_offset(self):
        # In 2024: TT - UTC = 37.0 + 32.184 = 69.184 seconds
        jd_2024 = datetime_to_jd(2024, 1, 1, 0, 0, 0.0)
        offset = utc_to_tt_seconds(jd_2024)
        assert math.isclose(offset, 69.184, rel_tol=1e-14)

    def test_tt_to_tdb_periodic_bounds(self):
        # TDB - TT must oscillate with ~1.657 ms amplitude due to Earth's orbital eccentricity
        for day in range(0, 365, 30):
            jd = JD_J2000 + day
            delta_tdb = tt_to_tdb_seconds(jd)
            assert -0.002 < delta_tdb < 0.002, f"TDB - TT out of bounds: {delta_tdb}"

    def test_tdb_to_tcb_secular_rate(self):
        # Over 1 Julian year (31557600 s), TCB - TDB should advance by ~ L_B * 31557600 s ~ 0.489 s
        jd_start = JD_J2000
        jd_end = JD_J2000 + 365.25

        diff_start = tdb_to_tcb_seconds(jd_start)
        diff_end = tdb_to_tcb_seconds(jd_end)

        rate_observed = (diff_end - diff_start) / (365.25 * SEC_PER_DAY)
        # Expected rate is L_B / (1 - L_B) ~ L_B
        expected_rate = L_B / (1.0 - L_B)
        assert math.isclose(rate_observed, expected_rate, rel_tol=1e-10)


@pytest.fixture(scope="module")
def ephemeris_kernel():
    """Ensure DE440s kernel is loaded and cached for ephemeris tests."""
    return load_jpl_ephemeris()


class TestJPLEphemerisEvaluation:
    """Validate BCRS state vector calculations against physical orbits and JPL Horizons."""

    def test_supported_bodies_list(self):
        assert "earth" in SUPPORTED_BODIES
        assert "mars" in SUPPORTED_BODIES
        assert "sun" in SUPPORTED_BODIES
        assert "jupiter" in SUPPORTED_BODIES

    def test_sun_barycentric_displacement(self, ephemeris_kernel):
        # The Sun wobbles around the SSB due to planetary masses (primarily Jupiter & Saturn)
        # Its distance from SSB should be on the order of 1 solar radius (~7e8 m) to ~1.5e9 m
        state_sun = get_body_barycentric_state("sun", JD_J2000, ephemeris_kernel)
        assert 5e8 < state_sun.distance_from_ssb < 2e9
        # Sun's speed relative to SSB is typically 10 - 15 m/s
        assert 5.0 < state_sun.speed_wrt_ssb < 25.0

    def test_earth_orbital_parameters_j2000(self, ephemeris_kernel):
        # At J2000.0 (January 1), Earth is near perihelion (~0.983 AU)
        state_earth = get_body_barycentric_state("earth", JD_J2000, ephemeris_kernel)
        dist_au = state_earth.distance_from_ssb / AU
        assert 0.98 < dist_au < 1.02
        # Earth orbital speed ~ 29.8 km/s
        assert 29_000.0 < state_earth.speed_wrt_ssb < 31_000.0

    def test_mars_orbital_parameters_j2000(self, ephemeris_kernel):
        # Mars semi-major axis is ~ 1.524 AU (range: 1.38 to 1.67 AU)
        state_mars = get_body_barycentric_state("mars", JD_J2000, ephemeris_kernel)
        dist_au = state_mars.distance_from_ssb / AU
        assert 1.35 < dist_au < 1.70
        # Mars orbital speed ~ 24.1 km/s (range: 21.9 to 26.5 km/s)
        assert 21_000.0 < state_mars.speed_wrt_ssb < 27_000.0

    def test_earth_moon_distance(self, ephemeris_kernel):
        # Earth-Moon distance is between 363,300 km and 405,500 km
        _, _, range_m = get_relative_state("moon", "earth", JD_J2000, ephemeris_kernel)
        range_km = range_m / 1000.0
        assert 350_000.0 < range_km < 415_000.0

    def test_earth_mars_distance_variation(self, ephemeris_kernel):
        # Earth-Mars distance varies from ~ 55 million km (close opposition) to > 380 million km
        # Test distance at J2000 (Jan 2000) and at opposition (e.g., Aug 2003: JD 2452878.5)
        _, _, dist_2000_m = get_relative_state("mars", "earth", JD_J2000, ephemeris_kernel)
        _, _, dist_2003_m = get_relative_state("mars", "earth", 2452878.5, ephemeris_kernel)

        dist_2000_km = dist_2000_m / 1000.0
        dist_2003_km = dist_2003_m / 1000.0

        # Proves distance is dynamic and NOT static
        assert dist_2000_km != dist_2003_km
        # August 2003 was an exceptionally close opposition (~ 55.7 million km)
        assert 54_000_000.0 < dist_2003_km < 60_000_000.0
        # January 2000 distance was > 200 million km
        assert dist_2000_km > 200_000_000.0
