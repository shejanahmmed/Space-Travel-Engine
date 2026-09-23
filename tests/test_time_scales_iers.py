"""Automated validation tests for IERS Conventions (2010) relativistic time scales.

Validates:
1. Bidirectional conversion roundtrips across all 6 time scales (UTC, TAI, TT, TCG, TDB, TCB).
2. Exact standard offsets: TT - TAI = 32.184 seconds.
3. Modern leap seconds: TAI - UTC = 37.0 seconds.
4. Linear secular drift rate of TCB - TDB matching IAU 2006 L_B constant.
5. Linear secular drift rate of TCG - TT matching IAU 2000 L_G constant.
6. Spacecraft metric proper-time rate dtau / dt_TCB.
"""

import math
import numpy as np
import pytest

from relativistic_engine.constants import (
    C_LIGHT,
    GM_SUN,
    L_B,
    L_G,
    SEC_PER_DAY,
)
from relativistic_engine.time.iers_2010 import (
    convert_time_scale,
    proper_time_rate_tcb,
    tai_to_tt,
    tt_to_tai,
    utc_to_tai,
    tai_to_utc,
    TimeScale,
)
from relativistic_engine.time.time_scales import (
    JD_J2000,
    datetime_to_jd,
)


def test_standard_offsets_and_leap_seconds():
    """Verify TT - TAI = 32.184s and modern leap seconds TAI - UTC = 37.0s."""
    # 2024-01-01 00:00:00 UTC
    jd_utc = datetime_to_jd(2024, 1, 1, 0, 0, 0.0)

    jd_tai = utc_to_tai(jd_utc)
    tai_minus_utc_sec = (jd_tai - jd_utc) * SEC_PER_DAY
    assert math.isclose(tai_minus_utc_sec, 37.0, abs_tol=5e-5)

    jd_tt = tai_to_tt(jd_tai)
    tt_minus_tai_sec = (jd_tt - jd_tai) * SEC_PER_DAY
    assert math.isclose(tt_minus_tai_sec, 32.184, abs_tol=5e-5)


def test_bidirectional_roundtrip_all_scales():
    """Verify lossless roundtrip conversions across all supported time scales."""
    scales: list[TimeScale] = ["UTC", "TAI", "TT", "TCG", "TDB", "TCB"]
    jd_base = JD_J2000 + 365.25 * 25.0  # Epoch ~ 2025.0

    for s1 in scales:
        for s2 in scales:
            jd_converted = convert_time_scale(jd_base, s1, s2)
            jd_recovered = convert_time_scale(jd_converted, s2, s1)
            # Invertibility to sub-millisecond precision (~ 1e-9 days)
            assert abs(jd_recovered - jd_base) < 1e-8


def test_tcb_tdb_secular_drift_rate():
    """Verify that d(TCB - TDB)/dt matches the exact IAU 2006 rate L_B."""
    jd1 = JD_J2000
    jd2 = JD_J2000 + 365.25 * 10.0  # 10 Julian years later
    delta_days = jd2 - jd1
    delta_sec = delta_days * SEC_PER_DAY

    tcb1 = convert_time_scale(jd1, "TDB", "TCB")
    tcb2 = convert_time_scale(jd2, "TDB", "TCB")

    # Difference in seconds between TCB and TDB at both epochs
    diff1_sec = (tcb1 - jd1) * SEC_PER_DAY
    diff2_sec = (tcb2 - jd2) * SEC_PER_DAY

    drift_rate = (diff2_sec - diff1_sec) / delta_sec
    expected_rate = L_B / (1.0 - L_B)

    assert math.isclose(drift_rate, expected_rate, rel_tol=1e-5)


def test_tcg_tt_secular_drift_rate():
    """Verify that d(TCG - TT)/dt matches the exact IAU 2000 rate L_G."""
    jd1 = JD_J2000
    jd2 = JD_J2000 + 365.25 * 10.0
    delta_sec = (jd2 - jd1) * SEC_PER_DAY

    tcg1 = convert_time_scale(jd1, "TT", "TCG")
    tcg2 = convert_time_scale(jd2, "TT", "TCG")

    diff1_sec = (tcg1 - jd1) * SEC_PER_DAY
    diff2_sec = (tcg2 - jd2) * SEC_PER_DAY

    drift_rate = (diff2_sec - diff1_sec) / delta_sec
    expected_rate = L_G / (1.0 - L_G)

    assert math.isclose(drift_rate, expected_rate, rel_tol=2e-4)


def test_spacecraft_metric_proper_time_rate():
    """Verify proper-time rate dtau / dt_TCB includes both SR kinematic and GR gravitational dilation."""
    # Earth orbit: r = 1 AU, v = 29.78 km/s around Sun
    pos = np.array([1.495978707e11, 0.0, 0.0])
    vel = np.array([0.0, 29780.0, 0.0])
    bodies = [(np.array([0.0, 0.0, 0.0]), GM_SUN)]

    rate = proper_time_rate_tcb(pos, vel, bodies)

    # 1 - rate should be approx 0.5 * v^2 / c^2 + GM / (c^2 r)
    dilation = 1.0 - rate
    c2 = C_LIGHT ** 2
    expected_sr = 0.5 * (29780.0**2) / c2      # ~ 4.93e-9
    expected_gr = GM_SUN / (c2 * 1.495978707e11)  # ~ 9.87e-9
    expected_total = expected_sr + expected_gr     # ~ 1.48e-8

    assert math.isclose(dilation, expected_total, rel_tol=1e-4)
    assert 0.0 < rate < 1.0
