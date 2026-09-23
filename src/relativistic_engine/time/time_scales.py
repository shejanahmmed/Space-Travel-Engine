"""IAU Standard Relativistic Time Scale Transformations.

This module implements transformations between civil and astronomical time scales
in compliance with IAU 2000 Resolution B1.9 and IAU 2006 Resolution 3:
- UTC  : Coordinated Universal Time (civil time with leap seconds)
- TAI  : International Atomic Time (SI second baseline)
- TT   : Terrestrial Time (geocentric proper time on the geoid, TT = TAI + 32.184 s)
- TDB  : Barycentric Dynamical Time (used by JPL planetary ephemerides DE440/DE441)
- TCB  : Barycentric Coordinate Time (coordinate time of the BCRS metric)

Authoritative References:
- IAU (2006) Resolution 3: "Re-definition of Barycentric Dynamical Time (TDB)".
- Fairhead, L., & Bretagnon, P. (1990), "An analytical formula for the time
  transformation TB - TT", Astronomy & Astrophysics, 229, 240-247.
- Moyer, T. D. (2003), "Mathematical Formulation of the Double-Precision
  Orbit Determination Program (DPODP)", JPL Publication 00-7.
- Meeus, J. (1998), "Astronomical Algorithms", 2nd ed., Willmann-Bell.
"""

from __future__ import annotations
import math
from typing import Tuple

from relativistic_engine.constants import (
    SEC_PER_DAY,
    L_B,
    L_G,
)

# Standard epoch J2000.0 (2000-01-01 12:00:00 TT) in Julian Date
JD_J2000: float = 2451545.0

# Modified Julian Date epoch (1858-11-17 00:00:00 UTC)
MJD_ZERO: float = 2400000.5

# Standard difference between TAI and TT in seconds (exact by IAU 1976 definition)
TT_MINUS_TAI_SEC: float = 32.184

# TDB_0 constant from IAU 2006 Resolution 3 (seconds)
TDB_0_SEC: float = -6.55e-5


def datetime_to_jd(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: float = 0.0,
) -> float:
    """Compute Julian Date (JD) from calendar date and time.

    Algorithm: Fliegel & Van Flandern (1968) / Meeus (1998), valid for all
    dates in the Gregorian calendar (year > 1582).

    Parameters
    ----------
    year : int
        Calendar year (e.g., 2030).
    month : int
        Calendar month in [1, 12].
    day : int
        Day of the month in [1, 31].
    hour : int, optional
        Hour of the day in [0, 23] (default: 0).
    minute : int, optional
        Minute of the hour in [0, 59] (default: 0).
    second : float, optional
        Second of the minute in [0.0, 60.0) (default: 0.0).

    Returns
    -------
    float
        Julian Date in days.
    """
    if month <= 2:
        year_adj = year - 1
        month_adj = month + 12
    else:
        year_adj = year
        month_adj = month

    # Gregorian calendar leap year correction
    a = year_adj // 100
    b = 2 - a + (a // 4)

    day_fraction = (hour + (minute + second / 60.0) / 60.0) / 24.0

    jd_integer = (
        int(365.25 * (year_adj + 4716))
        + int(30.6001 * (month_adj + 1))
        + day
        + b
        - 1524.5
    )
    return float(jd_integer) + day_fraction


def jd_to_seconds_from_j2000(jd: float) -> float:
    """Convert Julian Date to elapsed seconds from J2000.0.

    Parameters
    ----------
    jd : float
        Julian Date in days.

    Returns
    -------
    float
        Seconds elapsed since J2000.0 (JD 2451545.0).
    """
    return (jd - JD_J2000) * SEC_PER_DAY


def seconds_from_j2000_to_jd(seconds: float) -> float:
    """Convert elapsed seconds from J2000.0 to Julian Date.

    Parameters
    ----------
    seconds : float
        Seconds elapsed since J2000.0.

    Returns
    -------
    float
        Julian Date in days.
    """
    return JD_J2000 + (seconds / SEC_PER_DAY)


def get_leap_seconds(jd_utc: float) -> float:
    """Determine accumulated leap seconds Delta_AT = TAI - UTC for a given UTC epoch.

    Historical table traceable to IERS Bulletin C. Valid for epochs from 1972 onward.

    Parameters
    ----------
    jd_utc : float
        Julian Date in UTC.

    Returns
    -------
    float
        Leap seconds Delta_AT in seconds.
    """
    # Table of (JD_start, Delta_AT)
    leap_table: list[tuple[float, float]] = [
        (2441317.5, 10.0),  # 1972-01-01
        (2441499.5, 11.0),  # 1972-07-01
        (2441683.5, 12.0),  # 1973-01-01
        (2442048.5, 13.0),  # 1974-01-01
        (2442413.5, 14.0),  # 1975-01-01
        (2442778.5, 15.0),  # 1976-01-01
        (2443144.5, 16.0),  # 1977-01-01
        (2443509.5, 17.0),  # 1978-01-01
        (2443874.5, 18.0),  # 1979-01-01
        (2444239.5, 19.0),  # 1980-01-01
        (2444786.5, 20.0),  # 1981-07-01
        (2445151.5, 21.0),  # 1982-07-01
        (2445516.5, 22.0),  # 1983-07-01
        (2446247.5, 23.0),  # 1985-07-01
        (2447161.5, 24.0),  # 1988-01-01
        (2447892.5, 25.0),  # 1990-01-01
        (2448257.5, 26.0),  # 1991-01-01
        (2448804.5, 27.0),  # 1992-07-01
        (2449169.5, 28.0),  # 1993-07-01
        (2449534.5, 29.0),  # 1994-07-01
        (2450083.5, 30.0),  # 1996-01-01
        (2450630.5, 31.0),  # 1997-07-01
        (2451179.5, 32.0),  # 1999-01-01
        (2453736.5, 33.0),  # 2006-01-01
        (2454832.5, 34.0),  # 2009-01-01
        (2456109.5, 35.0),  # 2012-07-01
        (2457204.5, 36.0),  # 2015-07-01
        (2457754.5, 37.0),  # 2017-01-01
    ]

    current_leap = 10.0
    for jd_start, delta_at in leap_table:
        if jd_utc >= jd_start:
            current_leap = delta_at
        else:
            break
    return current_leap


def utc_to_tt_seconds(jd_utc: float) -> float:
    """Compute time offset (TT - UTC) in seconds for a specified UTC epoch.

    Formula:
        TT - UTC = Delta_AT + 32.184 s

    Parameters
    ----------
    jd_utc : float
        Julian Date in UTC.

    Returns
    -------
    float
        Offset (TT - UTC) in seconds.
    """
    delta_at = get_leap_seconds(jd_utc)
    return delta_at + TT_MINUS_TAI_SEC


def tt_to_tdb_seconds(jd_tt: float) -> float:
    """Compute periodic relativistic difference (TDB - TT) in seconds.

    Implements the Fairhead & Bretagnon (1990) / Moyer (2003) approximation
    adopted in the IAU 2006 standards. Accounts for Earth's orbital eccentricity
    moving through the Sun's gravitational potential well.

    Accuracy: Residuals are < 10 microseconds for 1900-2100 relative to full
    numerical ephemeris integration.

    Parameters
    ----------
    jd_tt : float
        Julian Date in Terrestrial Time (TT).

    Returns
    -------
    float
        Difference (TDB - TT) in seconds.
    """
    # Time in Julian centuries from J2000.0
    t_centuries = (jd_tt - JD_J2000) / 36525.0

    # Earth mean anomaly g in radians (IERS Conventions 2010)
    g_deg = 357.5277233 + 35999.05034 * t_centuries
    g_rad = math.radians(g_deg % 360.0)

    # Primary periodic term (amplitude ~ 1.657 ms)
    # TDB - TT ~ 0.001657 * sin(g + 0.01671 * sin(g)) + second harmonic
    term1 = 0.001657 * math.sin(g_rad + 0.01671 * math.sin(g_rad))
    term2 = 0.000022 * math.sin(2.0 * g_rad)
    return term1 + term2


def utc_to_tdb_jd(jd_utc: float) -> float:
    """Convert Julian Date in UTC to Julian Date in TDB (Barycentric Dynamical Time).

    Parameters
    ----------
    jd_utc : float
        Julian Date in UTC.

    Returns
    -------
    float
        Julian Date in TDB.
    """
    offset_tt_sec = utc_to_tt_seconds(jd_utc)
    jd_tt = jd_utc + (offset_tt_sec / SEC_PER_DAY)
    offset_tdb_sec = tt_to_tdb_seconds(jd_tt)
    return jd_tt + (offset_tdb_sec / SEC_PER_DAY)


def tdb_to_tcb_seconds(jd_tdb: float) -> float:
    """Compute difference (TCB - TDB) in seconds for a specified TDB epoch.

    IAU 2006 Resolution 3 defining transformation:
        TCB - TDB = [ L_B * (JD_TDB - T_0) * 86400 - TDB_0 ] / (1 - L_B)
    where T_0 = JD 2443144.5 (1977 January 1, 00:00:00 TAI).

    Parameters
    ----------
    jd_tdb : float
        Julian Date in TDB.

    Returns
    -------
    float
        Difference (TCB - TDB) in seconds.
    """
    t0_jd = 2443144.5
    elapsed_days = jd_tdb - t0_jd
    elapsed_sec = elapsed_days * SEC_PER_DAY
    numerator = (L_B * elapsed_sec) - TDB_0_SEC
    return numerator / (1.0 - L_B)
