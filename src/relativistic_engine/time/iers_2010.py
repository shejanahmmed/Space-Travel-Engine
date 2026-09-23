"""IERS Conventions (2010) Relativistic Time Scale Conversions.

Implements the complete bidirectional conversion graph among the standard
relativistic and astronomical time scales defined by IAU and IERS:
    UTC <---> TAI <---> TT <---> TDB <---> TCB
                        |
                        +-----> TCG

Authoritative References:
- Petit, G., & Luzum, B. (eds.) (2010), "IERS Conventions (2010)", IERS Technical
  Note 36, Verlag des Bundesamts für Kartographie und Geodäsie, Chapter 10.
- IAU (2000) Resolution B1.9: "Re-definition of Terrestrial Time (TT) and
  Geocentric Coordinate Time (TCG)".
- IAU (2006) Resolution 3: "Re-definition of Barycentric Dynamical Time (TDB)".
- Soffel, M., et al. (2003), "The IAU 2000 Resolutions for Astrometry, Celestial
  Mechanics, and Metrology in the Relativistic Framework", Astron. J., 126:2687.
"""

from __future__ import annotations

import math
from typing import Dict, Literal, Tuple
import numpy as np

from relativistic_engine.constants import (
    C_LIGHT,
    L_B,
    L_G,
    SEC_PER_DAY,
)
from relativistic_engine.time.time_scales import (
    JD_J2000,
    TT_MINUS_TAI_SEC,
    TDB_0_SEC,
    datetime_to_jd,
    get_leap_seconds,
    tt_to_tdb_seconds,
)

# Standard T0 epoch for TCG and TCB: 1977 January 1, 00:00:00 TAI at geocenter
# In TT: 1977-01-01 00:00:32.184 TT = JD 2443144.5003725
T0_TAI_JD: float = 2443144.5
T0_TT_JD: float = 2443144.5003725

TimeScale = Literal["UTC", "TAI", "TT", "TCG", "TDB", "TCB"]


def utc_to_tai(jd_utc: float) -> float:
    """Convert Julian Date in UTC to TAI (International Atomic Time)."""
    delta_at = get_leap_seconds(jd_utc)
    return jd_utc + (delta_at / SEC_PER_DAY)


def tai_to_utc(jd_tai: float) -> float:
    """Convert Julian Date in TAI to UTC (iterative leap-second inversion)."""
    # Estimate UTC from TAI with nominal leap seconds
    jd_utc_est = jd_tai - (37.0 / SEC_PER_DAY)
    delta_at = get_leap_seconds(jd_utc_est)
    return jd_tai - (delta_at / SEC_PER_DAY)


def tai_to_tt(jd_tai: float) -> float:
    """Convert Julian Date in TAI to TT (Terrestrial Time). Exact offset: 32.184 s."""
    return jd_tai + (TT_MINUS_TAI_SEC / SEC_PER_DAY)


def tt_to_tai(jd_tt: float) -> float:
    """Convert Julian Date in TT to TAI (International Atomic Time)."""
    return jd_tt - (TT_MINUS_TAI_SEC / SEC_PER_DAY)


def tt_to_tcg(jd_tt: float) -> float:
    """Convert Julian Date in TT to TCG (Geocentric Coordinate Time).

    Formula (IAU 2000 Resolution B1.9):
        TCG - TT = L_G * (JD_TT - T_0) * 86400 / (1 - L_G)
    """
    elapsed_sec = (jd_tt - T0_TT_JD) * SEC_PER_DAY
    delta_sec = (L_G * elapsed_sec) / (1.0 - L_G)
    return jd_tt + (delta_sec / SEC_PER_DAY)


def tcg_to_tt(jd_tcg: float) -> float:
    """Convert Julian Date in TCG to TT (Terrestrial Time).

    Formula (IAU 2000 Resolution B1.9):
        TT - TCG = - L_G * (JD_TCG - T_0) * 86400
    """
    elapsed_sec = (jd_tcg - T0_TT_JD) * SEC_PER_DAY
    delta_sec = - L_G * elapsed_sec
    return jd_tcg + (delta_sec / SEC_PER_DAY)


def tt_to_tdb(jd_tt: float) -> float:
    """Convert Julian Date in TT to TDB (Barycentric Dynamical Time)."""
    offset_sec = tt_to_tdb_seconds(jd_tt)
    return jd_tt + (offset_sec / SEC_PER_DAY)


def tdb_to_tt(jd_tdb: float) -> float:
    """Convert Julian Date in TDB to TT (iterative periodic inversion)."""
    # 2-step fixed point iteration handles microsecond periodic terms smoothly
    jd_tt_est = jd_tdb
    for _ in range(3):
        offset_sec = tt_to_tdb_seconds(jd_tt_est)
        jd_tt_est = jd_tdb - (offset_sec / SEC_PER_DAY)
    return jd_tt_est


def tdb_to_tcb(jd_tdb: float) -> float:
    """Convert Julian Date in TDB to TCB (Barycentric Coordinate Time).

    Formula (IAU 2006 Resolution 3):
        TCB - TDB = [ L_B * (JD_TDB - T_0) * 86400 - TDB_0 ] / (1 - L_B)
    """
    elapsed_sec = (jd_tdb - T0_TAI_JD) * SEC_PER_DAY
    delta_sec = ((L_B * elapsed_sec) - TDB_0_SEC) / (1.0 - L_B)
    return jd_tdb + (delta_sec / SEC_PER_DAY)


def tcb_to_tdb(jd_tcb: float) -> float:
    """Convert Julian Date in TCB to TDB (Barycentric Dynamical Time).

    Inversion of IAU 2006 Resolution 3:
        TDB - TCB = - L_B * (JD_TCB - T_0) * 86400 + TDB_0
    """
    elapsed_sec = (jd_tcb - T0_TAI_JD) * SEC_PER_DAY
    delta_sec = - L_B * elapsed_sec + TDB_0_SEC
    return jd_tcb + (delta_sec / SEC_PER_DAY)


def convert_time_scale(
    jd: float,
    from_scale: TimeScale,
    to_scale: TimeScale,
) -> float:
    """Convert a Julian Date between any two supported relativistic time scales.

    Pivot is TT (Terrestrial Time).

    Args:
        jd: Epoch in input scale (Julian Date).
        from_scale: Source scale ('UTC', 'TAI', 'TT', 'TCG', 'TDB', 'TCB').
        to_scale: Target scale ('UTC', 'TAI', 'TT', 'TCG', 'TDB', 'TCB').

    Returns:
        Converted epoch in target scale (Julian Date).
    """
    from_upper = from_scale.upper().strip()
    to_upper = to_scale.upper().strip()

    if from_upper == to_upper:
        return float(jd)

    # Direct transformations without pivot
    if from_upper == "TDB" and to_upper == "TCB":
        return tdb_to_tcb(jd)
    if from_upper == "TCB" and to_upper == "TDB":
        return tcb_to_tdb(jd)
    if from_upper == "TT" and to_upper == "TCG":
        return tt_to_tcg(jd)
    if from_upper == "TCG" and to_upper == "TT":
        return tcg_to_tt(jd)
    if from_upper == "TT" and to_upper == "TAI":
        return tt_to_tai(jd)
    if from_upper == "TAI" and to_upper == "TT":
        return tai_to_tt(jd)
    if from_upper == "TAI" and to_upper == "UTC":
        return tai_to_utc(jd)
    if from_upper == "UTC" and to_upper == "TAI":
        return utc_to_tai(jd)

    # Transform source time scale to Terrestrial Time (TT) pivot
    if from_upper == "TT":
        jd_tt = jd
    elif from_upper == "TAI":
        jd_tt = tai_to_tt(jd)
    elif from_upper == "UTC":
        jd_tt = tai_to_tt(utc_to_tai(jd))
    elif from_upper == "TCG":
        jd_tt = tcg_to_tt(jd)
    elif from_upper == "TDB":
        jd_tt = tdb_to_tt(jd)
    elif from_upper == "TCB":
        jd_tt = tdb_to_tt(tcb_to_tdb(jd))
    else:
        raise ValueError(f"Unknown source time scale '{from_scale}'.")

    # Transform Terrestrial Time (TT) pivot to target time scale
    if to_upper == "TT":
        return jd_tt
    elif to_upper == "TAI":
        return tt_to_tai(jd_tt)
    elif to_upper == "UTC":
        return tai_to_utc(tt_to_tai(jd_tt))
    elif to_upper == "TCG":
        return tt_to_tcg(jd_tt)
    elif to_upper == "TDB":
        return tt_to_tdb(jd_tt)
    elif to_upper == "TCB":
        return tdb_to_tcb(tt_to_tdb(jd_tt))
    else:
        raise ValueError(f"Unknown target time scale '{to_scale}'.")


def proper_time_rate_tcb(
    pos_bcrs: np.ndarray,
    vel_bcrs: np.ndarray,
    body_positions_and_gms: list[tuple[np.ndarray, float]],
    *,
    c: float = C_LIGHT,
) -> float:
    """Compute the instantaneous proper-time rate dtau / dt_TCB for a spacecraft in BCRS.

    Formula (IERS 2010 / IAU 2000):
        dtau / dt_TCB = sqrt( 1 - 2 * w(x) / c^2 - v^2 / c^2 )
    where w(x) = sum_b GM_b / |x - x_b| is the gravitational potential at spacecraft position.

    Args:
        pos_bcrs: Spacecraft position vector [x, y, z] in meters relative to SSB.
        vel_bcrs: Spacecraft velocity vector [vx, vy, vz] in m/s relative to SSB.
        body_positions_and_gms: List of (body_pos, gm_val) for gravitating bodies.
        c: Speed of light in m/s.

    Returns:
        Dimensionless proper-time rate dtau / dt_TCB <= 1.0.
    """
    pos = np.asarray(pos_bcrs, dtype=np.float64)
    vel = np.asarray(vel_bcrs, dtype=np.float64)

    v_sq = float(np.dot(vel, vel))
    c2 = c * c

    potential_w = 0.0
    for b_pos, gm in body_positions_and_gms:
        if gm <= 0.0:
            continue
        dist = float(np.linalg.norm(pos - np.asarray(b_pos, dtype=np.float64)))
        if dist > 0.0:
            potential_w += gm / dist

    metric_factor = 1.0 - (2.0 * potential_w / c2) - (v_sq / c2)
    return math.sqrt(max(0.0, metric_factor))
