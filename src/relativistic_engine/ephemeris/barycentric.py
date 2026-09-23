"""Barycentric State Vector Evaluator for Solar System Bodies.

Computes precise Cartesian position and velocity state vectors in the Barycentric
Celestial Reference System (BCRS / ICRF) using NASA/JPL DE440s ephemerides.

Reference Frame & Coordinates:
- Frame: BCRS (origin at Solar System Barycenter, axes aligned with ICRF)
- Time Coordinate: TDB (Barycentric Dynamical Time)
- Units: SI (meters, meters per second)

Authoritative References:
- Park, R. S., et al. (2021), "The JPL Planetary and Lunar Ephemerides DE440 and DE441",
  The Astronomical Journal, 161:105.
- IAU 2000 Resolution B1.3: "Definition of BCRS and GCRS".
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
from jplephem.spk import SPK

from relativistic_engine.constants import SEC_PER_DAY
from relativistic_engine.ephemeris.jpl_loader import load_jpl_ephemeris

# Conversion factors: jplephem outputs position in km and velocity in km/day
KM_TO_METERS: float = 1000.0
KM_PER_DAY_TO_MPS: float = 1000.0 / SEC_PER_DAY

# List of supported Solar System bodies
SUPPORTED_BODIES: tuple[str, ...] = (
    "sun",
    "mercury",
    "venus",
    "earth",
    "moon",
    "mars",
    "jupiter",
    "saturn",
    "uranus",
    "neptune",
)


@dataclass(frozen=True)
class CelestialBodyState:
    """Astrometric Cartesian state vector of a celestial body in the BCRS frame.

    Attributes
    ----------
    body_name : str
        Canonical name of the celestial body.
    jd_tdb : float
        Evaluation epoch in Julian Date (TDB time scale).
    position : np.ndarray
        Cartesian position vector [x, y, z] in meters relative to SSB.
    velocity : np.ndarray
        Cartesian velocity vector [vx, vy, vz] in m/s relative to SSB.
    distance_from_ssb : float
        Scalar Euclidean distance |r| from Solar System Barycenter in meters.
    speed_wrt_ssb : float
        Scalar magnitude |v| of coordinate velocity in m/s.
    """

    body_name: str
    jd_tdb: float
    position: np.ndarray
    velocity: np.ndarray
    distance_from_ssb: float
    speed_wrt_ssb: float


def get_body_barycentric_state(
    body_name: str,
    jd_tdb: float,
    spk: Optional[SPK] = None,
    *,
    jd_fraction: float = 0.0,
) -> CelestialBodyState:
    """Compute the BCRS Cartesian state vector (r, v) of a body at epoch jd_tdb.

    Parameters
    ----------
    body_name : str
        Name of the target body (case-insensitive):
        'sun', 'mercury', 'venus', 'earth', 'moon', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune'.
    jd_tdb : float
        Epoch in Julian Date (integer or standard JD in TDB scale).
    spk : Optional[SPK], optional
        Active SPK kernel handle. If None, loaded from default cache.
    jd_fraction : float, optional
        Fractional day component of the epoch (default: 0.0). Using a 2-part JD
        (jd_tdb, jd_fraction) avoids floating-point cancellation for sub-second precision.

    Returns
    -------
    CelestialBodyState
        Complete BCRS state vector in SI units (meters, m/s).

    Raises
    ------
    ValueError
        If the body name is unrecognized.
    """
    body = body_name.lower().strip()
    if body not in SUPPORTED_BODIES:
        raise ValueError(
            f"Unsupported celestial body '{body_name}'. Supported bodies: {SUPPORTED_BODIES}"
        )

    kernel = spk if spk is not None else load_jpl_ephemeris()

    # 2-part JD evaluation preserves full 53-bit mantissa on the fractional day
    if body == "sun":
        pos_km, vel_km_day = kernel[0, 10].compute_and_differentiate(jd_tdb, jd_fraction)
    elif body == "mercury":
        pos_km, vel_km_day = kernel[0, 1].compute_and_differentiate(jd_tdb, jd_fraction)
    elif body == "venus":
        pos_km, vel_km_day = kernel[0, 2].compute_and_differentiate(jd_tdb, jd_fraction)
    elif body == "earth":
        pos_emb, vel_emb = kernel[0, 3].compute_and_differentiate(jd_tdb, jd_fraction)
        pos_earth_emb, vel_earth_emb = kernel[3, 399].compute_and_differentiate(jd_tdb, jd_fraction)
        pos_km = pos_emb + pos_earth_emb
        vel_km_day = vel_emb + vel_earth_emb
    elif body == "moon":
        pos_emb, vel_emb = kernel[0, 3].compute_and_differentiate(jd_tdb, jd_fraction)
        pos_moon_emb, vel_moon_emb = kernel[3, 301].compute_and_differentiate(jd_tdb, jd_fraction)
        pos_km = pos_emb + pos_moon_emb
        vel_km_day = vel_emb + vel_moon_emb
    elif body == "mars":
        pos_km, vel_km_day = kernel[0, 4].compute_and_differentiate(jd_tdb, jd_fraction)
    elif body == "jupiter":
        pos_km, vel_km_day = kernel[0, 5].compute_and_differentiate(jd_tdb, jd_fraction)
    elif body == "saturn":
        pos_km, vel_km_day = kernel[0, 6].compute_and_differentiate(jd_tdb, jd_fraction)
    elif body == "uranus":
        pos_km, vel_km_day = kernel[0, 7].compute_and_differentiate(jd_tdb, jd_fraction)
    elif body == "neptune":
        pos_km, vel_km_day = kernel[0, 8].compute_and_differentiate(jd_tdb, jd_fraction)

    # Convert km and km/day to exact SI units (meters and m/s)
    pos_m = np.asarray(pos_km, dtype=np.float64) * KM_TO_METERS
    vel_mps = np.asarray(vel_km_day, dtype=np.float64) * KM_PER_DAY_TO_MPS

    dist_ssb = float(np.linalg.norm(pos_m))
    speed_ssb = float(np.linalg.norm(vel_mps))

    return CelestialBodyState(
        body_name=body,
        jd_tdb=jd_tdb,
        position=pos_m,
        velocity=vel_mps,
        distance_from_ssb=dist_ssb,
        speed_wrt_ssb=speed_ssb,
    )


def get_relative_state(
    target_body: str,
    observer_body: str,
    jd_tdb: float,
    spk: Optional[SPK] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute the relative state (r_rel, v_rel, distance) of target relative to observer.

    Parameters
    ----------
    target_body : str
        Target body name (e.g., 'mars').
    observer_body : str
        Observer body name (e.g., 'earth').
    jd_tdb : float
        Evaluation epoch in Julian Date (TDB scale).
    spk : Optional[SPK], optional
        Active SPK kernel handle.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        (r_relative_meters, v_relative_mps, scalar_range_meters).
    """
    s_target = get_body_barycentric_state(target_body, jd_tdb, spk)
    s_obs = get_body_barycentric_state(observer_body, jd_tdb, spk)

    r_rel = s_target.position - s_obs.position
    v_rel = s_target.velocity - s_obs.velocity
    range_m = float(np.linalg.norm(r_rel))

    return r_rel, v_rel, range_m
