"""NASA JPL Horizons ephemeris cross-validation harness.

Provides independent reference state vectors from NASA JPL Horizons DE440
and cross-validates engine BCRS planetary evaluations.

Authoritative Standards:
- Park, R. S., et al. (2021), "The JPL Planetary and Lunar Ephemerides DE440 and DE441", AJ 161, 105.
- NASA JPL Horizons Web Interface / Ephemeris System (ICRF / BCRS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence
import numpy as np
from jplephem.spk import SPK

from relativistic_engine.ephemeris.barycentric import get_body_barycentric_state


@dataclass(frozen=True)
class HorizonsReferenceVector:
    """Independent reference state vector from NASA JPL Horizons.

    Attributes
    ----------
    body : str
        Target body name.
    epoch_jd_tdb : float
        Epoch in Julian Date (TDB scale).
    position_m : np.ndarray
        BCRS position vector [x, y, z] in meters.
    velocity_m_s : np.ndarray
        BCRS velocity vector [vx, vy, vz] in m/s.
    description : str
        Ephemeris record description.
    """

    body: str
    epoch_jd_tdb: float
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    description: str


# Independent JPL Horizons DE440 reference states
# Reference frame: ICRF / BCRS (Barycentric Celestial Reference System)
# Time scale: TDB
HORIZONS_DE440_BENCHMARKS: Sequence[HorizonsReferenceVector] = [
    # J2000.0 (2000-Jan-01 12:00:00 TDB = JD 2451545.0)
    HorizonsReferenceVector(
        body="sun",
        epoch_jd_tdb=2451545.0,
        position_m=np.array([-1067706805.3809534, -396036184.79594624, -138065184.2868809]),
        velocity_m_s=np.array([9.312571926520471, -11.701506128177709, -5.251266205200356]),
        description="Sun Barycentric State at J2000.0 (DE440)",
    ),
    HorizonsReferenceVector(
        body="earth",
        epoch_jd_tdb=2451545.0,
        position_m=np.array([-27566740482.806046, 132361381153.54352, 57418653286.25131]),
        velocity_m_s=np.array([-29784.947498495447, -5029.753814914289, -2180.6450690318698]),
        description="Earth Barycentric State at J2000.0 (DE440)",
    ),
    HorizonsReferenceVector(
        body="mars",
        epoch_jd_tdb=2451545.0,
        position_m=np.array([206980433836.45142, -186417011.4371795, -5667227497.526179]),
        velocity_m_s=np.array([1171.985008531777, 23906.708194170737, 10933.920633307645]),
        description="Mars Barycentric State at J2000.0 (DE440)",
    ),
    HorizonsReferenceVector(
        body="jupiter",
        epoch_jd_tdb=2451545.0,
        position_m=np.array([597499876792.548, 408990313931.75867, 160756281938.7201]),
        velocity_m_s=np.array([-7900.525116640771, 10171.796309237907, 4552.467787262923]),
        description="Jupiter Barycentric State at J2000.0 (DE440)",
    ),
    # 2030-May-01 00:00:00 TDB (JD 2462622.5)
    HorizonsReferenceVector(
        body="earth",
        epoch_jd_tdb=2462622.5,
        position_m=np.array([-114992447936.25137, -89264949701.44916, -38685905895.92259]),
        velocity_m_s=np.array([18764.843996406988, -20965.614730886526, -9087.31325429579]),
        description="Earth Barycentric State at 2030-May-01 (DE440)",
    ),
    HorizonsReferenceVector(
        body="mars",
        epoch_jd_tdb=2462622.5,
        position_m=np.array([140137813813.99844, 156098401034.19122, 67825584077.17625]),
        velocity_m_s=np.array([-17791.120973078698, 15717.039402012202, 7688.641596992703]),
        description="Mars Barycentric State at 2030-May-01 (DE440)",
    ),
    # 2050-Jan-01 00:00:00 TDB (JD 2469807.5)
    HorizonsReferenceVector(
        body="earth",
        epoch_jd_tdb=2469807.5,
        position_m=np.array([-25552989891.650795, 132440488043.86417, 57404342424.57359]),
        velocity_m_s=np.array([-29802.97244751879, -4880.549291248201, -2114.264436233668]),
        description="Earth Barycentric State at 2050-Jan-01 (DE440)",
    ),
    HorizonsReferenceVector(
        body="mars",
        epoch_jd_tdb=2469807.5,
        position_m=np.array([-230744349446.9298, -71200885041.76103, -26431720749.316875]),
        velocity_m_s=np.array([8427.288034361149, -18969.44428987821, -8927.856703669346]),
        description="Mars Barycentric State at 2050-Jan-01 (DE440)",
    ),
]


@dataclass(frozen=True)
class HorizonsValidationReport:
    """Summary of cross-validation results against JPL Horizons DE440.

    Attributes
    ----------
    total_checks : int
        Number of state vector comparisons evaluated.
    max_position_residual_meters : float
        Maximum Euclidean distance residual ||r_engine - r_horizons|| across all checks.
    max_velocity_residual_m_s : float
        Maximum velocity residual ||v_engine - v_horizons|| across all checks.
    is_valid : bool
        True if all residuals satisfy tolerance limits.
    """

    total_checks: int
    max_position_residual_meters: float
    max_velocity_residual_m_s: float
    is_valid: bool


def validate_ephemeris_against_horizons(
    spk: Optional[SPK] = None,
    *,
    position_tolerance_m: float = 1.0e-3,  # 1 mm tolerance
    velocity_tolerance_m_s: float = 1.0e-6, # 1 um/s tolerance
) -> HorizonsValidationReport:
    """Validate engine planetary state evaluations against JPL Horizons DE440 records.

    Parameters
    ----------
    spk : Optional[SPK]
        Optional pre-loaded SPK kernel.
    position_tolerance_m : float
        Maximum acceptable position error in meters.
    velocity_tolerance_m_s : float
        Maximum acceptable velocity error in m/s.

    Returns
    -------
    HorizonsValidationReport
        Validation metrics and compliance status.
    """
    max_dr = 0.0
    max_dv = 0.0

    for ref in HORIZONS_DE440_BENCHMARKS:
        state = get_body_barycentric_state(ref.body, ref.epoch_jd_tdb, spk=spk)
        dr = float(np.linalg.norm(state.position - ref.position_m))
        dv = float(np.linalg.norm(state.velocity - ref.velocity_m_s))

        if dr > max_dr:
            max_dr = dr
        if dv > max_dv:
            max_dv = dv

    is_valid = (max_dr <= position_tolerance_m) and (max_dv <= velocity_tolerance_m_s)

    return HorizonsValidationReport(
        total_checks=len(HORIZONS_DE440_BENCHMARKS),
        max_position_residual_meters=max_dr,
        max_velocity_residual_m_s=max_dv,
        is_valid=is_valid,
    )
