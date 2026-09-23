"""Interstellar stellar kinematics and astrometry in ICRS / BCRS.

Propagates 3D barycentric position and space velocity vectors for nearby stars
using official Gaia DR3 and Hipparcos astrometric observables:
- Right Ascension (alpha) and Declination (delta)
- Parallax (varpi) in milliarcseconds (mas)
- Proper motions (mu_alpha_cosdec, mu_delta) in mas/yr
- Barycentric radial velocity (v_r) in m/s

Authoritative References:
- Gaia Collaboration et al. (2023), "Gaia Data Release 3: Summary of the content and survey properties", A&A 674, A1.
- van Leeuwen, F. (2007), "Validation of the new Hipparcos reduction", A&A 474, 653-664.
- IAU 2000 / 2006 Resolutions on the International Celestial Reference System (ICRS).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping
import numpy as np

from relativistic_engine.constants import AU, SEC_PER_DAY


# Conversion constant k = AU / (1 Julian Year in seconds) in m*yr/(s*AU)
# Relates proper motion (mas/yr) and parallax (mas) to transverse velocity (m/s)
K_KM_PER_S = AU / (365.25 * SEC_PER_DAY)

# J2000.0 epoch in Julian Date
JD_J2000 = 2451545.0

# Gaia DR3 reference epoch: J2016.0 (2016 Jan 1 12:00:00 TCB = JD 2457388.5)
JD_GAIA_DR3 = 2457388.5


@dataclass(frozen=True)
class StarCatalogEntry:
    """Astrometric catalog entry for an interstellar star in the ICRS.

    Attributes
    ----------
    name : str
        Common star identifier.
    ra_deg : float
        Right Ascension in degrees (ICRS).
    dec_deg : float
        Declination in degrees (ICRS).
    parallax_mas : float
        Trigonometric parallax in milliarcseconds (mas).
    pm_ra_cosdec_mas_yr : float
        Proper motion in right ascension mu_alpha* = mu_alpha * cos(dec) in mas/yr.
    pm_dec_mas_yr : float
        Proper motion in declination mu_delta in mas/yr.
    radial_velocity_m_s : float
        Barycentric radial velocity v_r in m/s (positive receding, negative approaching).
    epoch_jd : float
        Catalog reference epoch in Julian Date (TDB/TCB).
    """

    name: str
    ra_deg: float
    dec_deg: float
    parallax_mas: float
    pm_ra_cosdec_mas_yr: float
    pm_dec_mas_yr: float
    radial_velocity_m_s: float
    epoch_jd: float

    @property
    def distance_meters(self) -> float:
        """Nominal distance in meters at catalog epoch based on parallax."""
        # varpi in radians: mas * 1e-3 * (pi / (180 * 3600))
        varpi_rad = self.parallax_mas * 1.0e-3 * (math.pi / 648000.0)
        return AU / math.sin(varpi_rad)

    @property
    def distance_light_years(self) -> float:
        """Nominal distance in light years at catalog epoch."""
        from relativistic_engine.constants import LIGHT_YEAR
        return self.distance_meters / LIGHT_YEAR


@dataclass(frozen=True)
class StarStateBCRS:
    """Barycentric state vector of an interstellar star at an arbitrary epoch.

    Attributes
    ----------
    name : str
        Star name.
    position : np.ndarray
        3D position vector [x, y, z] in meters relative to Solar System Barycenter (BCRS).
    velocity : np.ndarray
        3D space velocity vector [vx, vy, vz] in m/s relative to Solar System Barycenter.
    epoch_jd : float
        Julian Date (TDB scale) at which state is evaluated.
    distance_meters : float
        Euclidean distance from Solar System Barycenter in meters.
    distance_light_years : float
        Euclidean distance in light years.
    """

    name: str
    position: np.ndarray
    velocity: np.ndarray
    epoch_jd: float
    distance_meters: float
    distance_light_years: float


# Official Gaia DR3 / Hipparcos astrometric catalog for nearby stars
INTERSTELLAR_CATALOG: Mapping[str, StarCatalogEntry] = {
    "proxima_centauri": StarCatalogEntry(
        name="Proxima Centauri",
        ra_deg=217.428953,             # 14h 29m 42.95s
        dec_deg=-62.679484,            # -62d 40' 46.1''
        parallax_mas=768.0665,         # Gaia DR3 5853498713190525696 (0.0499 mas unc)
        pm_ra_cosdec_mas_yr=-3781.741, # Gaia DR3
        pm_dec_mas_yr=769.465,         # Gaia DR3
        radial_velocity_m_s=-22200.0,  # Gaia DR3 (-22.20 km/s)
        epoch_jd=JD_GAIA_DR3,
    ),
    "alpha_centauri_a": StarCatalogEntry(
        name="Alpha Centauri A",
        ra_deg=219.902058,             # 14h 39m 36.49s
        dec_deg=-60.833975,            # -60d 50' 02.3''
        parallax_mas=747.23,           # Hipparcos / Gaia
        pm_ra_cosdec_mas_yr=-3679.25,  # Hipparcos
        pm_dec_mas_yr=473.67,          # Hipparcos
        radial_velocity_m_s=-22300.0,  # -22.3 km/s
        epoch_jd=JD_J2000,
    ),
    "alpha_centauri_b": StarCatalogEntry(
        name="Alpha Centauri B",
        ra_deg=219.896096,             # 14h 39m 35.06s
        dec_deg=-60.837528,            # -60d 50' 15.1''
        parallax_mas=747.23,           # Hipparcos / Gaia
        pm_ra_cosdec_mas_yr=-3614.0,   # Hipparcos
        pm_dec_mas_yr=802.0,           # Hipparcos
        radial_velocity_m_s=-22700.0,  # -22.7 km/s
        epoch_jd=JD_J2000,
    ),
    "barnards_star": StarCatalogEntry(
        name="Barnard's Star",
        ra_deg=269.452075,             # 17h 57m 48.50s
        dec_deg=4.693392,              # +04d 41' 36.2''
        parallax_mas=546.9759,         # Gaia DR3 4472832130942585856
        pm_ra_cosdec_mas_yr=-801.551,  # Gaia DR3
        pm_dec_mas_yr=10362.544,       # Gaia DR3
        radial_velocity_m_s=-110510.0, # Gaia DR3 (-110.51 km/s)
        epoch_jd=JD_GAIA_DR3,
    ),
}


def get_star_barycentric_state(
    star_name: str,
    epoch_jd_tdb: float,
    *,
    jd_fraction: float = 0.0,
) -> StarStateBCRS:
    """Compute the 3D BCRS position and velocity of an interstellar star.

    Propagates the initial ICRS position and space velocity vector under rectilinear
    space motion over delta_t = (jd_tdb - epoch_catalog).

    Parameters
    ----------
    star_name : str
        Identifier matching an entry in `INTERSTELLAR_CATALOG` (case-insensitive).
    epoch_jd_tdb : float
        Target Julian Date (TDB scale).
    jd_fraction : float
        Fractional day component for precision time representation.

    Returns
    -------
    StarStateBCRS
        State containing 3D BCRS position, space velocity, and distance metrics.
    """
    key = star_name.lower().strip().replace(" ", "_").replace("'", "")
    if key not in INTERSTELLAR_CATALOG:
        available = list(INTERSTELLAR_CATALOG.keys())
        raise KeyError(f"Unknown interstellar star '{star_name}'. Available: {available}")

    entry = INTERSTELLAR_CATALOG[key]

    # Convert angular coordinates to radians
    alpha_rad = math.radians(entry.ra_deg)
    delta_rad = math.radians(entry.dec_deg)

    cos_a = math.cos(alpha_rad)
    sin_a = math.sin(alpha_rad)
    cos_d = math.cos(delta_rad)
    sin_d = math.sin(delta_rad)

    # Unit vectors forming the local astrometric triad (u0, p0, q0)
    # u0: Line-of-sight unit vector pointing from Solar System Barycenter to star
    u0 = np.array([cos_d * cos_a, cos_d * sin_a, sin_d], dtype=np.float64)

    # p0: Unit vector in direction of increasing Right Ascension
    p0 = np.array([-sin_a, cos_a, 0.0], dtype=np.float64)

    # q0: Unit vector in direction of increasing Declination
    q0 = np.array([-sin_d * cos_a, -sin_d * sin_a, cos_d], dtype=np.float64)

    # Initial distance r0 at catalog epoch
    r0 = entry.distance_meters
    r0_vec = r0 * u0

    # Space velocity calculation (tangential components via proper motion)
    # v_alpha = (mu_alpha* / varpi) * K_KM_PER_S
    # v_delta = (mu_delta / varpi) * K_KM_PER_S
    v_alpha = (entry.pm_ra_cosdec_mas_yr / entry.parallax_mas) * K_KM_PER_S
    v_delta = (entry.pm_dec_mas_yr / entry.parallax_mas) * K_KM_PER_S
    v_r = entry.radial_velocity_m_s

    # 3D Space velocity vector v_* in BCRS (m/s)
    v_vec = v_r * u0 + v_alpha * p0 + v_delta * q0

    # Linear space velocity propagation over transit epoch
    # delta_t in seconds
    delta_t_days = (epoch_jd_tdb - entry.epoch_jd) + jd_fraction
    delta_t_sec = delta_t_days * SEC_PER_DAY

    # Position at target epoch
    r_target_vec = r0_vec + v_vec * delta_t_sec

    dist_meters = float(np.linalg.norm(r_target_vec))
    from relativistic_engine.constants import LIGHT_YEAR
    dist_ly = dist_meters / LIGHT_YEAR

    return StarStateBCRS(
        name=entry.name,
        position=r_target_vec,
        velocity=v_vec,
        epoch_jd=epoch_jd_tdb + jd_fraction,
        distance_meters=dist_meters,
        distance_light_years=dist_ly,
    )
