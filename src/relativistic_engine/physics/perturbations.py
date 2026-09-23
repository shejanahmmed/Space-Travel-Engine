"""Non-gravitational orbital perturbations module: Solar Radiation Pressure (SRP) and shadowing.

Implements high-fidelity photon momentum transfer dynamics on interplanetary spacecraft:
1. Cannonball Solar Radiation Pressure (SRP) with inverse-square heliocentric distance scaling.
2. Dual-cone (umbra and penumbra) geometric planetary shadow model with continuous disk intersection.
3. Multi-body occultation support (Earth, Moon, Mars, etc.).

Authoritative References:
- Kopp, G., & Lean, J. L. (2011), "A new, lower value of total solar irradiance: Evidence and
  climate significance", Geophys. Res. Lett. 38, L01706 (SORCE / TSIS-1 reference 1361 W/m^2).
- IAU 2015 Resolution B3: Nominal Solar System body radii and astronomical constants.
- Montenbruck, O., & Gill, E. (2000), "Satellite Orbits: Models, Methods and Applications",
  Springer, Section 3.4 (Radiation Pressure and Shadow Models).
- Vallado, D. A. (2013), "Fundamentals of Astrodynamics and Applications", 4th ed., Microcosm Press.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence
import numpy as np
from jplephem.spk import SPK

from relativistic_engine.constants import (
    AU,
    C_LIGHT,
    GM_SUN,
)
from relativistic_engine.physics.potential import (
    STANDARD_BODY_RADIUS,
)
from relativistic_engine.ephemeris.barycentric import get_body_barycentric_state

# Authoritative Total Solar Irradiance (TSI) at 1 AU (W / m^2, SORCE / TSIS-1 standard)
SOLAR_IRRADIANCE_1AU: float = 1361.0

# Radiation pressure of sunlight at 1 AU: P0 = S0 / c (N / m^2)
SOLAR_RADIATION_PRESSURE_1AU: float = SOLAR_IRRADIANCE_1AU / C_LIGHT  # ~ 4.539822e-6 N / m^2


@dataclass(frozen=True)
class SRPParameters:
    """Spacecraft parameters for Solar Radiation Pressure modeling.

    Attributes:
        cr: Radiation pressure coefficient (dimensionless, 1.0 <= cr <= 2.0).
            cr = 1.0 for perfect blackbody absorption,
            cr = 2.0 for total specular reflection.
            Typical values: 1.2 to 1.5.
        area_m2: Effective sun-facing cross-sectional area in square meters (A > 0).
        mass_kg: Spacecraft wet mass in kilograms (m > 0).
        occulting_bodies: Optional sequence of planetary body names that can cast shadows
            (e.g., ['earth', 'moon', 'mars']). Defaults to ['earth', 'moon'].
    """

    cr: float = 1.2
    area_m2: float = 10.0
    mass_kg: float = 1000.0
    occulting_bodies: tuple[str, ...] = ("earth", "moon")

    def __post_init__(self) -> None:
        if self.cr < 0.0:
            raise ValueError(f"Radiation pressure coefficient cr must be non-negative, got {self.cr}")
        if self.area_m2 <= 0.0:
            raise ValueError(f"Cross-sectional area must be strictly positive, got {self.area_m2}")
        if self.mass_kg <= 0.0:
            raise ValueError(f"Spacecraft mass must be strictly positive, got {self.mass_kg}")


def compute_shadow_factor_dual_cone(
    r_sc_bcrs: np.ndarray | Sequence[float],
    r_sun_bcrs: np.ndarray | Sequence[float],
    r_body_bcrs: np.ndarray | Sequence[float],
    *,
    r_sun_radius: float = 6.957e8,
    r_body_radius: float = 6.371e6,
) -> float:
    """Compute geometric shadow factor nu in [0, 1] using exact dual-cone disk intersection.

    Evaluates apparent angular disk radii of the Sun and occulting body as seen from
    the spacecraft, calculating the exact overlapping area of the two spherical caps
    projected on the local celestial sphere.

    Shadow regimes:
    - nu = 1.0: Full sunlight (no occultation)
    - nu = 0.0: Total eclipse (umbra)
    - 0 < nu < 1.0: Partial eclipse (penumbra) or annular eclipse

    Args:
        r_sc_bcrs: Spacecraft position vector [x, y, z] in meters relative to SSB.
        r_sun_bcrs: Sun position vector [x, y, z] in meters relative to SSB.
        r_body_bcrs: Occulting body center [x, y, z] in meters relative to SSB.
        r_sun_radius: Physical radius of the Sun in meters (default: 6.957e8 m).
        r_body_radius: Physical radius of the occulting body in meters.

    Returns:
        Shadow factor nu in [0.0, 1.0] (fraction of solar disk area visible).
    """
    r_sc = np.asarray(r_sc_bcrs, dtype=np.float64)
    r_sun = np.asarray(r_sun_bcrs, dtype=np.float64)
    r_body = np.asarray(r_body_bcrs, dtype=np.float64)

    # Relative vectors from spacecraft to Sun and to occulting body
    d_sun = r_sun - r_sc
    d_body = r_body - r_sc

    dist_sun = float(np.linalg.norm(d_sun))
    dist_body = float(np.linalg.norm(d_body))

    if dist_sun <= r_sun_radius or dist_body <= r_body_radius:
        # Spacecraft inside physical body surface
        return 0.0

    # Unit direction vectors
    u_sun = d_sun / dist_sun
    u_body = d_body / dist_body

    cos_sep = float(np.dot(u_sun, u_body))
    cos_sep = max(-1.0, min(1.0, cos_sep))

    # If occulting body is in the opposite hemisphere from the Sun, no eclipse is possible
    if cos_sep <= 0.0:
        return 1.0

    # If spacecraft is closer to Sun than body along line of sight, body cannot shadow spacecraft
    if float(np.dot(d_body, u_sun)) > dist_sun:
        return 1.0

    # Apparent angular radii of Sun and occulting body as seen from spacecraft
    sin_theta_sun = min(1.0, r_sun_radius / dist_sun)
    sin_theta_body = min(1.0, r_body_radius / dist_body)

    theta_sun = math.asin(sin_theta_sun)
    theta_body = math.asin(sin_theta_body)

    # Apparent angular separation between centers of Sun and body
    theta = math.acos(cos_sep)

    # Case 1: Disjoint disks -> Full sunlight
    if theta >= theta_sun + theta_body:
        return 1.0

    # Case 2: Body completely covers the Sun -> Total eclipse (Umbra)
    if theta_body >= theta_sun and theta <= theta_body - theta_sun:
        return 0.0

    # Case 3: Sun completely encloses the body -> Annular eclipse
    if theta_sun > theta_body and theta <= theta_sun - theta_body:
        unocculted_fraction = 1.0 - (theta_body / theta_sun) ** 2
        return max(0.0, min(1.0, unocculted_fraction))

    # Case 4: Partial overlap (Penumbra) -> Exact intersection area of two circular segments
    # Area of overlap of two disks of radii r1 = theta_sun, r2 = theta_body at separation d = theta
    r1 = theta_sun
    r2 = theta_body
    d = theta

    c1 = (d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1)
    c2 = (d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2)

    c1 = max(-1.0, min(1.0, c1))
    c2 = max(-1.0, min(1.0, c2))

    alpha1 = math.acos(c1)
    alpha2 = math.acos(c2)

    # Intersecting area: A_overlap = r1^2 * (alpha1 - sin(alpha1)*cos(alpha1)) + r2^2 * (alpha2 - sin(alpha2)*cos(alpha2))
    a_overlap = (r1 * r1) * (alpha1 - math.sin(alpha1) * c1) + (r2 * r2) * (alpha2 - math.sin(alpha2) * c2)

    sun_disk_area = math.pi * r1 * r1
    occulted_fraction = a_overlap / sun_disk_area

    nu = 1.0 - occulted_fraction
    return max(0.0, min(1.0, nu))


def compute_srp_acceleration(
    r_sc_bcrs: np.ndarray | Sequence[float],
    jd_tdb: float,
    params: SRPParameters,
    spk: Optional[SPK] = None,
    *,
    jd_fraction: float = 0.0,
) -> np.ndarray:
    """Compute 3D Cartesian Solar Radiation Pressure acceleration vector in BCRS.

    Formula:
        a_SRP = nu * P0 * (AU / r_sun)^2 * (Cr * A / m) * u_sun
    where u_sun = (r_sc - r_sun) / ||r_sc - r_sun|| is directed away from the Sun.

    Args:
        r_sc_bcrs: Cartesian position of spacecraft [x, y, z] in meters relative to SSB.
        jd_tdb: Epoch in Julian Date (TDB scale).
        params: Spacecraft SRPParameters (Cr, area, mass, occulting bodies).
        spk: Optional pre-loaded SPK kernel.
        jd_fraction: Optional fractional day component.

    Returns:
        Cartesian acceleration vector [ax, ay, az] in m/s^2.
    """
    r_sc = np.asarray(r_sc_bcrs, dtype=np.float64)

    # Retrieve Sun state in BCRS
    sun_state = get_body_barycentric_state("sun", jd_tdb, spk=spk, jd_fraction=jd_fraction)
    r_sun = sun_state.position
    r_sun_radius = STANDARD_BODY_RADIUS.get("sun", 6.957e8)

    # Vector from Sun to spacecraft
    d_sun = r_sc - r_sun
    dist_sun = float(np.linalg.norm(d_sun))

    if dist_sun <= r_sun_radius:
        return np.zeros(3, dtype=np.float64)

    u_sun_away = d_sun / dist_sun

    # Evaluate shadow factor across all declared occulting bodies
    nu = 1.0
    for body_name in params.occulting_bodies:
        b_name = body_name.lower()
        if b_name == "sun":
            continue
        body_state = get_body_barycentric_state(b_name, jd_tdb, spk=spk, jd_fraction=jd_fraction)
        body_radius = STANDARD_BODY_RADIUS.get(b_name, 1.0e6)
        nu_body = compute_shadow_factor_dual_cone(
            r_sc,
            r_sun,
            body_state.position,
            r_sun_radius=r_sun_radius,
            r_body_radius=body_radius,
        )
        nu = min(nu, nu_body)
        if nu <= 0.0:
            return np.zeros(3, dtype=np.float64)

    # Cannonball pressure scaling: P(r) = P0 * (AU / dist_sun)^2
    scaling = (AU / dist_sun) ** 2
    a_mag = nu * SOLAR_RADIATION_PRESSURE_1AU * scaling * (params.cr * params.area_m2 / params.mass_kg)

    return a_mag * u_sun_away
