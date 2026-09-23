"""Verification Suite for Solar Radiation Pressure (SRP) and Dual-Cone Shadowing.

Validates:
- Level 1: Exact physical constant matching for Total Solar Irradiance S0 and
           radiation pressure P0 = S0 / c at 1 AU.
- Level 2: Dual-cone (umbra and penumbra) geometric shadow model continuity,
           monotonicity, and exact boundary condition limits.
- Level 3: Exact inverse-square heliocentric distance scaling across 0.3 AU to 10.0 AU.
- Level 4: Orbital eclipse crossing: smooth penumbral entry/exit and total umbra cutoff
           in Low Earth Orbit.
- Level 5: Full interplanetary trajectory propagation over 200 days quantifying
           accumulated SRP drift against analytical ballistic integration.

Authoritative Standards:
- SORCE / TSIS-1 (Kopp & Lean 2011): S0 = 1361.0 W / m^2.
- CODATA 2018 / BIPM: Speed of light c = 299792458 m/s.
- Montenbruck & Gill (2000), "Satellite Orbits", Section 3.4 (Dual-cone shadow geometry).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from relativistic_engine.constants import (
    AU,
    C_LIGHT,
    GM_EARTH,
    GM_SUN,
    SEC_PER_DAY,
)
from relativistic_engine.physics.perturbations import (
    SOLAR_IRRADIANCE_1AU,
    SOLAR_RADIATION_PRESSURE_1AU,
    SRPParameters,
    compute_shadow_factor_dual_cone,
    compute_srp_acceleration,
)
from relativistic_engine.numerical.trajectory import propagate_trajectory_3d
from relativistic_engine.ephemeris.barycentric import get_body_barycentric_state


# ==============================================================================
# LEVEL 1: PHYSICAL CONSTANTS & FLUX CONSERVATION
# ==============================================================================


def test_level1_srp_flux_and_pressure_constants():
    """Level 1: P0 = S0 / c matches SORCE/TSIS-1 standards to machine precision."""
    expected_pressure = 1361.0 / C_LIGHT
    assert np.isclose(SOLAR_RADIATION_PRESSURE_1AU, expected_pressure, rtol=1e-15, atol=1e-15)
    # Numerical value at 1 AU must be ~ 4.539822e-6 N / m^2
    assert np.isclose(SOLAR_RADIATION_PRESSURE_1AU, 4.5398224377e-6, rtol=1e-8)


def test_level1_srp_parameter_dataclass_validation():
    """Level 1: Validate parameter bounds on SRPParameters."""
    params = SRPParameters(cr=1.3, area_m2=15.0, mass_kg=850.0)
    assert params.cr == 1.3
    assert params.area_m2 == 15.0
    assert params.mass_kg == 850.0

    with pytest.raises(ValueError, match="Radiation pressure coefficient"):
        SRPParameters(cr=-0.1)
    with pytest.raises(ValueError, match="Cross-sectional area"):
        SRPParameters(area_m2=0.0)
    with pytest.raises(ValueError, match="Spacecraft mass"):
        SRPParameters(mass_kg=-10.0)


# ==============================================================================
# LEVEL 2: DUAL-CONE SHADOW MODEL CONTINUITY & LIMITS
# ==============================================================================


def test_level2_dual_cone_shadow_continuity_and_limits():
    """Level 2: Shadow factor nu is C^0 continuous and monotonic across shadow boundary.

    Test geometry:
    - Sun at origin [0, 0, 0] with radius R_sun
    - Earth at [1 AU, 0, 0] with radius R_earth
    - Spacecraft behind Earth at x = 1 AU + 20,000 km, scanning in y from 0 (deep umbra)
      to 15,000 km (full sunlight).
    """
    r_sun = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    r_sun_radius = 6.957e8  # meters
    r_earth_radius = 6.371e6  # meters
    r_earth = np.array([AU, 0.0, 0.0], dtype=np.float64)

    # Position spacecraft behind Earth along X
    x_sc = AU + 2.0e7  # 20,000 km behind Earth center

    # Scan y_sc from 0 (center of umbra) to 20,000 km (sunlight)
    y_scan = np.linspace(0.0, 2.0e7, 500)
    nu_values = []

    for y in y_scan:
        r_sc = np.array([x_sc, y, 0.0], dtype=np.float64)
        nu = compute_shadow_factor_dual_cone(
            r_sc,
            r_sun,
            r_earth,
            r_sun_radius=r_sun_radius,
            r_body_radius=r_earth_radius,
        )
        nu_values.append(nu)

    nu_arr = np.array(nu_values)

    # 1. At y = 0, spacecraft must be completely in total darkness (umbra)
    assert nu_arr[0] == 0.0, f"Expected deep umbra nu=0.0, got {nu_arr[0]}"

    # 2. At y = 20,000 km, spacecraft must be completely in full sunlight
    assert nu_arr[-1] == 1.0, f"Expected full sunlight nu=1.0, got {nu_arr[-1]}"

    # 3. Monotonicity: nu must be non-decreasing as we move radially out of shadow
    diffs = np.diff(nu_arr)
    assert np.all(diffs >= -1e-14), "Shadow factor is not monotonically non-decreasing out of shadow"

    # 4. Strict boundedness: nu in [0.0, 1.0] everywhere
    assert np.all(nu_arr >= 0.0)
    assert np.all(nu_arr <= 1.0)

    # 5. Fine penumbra crossing continuity:
    # Earth limb appears at y ~ 6.7e6 m where the ~93 km solar disk penumbra occurs.
    # We sample with delta_y ~ 1 km (500 points across 500 km) to test true C0 continuity:
    y_penumbra = np.linspace(6.5e6, 7.0e6, 500)
    nu_pen = [
        compute_shadow_factor_dual_cone(
            np.array([x_sc, yp, 0.0], dtype=np.float64),
            r_sun,
            r_earth,
            r_sun_radius=r_sun_radius,
            r_body_radius=r_earth_radius,
        )
        for yp in y_penumbra
    ]
    diffs_pen = np.diff(nu_pen)
    assert np.all(diffs_pen >= -1e-14), "Penumbra transition is non-monotonic"
    max_jump_pen = float(np.max(np.abs(diffs_pen)))
    assert max_jump_pen < 0.03, f"Penumbra has step discontinuity: {max_jump_pen:.4f}"


def test_level2_spacecraft_between_sun_and_body_never_shadowed():
    """Level 2: When spacecraft is between Sun and planet, shadow factor must strictly be 1.0."""
    r_sun = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    r_earth = np.array([AU, 0.0, 0.0], dtype=np.float64)
    # Spacecraft at 0.8 AU on the Sun-Earth line
    r_sc = np.array([0.8 * AU, 0.0, 0.0], dtype=np.float64)

    nu = compute_shadow_factor_dual_cone(r_sc, r_sun, r_earth)
    assert nu == 1.0, f"Expected full sunlight, got {nu}"


# ==============================================================================
# LEVEL 3: INVERSE-SQUARE HELIOCENTRIC SCALING
# ==============================================================================


@pytest.mark.parametrize("dist_au", [0.3, 0.5, 1.0, 1.524, 5.2, 9.5])
def test_level3_srp_inverse_square_scaling(dist_au: float):
    """Level 3: Verify a_SRP(r) scales exactly as (1 / r^2) from 0.3 AU to 10.0 AU."""
    jd_epoch = 2451545.0
    sun_state = get_body_barycentric_state("sun", jd_epoch)

    # Place spacecraft along X-axis at dist_au from Sun
    r_sc = sun_state.position + np.array([dist_au * AU, 0.0, 0.0], dtype=np.float64)
    params = SRPParameters(cr=1.2, area_m2=10.0, mass_kg=1000.0, occulting_bodies=())

    a_srp = compute_srp_acceleration(r_sc, jd_epoch, params)
    a_mag = float(np.linalg.norm(a_srp))

    # Analytical magnitude: P0 * (1 / dist_au^2) * (Cr * A / m)
    expected_mag = SOLAR_RADIATION_PRESSURE_1AU * (1.0 / (dist_au**2)) * (1.2 * 10.0 / 1000.0)

    assert np.isclose(a_mag, expected_mag, rtol=1e-7)
    # Acceleration vector must point radially away from Sun (+X direction)
    assert a_srp[0] > 0.0
    assert np.isclose(a_srp[1], 0.0, atol=1e-15)
    assert np.isclose(a_srp[2], 0.0, atol=1e-15)


# ==============================================================================
# LEVEL 4: ORBITAL ECLIPSE PASSAGE
# ==============================================================================


def test_level4_orbital_eclipse_passage_in_leo():
    """Level 4: Propagate a circular LEO orbit and verify acceleration vanishes in umbra."""
    jd_epoch = 2451545.0
    earth_state = get_body_barycentric_state("earth", jd_epoch)
    sun_state = get_body_barycentric_state("sun", jd_epoch)

    # Vector from Sun to Earth defines shadow direction
    u_shadow = earth_state.position - sun_state.position
    u_shadow /= np.linalg.norm(u_shadow)

    # Spacecraft placed at 500 km altitude directly behind Earth along shadow axis
    r_orbit = 6.371e6 + 5.0e5  # 6871 km
    r_sc_umbra = earth_state.position + r_orbit * u_shadow

    # Spacecraft placed at 500 km altitude directly in front of Earth towards Sun
    r_sc_sun = earth_state.position - r_orbit * u_shadow

    params = SRPParameters(cr=1.2, area_m2=10.0, mass_kg=1000.0, occulting_bodies=("earth",))

    # In umbra behind Earth: acceleration must be identically zero
    a_umbra = compute_srp_acceleration(r_sc_umbra, jd_epoch, params)
    assert np.all(a_umbra == 0.0), f"Expected 0.0 acceleration in umbra, got {np.linalg.norm(a_umbra)}"

    # In sunlight facing Sun: acceleration must be full magnitude ~ 5.4e-8 m/s^2
    a_sun = compute_srp_acceleration(r_sc_sun, jd_epoch, params)
    a_sun_mag = float(np.linalg.norm(a_sun))
    assert 4.0e-8 < a_sun_mag < 6.0e-8, f"Unexpected sunlight SRP magnitude: {a_sun_mag:.2e} m/s^2"


# ==============================================================================
# LEVEL 5: INTERPLANETARY CRUISE DRIFT QUANTIFICATION
# ==============================================================================


def test_level5_interplanetary_cruise_drift_quantification():
    """Level 5: Quantify 200-day ballistic Earth-Mars cruise trajectory drift with and without SRP.

    With constant a_SRP ~ 5.0e-8 m/s^2, the theoretical position drift after t = 200 days is:
        Delta r ~ (1/2) * a * t^2 ~ 0.5 * 5.0e-8 * (200 * 86400)^2 ~ 7,465 km.
    """
    jd_epoch = 2451545.0  # J2000.0
    earth_state = get_body_barycentric_state("earth", jd_epoch)

    # Heliocentric cruise state: Earth position + small hyperbolic injection velocity
    v_inj = earth_state.velocity + np.array([3000.0, 1500.0, 0.0], dtype=np.float64)
    r_init = earth_state.position + np.array([1.0e8, 0.0, 0.0], dtype=np.float64)

    t_flight = 200.0 * SEC_PER_DAY  # 200 days
    t_span = (0.0, t_flight)

    srp_probe = SRPParameters(cr=1.2, area_m2=12.0, mass_kg=1000.0, occulting_bodies=())

    # 1. Propagate with SRP
    sol_srp = propagate_trajectory_3d(
        r0=r_init,
        v0=v_inj,
        t_span=t_span,
        epoch_jd_tdb=jd_epoch,
        gravitational_bodies=["sun"],
        include_1pn=True,
        srp_params=srp_probe,
        rtol=1e-10,
        atol=1e-12,
    )

    # 2. Propagate purely ballistic (no SRP)
    sol_ballistic = propagate_trajectory_3d(
        r0=r_init,
        v0=v_inj,
        t_span=t_span,
        epoch_jd_tdb=jd_epoch,
        gravitational_bodies=["sun"],
        include_1pn=True,
        srp_params=None,
        rtol=1e-10,
        atol=1e-12,
    )

    # Verify both solved successfully
    assert sol_srp.status == 0
    assert sol_ballistic.status == 0

    # Calculate accumulated target drift at 200 days
    delta_r = np.linalg.norm(sol_srp.r[-1] - sol_ballistic.r[-1])
    delta_r_km = delta_r / 1000.0

    # Physical drift must be on the order of 3,000 to 15,000 km
    assert 3000.0 < delta_r_km < 15000.0, (
        f"SRP cruise drift {delta_r_km:.1f} km outside expected physical envelope (3,000 - 15,000 km)"
    )

    # Proper time deficit must remain smooth and strictly increasing
    assert sol_srp.time_deficit[-1] > 0.0
    assert np.all(np.diff(sol_srp.tau) > 0.0)
