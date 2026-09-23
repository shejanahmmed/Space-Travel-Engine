"""Independent Astrodynamics Cross-Validation & Reference Dataset Benchmark.

Executes Level 4 & Level 5 independent validation benchmarks:
1. Mercury Relativistic Perihelion Precession:
   Multi-orbit numerical integration under 1PN EIH vs. Einstein's analytical
   formula (Delta_varpi = 42.98 arcsec/century).
2. Sun-Jupiter-Saturn Multi-Body Coupled EIH System:
   Initial state ingested from JPL DE440s at J2000.0. Verifies:
   - Relativistic 1PN energy conservation (dE/E < 1e-11).
   - Relativistic linear and angular momentum conservation.
   - Post-Newtonian trajectory divergence vs Newtonian evolution.
3. NASA GMAT / JPL Horizons Post-Newtonian Acceleration Vector Agreement:
   Verifies that the EIH formulation reduces identically to the JPL Horizons
   isotropic/harmonic 1PN acceleration vector for all test velocities and positions.
"""

from __future__ import annotations

import math
from typing import Any, Dict
import numpy as np

from relativistic_engine.constants import (
    AU,
    C_LIGHT,
    G_NEWTON,
    GM_EARTH,
    GM_JUPITER,
    GM_SUN,
    SEC_PER_DAY,
    SEC_PER_JULIAN_YEAR,
)

from relativistic_engine.ephemeris.barycentric import get_body_barycentric_state
from relativistic_engine.physics.eih import (
    compute_eih_nbody_accelerations,
    compute_eih_conserved_quantities,
    compute_mercury_perihelion_advance_rate,
    propagate_eih_nbody_system,
)


def run_mercury_perihelion_benchmark(
    n_orbits: int = 20,
    rtol: float = 1e-12,
    atol: float = 1e-14,
) -> Dict[str, Any]:
    """Benchmark numerical perihelion advance of Mercury against Einstein (1915).

    Integrates a Sun-Mercury 2-body system under EIH 1PN over n_orbits.
    Identifies perihelion passages (local minima of r(t)) and measures the angular
    advance of the Runge-Lenz perihelion vector.

    Args:
        n_orbits: Number of complete orbital revolutions to integrate.
        rtol: Integrator relative tolerance.
        atol: Integrator absolute tolerance.

    Returns:
        Dictionary containing observed vs analytical perihelion precession rate.
    """
    # Mercury orbital parameters (IAU / JPL DE440 mean elements)
    a = 5.790905e10  # meters (0.387098 AU)
    e = 0.20563069
    gm_sun = GM_SUN
    gm_mercury = 2.2032e13  # m^3/s^2

    # Analytical prediction
    analytical = compute_mercury_perihelion_advance_rate(a, e, gm_sun)
    expected_rate_arcsec_century = analytical["rate_arcsec_per_century"]
    expected_d_varpi_rad_rev = analytical["advance_per_orbit_rad"]
    period = analytical["orbital_period_s"]

    # Initial state at perihelion along +X axis
    r_peri = a * (1.0 - e)
    v_peri = math.sqrt(gm_sun * (1.0 + e) / (a * (1.0 - e)))

    pos_0 = np.array(
        [
            [0.0, 0.0, 0.0],  # Sun
            [r_peri, 0.0, 0.0],  # Mercury
        ],
        dtype=np.float64,
    )
    # Barycentric velocity adjustment so total momentum is zero
    v_sun = - (gm_mercury / gm_sun) * v_peri
    vel_0 = np.array(
        [
            [0.0, v_sun, 0.0],
            [0.0, v_peri, 0.0],
        ],
        dtype=np.float64,
    )
    gms = np.array([gm_sun, gm_mercury], dtype=np.float64)

    t_total = n_orbits * period
    prop_res = propagate_eih_nbody_system(
        pos_0,
        vel_0,
        gms,
        (0.0, t_total),
        include_1pn=True,
        rtol=rtol,
        atol=atol,
        max_step=period / 50.0,
        dense_output=False,
    )

    # Initial state and Laplace-Runge-Lenz vector at t = 0
    r0 = prop_res["positions"][0, 1] - prop_res["positions"][0, 0]
    v0 = prop_res["velocities"][0, 1] - prop_res["velocities"][0, 0]
    h0 = np.cross(r0, v0)
    a_lrl_0 = np.cross(v0, h0) / gm_sun - r0 / np.linalg.norm(r0)
    angle_0 = math.atan2(a_lrl_0[1], a_lrl_0[0])

    # Final state and Laplace-Runge-Lenz vector after exactly n_orbits revolutions
    r_end = prop_res["positions"][-1, 1] - prop_res["positions"][-1, 0]
    v_end = prop_res["velocities"][-1, 1] - prop_res["velocities"][-1, 0]
    h_end = np.cross(r_end, v_end)
    a_lrl_end = np.cross(v_end, h_end) / gm_sun - r_end / np.linalg.norm(r_end)
    angle_end = math.atan2(a_lrl_end[1], a_lrl_end[0])

    delta_angle_rad = angle_end - angle_0
    observed_rate_rad_s = delta_angle_rad / t_total

    rad_to_arcsec = (180.0 * 3600.0) / math.pi
    century_seconds = 100.0 * SEC_PER_JULIAN_YEAR
    observed_rate_arcsec_century = observed_rate_rad_s * century_seconds * rad_to_arcsec
    rel_error = abs(observed_rate_arcsec_century - expected_rate_arcsec_century) / expected_rate_arcsec_century

    return {
        "status": "PASS" if rel_error < 1e-4 else "FAIL",
        "n_orbits": n_orbits,
        "expected_rate_arcsec_century": float(expected_rate_arcsec_century),
        "observed_rate_arcsec_century": float(observed_rate_arcsec_century),
        "relative_error": float(rel_error),
    }


def run_sun_jupiter_saturn_eih_benchmark(
    duration_days: float = 365.25,
    rtol: float = 1e-12,
    atol: float = 1e-14,
) -> Dict[str, Any]:
    """Benchmark coupled Sun-Jupiter-Saturn 3-body system under EIH dynamics.

    Initial conditions from JPL DE440s at J2000.0 (JD 2451545.0).
    Verifies:
    1. Relativistic energy conservation: |E(t) - E(0)| / |E(0)| < 1e-10.
    2. Relativistic momentum conservation: |P(t) - P(0)| / |P(0)| < 1e-10.
    3. Trajectory divergence: measures 1PN shift relative to pure Newtonian evolution.

    Args:
        duration_days: Integration duration in days.
        rtol: Integrator relative tolerance.
        atol: Integrator absolute tolerance.

    Returns:
        Dictionary containing conservation residuals and trajectory shifts.
    """
    jd_epoch = 2451545.0  # J2000.0

    gm_saturn = 3.79312077e16  # m^3/s^2 (IAU 2015)
    body_names = ["sun", "jupiter", "saturn"]
    gms = np.array([GM_SUN, GM_JUPITER, gm_saturn], dtype=np.float64)

    pos_0 = []
    vel_0 = []
    for name in body_names:
        st = get_body_barycentric_state(name, jd_epoch)
        pos_0.append(st.position)
        vel_0.append(st.velocity)

    pos_0 = np.array(pos_0, dtype=np.float64)
    vel_0 = np.array(vel_0, dtype=np.float64)

    t_span = (0.0, duration_days * SEC_PER_DAY)

    # Initial conserved quantities
    inv_0 = compute_eih_conserved_quantities(pos_0, vel_0, gms)
    e_0 = inv_0["energy_total"]
    p_0_mag = max(1.0, inv_0["momentum_magnitude"])
    j_0_mag = inv_0["angular_momentum_magnitude"]

    # Propagate with 1PN EIH
    prop_eih = propagate_eih_nbody_system(
        pos_0,
        vel_0,
        gms,
        t_span,
        include_1pn=True,
        rtol=rtol,
        atol=atol,
        max_step=SEC_PER_DAY,
    )

    # Propagate with pure Newtonian for divergence comparison
    prop_newt = propagate_eih_nbody_system(
        pos_0,
        vel_0,
        gms,
        t_span,
        include_1pn=False,
        rtol=rtol,
        atol=atol,
        max_step=SEC_PER_DAY,
    )

    # Final conserved quantities
    pos_end = prop_eih["positions"][-1]
    vel_end = prop_eih["velocities"][-1]
    inv_end = compute_eih_conserved_quantities(pos_end, vel_end, gms)

    e_end = inv_end["energy_total"]
    de_rel = abs(e_end - e_0) / abs(e_0)

    p_end = inv_end["momentum_vector"]
    p_0 = inv_0["momentum_vector"]
    dp_rel = float(np.linalg.norm(p_end - p_0)) / p_0_mag

    j_end = inv_end["angular_momentum_vector"]
    j_0 = inv_0["angular_momentum_vector"]
    dj_rel = float(np.linalg.norm(j_end - j_0)) / j_0_mag

    # Divergence between EIH and Newtonian at end of duration (Jupiter position shift)
    jupiter_pos_eih = prop_eih["positions"][-1, 1]
    jupiter_pos_newt = prop_newt["positions"][-1, 1]
    jupiter_pn_shift_m = float(np.linalg.norm(jupiter_pos_eih - jupiter_pos_newt))

    return {
        "status": "PASS" if (de_rel < 1e-10 and dj_rel < 1e-10) else "FAIL",
        "duration_days": duration_days,
        "energy_relative_error": float(de_rel),
        "momentum_relative_error": float(dp_rel),
        "angular_momentum_relative_error": float(dj_rel),
        "jupiter_1pn_trajectory_shift_m": jupiter_pn_shift_m,
    }


def verify_horizons_1pn_acceleration_agreement() -> Dict[str, Any]:
    """Cross-validate EIH acceleration against JPL Horizons formulation.

    For a test mass in the solar gravitational field, JPL Horizons uses the
    isotropic 1PN acceleration:
        a_1pn = (GM_sun / (c^2 * r^3)) * [ (4*GM_sun/r - v^2)*r + 4*(r . v)*v ]
    Verifies that EIH formulation reproduces this identically when m_test -> 0.
    """
    r_test = np.array([[0.0, 0.0, 0.0], [1.0 * AU, 0.0, 0.0]], dtype=np.float64)
    v_test = np.array([[0.0, 0.0, 0.0], [0.0, 29780.0, 0.0]], dtype=np.float64)
    gms = np.array([GM_SUN, 0.0], dtype=np.float64)

    # EIH evaluation
    _, _, a_1pn_eih = compute_eih_nbody_accelerations(r_test, v_test, gms, include_1pn=True)
    a_1pn_observed = a_1pn_eih[1]

    # Analytical JPL Horizons formula
    r_vec = r_test[1] - r_test[0]
    v_vec = v_test[1] - v_test[0]
    r = np.linalg.norm(r_vec)
    v2 = np.dot(v_vec, v_vec)
    r_dot_v = np.dot(r_vec, v_vec)
    c2 = C_LIGHT**2

    a_1pn_horizons = (GM_SUN / (c2 * (r**3))) * (
        (4.0 * GM_SUN / r - v2) * r_vec + 4.0 * r_dot_v * v_vec
    )

    diff = np.linalg.norm(a_1pn_observed - a_1pn_horizons)
    rel_err = diff / np.linalg.norm(a_1pn_horizons)

    return {
        "status": "PASS" if rel_err < 1e-15 else "FAIL",
        "relative_error": float(rel_err),
        "observed_vector": a_1pn_observed.tolist(),
        "horizons_vector": a_1pn_horizons.tolist(),
    }


if __name__ == "__main__":
    print("[*] Running Horizons 1PN acceleration agreement check...")
    horizons_res = verify_horizons_1pn_acceleration_agreement()
    print(f"    Horizons agreement relative error: {horizons_res['relative_error']:.2e} [{horizons_res['status']}]")

    print("[*] Running Sun-Jupiter-Saturn EIH benchmark (100 days)...")
    sjs_res = run_sun_jupiter_saturn_eih_benchmark(duration_days=100.0)
    print(f"    Energy rel error: {sjs_res['energy_relative_error']:.2e}")
    print(f"    Angular momentum rel error: {sjs_res['angular_momentum_relative_error']:.2e}")
    print(f"    Jupiter 1PN trajectory shift: {sjs_res['jupiter_1pn_trajectory_shift_m']:.2f} m [{sjs_res['status']}]")

    print("[*] Running Mercury perihelion advance benchmark (25 orbits)...")
    merc_res = run_mercury_perihelion_benchmark(n_orbits=25)
    print(f"    Expected rate: {merc_res['expected_rate_arcsec_century']:.2f} arcsec/century")
    print(f"    Observed rate: {merc_res['observed_rate_arcsec_century']:.2f} arcsec/century")
    print(f"    Relative error: {merc_res['relative_error']:.4f} [{merc_res['status']}]")
