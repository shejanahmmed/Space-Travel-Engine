"""3D Relativistic Worldline Propagator using adaptive 8th-order Runge-Kutta.

Propagates the full 7-dimensional relativistic state vector:
    y(t) = [rx, ry, rz, vx, vy, vz, Delta]
where r is BCRS Cartesian position in meters, v is BCRS coordinate velocity in m/s,
and Delta(t) = t - tau(t) is the coordinate time deficit in seconds.

Authoritative Standards:
- IAU 2000 Resolution B1.3 / IAU 2006 Resolution 3 (BCRS metric).
- Hairer, E., Norsett, S. P., Wanner, G., "Solving Ordinary Differential Equations I:
  Nonstiff Problems", Springer Series in Computational Mathematics (DOP853 algorithm).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence
import math
import numpy as np
from scipy.integrate import solve_ivp
from jplephem.spk import SPK

from relativistic_engine.constants import C_LIGHT, SEC_PER_DAY
from relativistic_engine.physics.dynamics import proper_to_coordinate_acceleration
from relativistic_engine.physics.metrics import coordinate_time_deficit_rate
from relativistic_engine.physics.potential import (
    solar_system_potential,
    solar_system_gravitational_acceleration,
)


@dataclass(frozen=True)
class TrajectoryResult3D:
    """Numerical solution of a 3D relativistic worldline.

    Attributes
    ----------
    t : np.ndarray
        Coordinate time grid in seconds.
    r : np.ndarray
        BCRS position array of shape (N, 3) in meters.
    v : np.ndarray
        BCRS velocity array of shape (N, 3) in m/s.
    time_deficit : np.ndarray
        Coordinate time deficit Delta(t) = t - tau(t) in seconds.
    tau : np.ndarray
        Proper time array tau(t) = t - Delta(t) in seconds.
    status : int
        Solver termination status (0 = success).
    message : str
        Solver termination message.
    nfev : int
        Total number of right-hand side evaluations.
    """

    t: np.ndarray
    r: np.ndarray
    v: np.ndarray
    time_deficit: np.ndarray
    tau: np.ndarray
    status: int
    message: str
    nfev: int


ThrustFunction = Callable[[float, np.ndarray, np.ndarray], np.ndarray]
GravityFunction = Callable[[np.ndarray, np.ndarray, float], tuple[np.ndarray, float]]


def propagate_trajectory_3d(
    r0: np.ndarray | Sequence[float],
    v0: np.ndarray | Sequence[float],
    t_span: tuple[float, float],
    *,
    thrust_func: Optional[ThrustFunction] = None,
    epoch_jd_tdb: Optional[float] = None,
    spk: Optional[SPK] = None,
    gravitational_bodies: Optional[Sequence[str]] = None,
    include_1pn: bool = True,
    include_zonals: bool = False,
    max_zonal_degree: int = 4,
    include_frame_dragging: bool = False,
    srp_params: Optional[SRPParameters] = None,
    include_radiation_reaction: bool = False,
    spacecraft_mass_kg: float = 1000.0,
    ism_conditions: Optional[Any] = None,
    ism_frontal_area: float = 0.0,
    ism_drag_coefficient: float = 1.0,
    custom_gravity_func: Optional[GravityFunction] = None,
    delta0: float = 0.0,
    t_eval: Optional[np.ndarray] = None,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    max_step: float = np.inf,
) -> TrajectoryResult3D:
    """Integrate 3D relativistic equations of motion and BCRS metric proper time.

    Parameters
    ----------
    r0 : Sequence[float]
        Initial position [x, y, z] in meters relative to SSB.
    v0 : Sequence[float]
        Initial velocity [vx, vy, vz] in m/s relative to SSB.
    t_span : tuple[float, float]
        Integration interval (t_start, t_end) in coordinate seconds.
    thrust_func : Optional[ThrustFunction], optional
        Callable returning proper acceleration vector [ax, ay, az] in m/s^2 as
        measured in the spacecraft's instantaneous co-moving rest frame.
        Signature: thrust_func(t, r, v) -> a_proper.
        If None, flight is ballistic (zero thrust).
    epoch_jd_tdb : Optional[float], optional
        Julian Date (TDB) corresponding to coordinate time t = 0.
        Required when using NASA/JPL ephemerides for gravity.
    spk : Optional[SPK], optional
        Pre-loaded NASA/JPL SPK kernel handle.
    gravitational_bodies : Optional[Sequence[str]], optional
        Names of celestial bodies to include in the potential field.
    include_1pn : bool, optional
        Whether to include 1PN general relativistic solar acceleration.
    include_zonals : bool, optional
        Whether to evaluate non-spherical zonal harmonics (J2 through J4).
    max_zonal_degree : int, optional
        Maximum zonal harmonic degree when include_zonals is True (2 to 4).
    include_frame_dragging : bool, optional
        Whether to evaluate Lense-Thirring gravitomagnetic acceleration.
    srp_params : Optional[SRPParameters], optional
        Solar Radiation Pressure parameters (Cr, area, mass, occulting bodies).
    include_radiation_reaction : bool, optional
        Whether to evaluate 2.5PN quadrupole gravitational radiation reaction damping.
    spacecraft_mass_kg : float, optional
        Spacecraft wet mass in kg (required when radiation reaction is enabled).
    custom_gravity_func : Optional[GravityFunction], optional
        Optional override returning (a_grav, potential) for idealized test cases.
        Signature: custom_gravity_func(r, v, t) -> (a_grav_vec, scalar_w).
    delta0 : float, optional
        Initial coordinate time deficit t - tau at t_start (default: 0.0).
    t_eval : Optional[np.ndarray], optional
        Specific coordinate times at which to store the solution.
    rtol : float, optional
        Relative integration tolerance (default: 1e-11).
    atol : float, optional
        Absolute integration tolerance (default: 1e-12).
    max_step : float, optional
        Maximum integrator step size in seconds.

    Returns
    -------
    TrajectoryResult3D
        Full trajectory trajectory solution including positions, velocities,
        and proper times.
    """
    from relativistic_engine.physics.perturbations import compute_srp_acceleration
    from relativistic_engine.physics.post_newtonian import compute_2_5pn_radiation_reaction
    from relativistic_engine.constants import GM_SUN, G_NEWTON

    r_init = np.asarray(r0, dtype=np.float64)
    v_init = np.asarray(v0, dtype=np.float64)

    # Initial state vector: [rx, ry, rz, vx, vy, vz, delta]
    y0 = np.concatenate([r_init, v_init, [float(delta0)]])

    # Guard margin against unphysical superluminal steps in trial stages
    # Must use relative epsilon scaling: C_LIGHT * (1 - 1e-12) because 1e-10 is below float64 ULP at 3e8
    v_speed_limit = C_LIGHT * (1.0 - 1.0e-12)


    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        r = y[0:3]
        v = y[3:6]

        # Guard trial step velocity norm
        v_norm = float(np.linalg.norm(v))
        if v_norm >= v_speed_limit:
            v_eval = v * (v_speed_limit / v_norm)
        else:
            v_eval = v

        # 1. Gravitational acceleration and potential
        if custom_gravity_func is not None:
            a_grav, w_pot = custom_gravity_func(r, v_eval, t)
            a_srp = np.zeros(3, dtype=np.float64)
            a_25pn = np.zeros(3, dtype=np.float64)
        elif epoch_jd_tdb is not None:
            # 2-part Julian Date: base epoch + elapsed days
            jd_frac = t / SEC_PER_DAY
            w_pot = solar_system_potential(
                r,
                epoch_jd_tdb,
                spk=spk,
                jd_fraction=jd_frac,
                bodies=gravitational_bodies,
                include_zonals=include_zonals,
                max_zonal_degree=max_zonal_degree,
            )
            a_grav = solar_system_gravitational_acceleration(
                r,
                v_eval,
                epoch_jd_tdb,
                spk=spk,
                jd_fraction=jd_frac,
                bodies=gravitational_bodies,
                include_1pn=include_1pn,
                include_zonals=include_zonals,
                max_zonal_degree=max_zonal_degree,
                include_frame_dragging=include_frame_dragging,
            )
            if srp_params is not None:
                a_srp = compute_srp_acceleration(
                    r,
                    epoch_jd_tdb,
                    srp_params,
                    spk=spk,
                    jd_fraction=jd_frac,
                )
            else:
                a_srp = np.zeros(3, dtype=np.float64)

            if include_radiation_reaction:
                sun_state = get_body_barycentric_state("sun", epoch_jd_tdb, spk=spk, jd_fraction=jd_frac)
                r_sun_rel = r - sun_state.position
                v_sun_rel = v_eval - sun_state.velocity
                m_sun = GM_SUN / G_NEWTON
                a_25pn = compute_2_5pn_radiation_reaction(r_sun_rel, v_sun_rel, m1=m_sun, m2=spacecraft_mass_kg)
            else:
                a_25pn = np.zeros(3, dtype=np.float64)
        else:
            # Flat Minkowski spacetime
            w_pot = 0.0
            a_grav = np.zeros(3, dtype=np.float64)
            a_srp = np.zeros(3, dtype=np.float64)
            a_25pn = np.zeros(3, dtype=np.float64)

        # 2. Proper thrust acceleration converted to coordinate acceleration
        if thrust_func is not None:
            a_prop = np.asarray(thrust_func(t, r, v_eval), dtype=np.float64)
            a_thrust = proper_to_coordinate_acceleration(v_eval, a_prop)
        else:
            a_thrust = np.zeros(3, dtype=np.float64)

        # 3. Interstellar Medium (ISM) relativistic ram drag
        if ism_conditions is not None and ism_frontal_area > 0.0:
            from relativistic_engine.physics.ism import compute_ism_drag_acceleration
            rho_val = (
                ism_conditions.total_mass_density
                if hasattr(ism_conditions, "total_mass_density")
                else float(ism_conditions)
            )
            a_ism = compute_ism_drag_acceleration(
                velocity_vector=v_eval,
                ship_mass=spacecraft_mass_kg,
                frontal_area=ism_frontal_area,
                rho_ism=rho_val,
                c_d=ism_drag_coefficient,
            )
        else:
            a_ism = np.zeros(3, dtype=np.float64)

        # 4. Coordinate acceleration (Gravity + Thrust + SRP + 2.5PN Radiation Damping + ISM Drag)
        dv_dt = a_grav + a_thrust + a_srp + a_25pn + a_ism

        # 5. Numerically stabilized rate of coordinate time deficit d(t - tau)/dt
        d_delta_dt = coordinate_time_deficit_rate(v_eval, w_pot)

        return np.concatenate([v, dv_dt, [d_delta_dt]])

    sol = solve_ivp(
        rhs,
        t_span,
        y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        t_eval=t_eval,
    )

    t_arr = sol.t
    r_arr = sol.y[0:3, :].T
    v_arr = sol.y[3:6, :].T
    delta_arr = sol.y[6, :]
    tau_arr = t_arr - delta_arr

    return TrajectoryResult3D(
        t=t_arr,
        r=r_arr,
        v=v_arr,
        time_deficit=delta_arr,
        tau=tau_arr,
        status=sol.status,
        message=sol.message,
        nfev=sol.nfev,
    )


from relativistic_engine.physics.eih import propagate_eih_nbody_system  # noqa: E402

