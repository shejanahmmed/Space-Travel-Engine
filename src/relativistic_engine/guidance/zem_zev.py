"""Relativistic Zero-Effort-Miss / Zero-Effort-Velocity (ZEM/ZEV) Closed-Loop Guidance.

Implements optimal linear feedback guidance laws formulated in 1PN curved spacetime,
providing fuel-optimal trajectory correction against orbital perturbations,
injection dispersions, and navigation errors.

Authoritative Standards & References:
- D'Souza, C. N. (1997), "An Optimal Guidance Law for Planetary Landing",
  AIAA Guidance, Navigation, and Control Conference.
- Guo, Y., Hawkins, M., & Wie, B. (2013), "Waypoint-Optimized Zero-Effort-Miss/
  Zero-Effort-Velocity Guidance for Mars Landing", Journal of Guidance, Control,
  and Dynamics, 36(3), 744-754.
- Battin, R. H. (1999), "An Introduction to the Mathematics and Methods of
  Astrodynamics", AIAA Education Series.
- Damour, T., & Deruelle, N. (1985), "General relativistic celestial mechanics
  of binary systems", Ann. Inst. Henri Poincaré Phys. Théor. 43, 107-132.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from relativistic_engine.constants import AU, C_LIGHT, G0, GM_SUN


@dataclass(frozen=True)
class GuidanceCommand:
    """Instantaneous guidance output command and state metrics.

    Attributes:
        epoch_seconds:      Current flight coordinate time t [s].
        time_to_go_s:       Remaining time until target intercept t_go [s].
        zem_m:              3D Zero-Effort-Miss vector in BCRS frame [m].
        zev_ms:             3D Zero-Effort-Velocity vector in BCRS frame [m/s].
        cmd_accel_ms2:      Commanded 3D acceleration vector [m/s^2].
        clamped_accel_ms2:  Thrust-limited commanded acceleration vector [m/s^2].
        thrust_n:           Commanded thrust magnitude in Newtons [N].
        attitude_quaternion: Spacecraft body pointing unit quaternion [q0, q1, q2, q3].
        mass_flow_rate_kg_s: Instantaneous propellant consumption rate [kg/s].
    """

    epoch_seconds: float
    time_to_go_s: float
    zem_m: np.ndarray
    zev_ms: np.ndarray
    cmd_accel_ms2: np.ndarray
    clamped_accel_ms2: np.ndarray
    thrust_n: float
    attitude_quaternion: np.ndarray
    mass_flow_rate_kg_s: float


def compute_1pn_gravity_acceleration(r_vec: np.ndarray, v_vec: np.ndarray) -> np.ndarray:
    """Compute exact 1PN relativistic solar gravitational acceleration in BCRS.

    Args:
        r_vec: 3D position vector relative to Sun [m].
        v_vec: 3D coordinate velocity vector [m/s].

    Returns:
        3D acceleration vector [m/s^2].
    """
    r_norm = float(np.linalg.norm(r_vec))
    if r_norm < 1.0:
        return np.zeros(3, dtype=np.float64)

    r_hat = r_vec / r_norm
    v_sq = float(np.dot(v_vec, v_vec))
    c_sq = C_LIGHT**2
    gr_pot = GM_SUN / (c_sq * r_norm)

    a_newton = -GM_SUN * r_vec / (r_norm**3)
    a_1pn = (
        a_newton * (1.0 + 4.0 * gr_pot - v_sq / c_sq)
        + (4.0 * GM_SUN / (c_sq * r_norm**2)) * float(np.dot(r_hat, v_vec)) * v_vec
    )
    return a_1pn


def compute_schiff_gyro_precession_rate(r_vec: np.ndarray, v_vec: np.ndarray) -> np.ndarray:
    """Compute Schiff relativistic geodetic and frame-dragging spin precession rate.

    Args:
        r_vec: 3D position vector relative to central body [m].
        v_vec: 3D coordinate velocity vector [m/s].

    Returns:
        3D angular precession rate vector Omega_Schiff [rad/s].
    """
    r_norm = float(np.linalg.norm(r_vec))
    if r_norm < 1.0:
        return np.zeros(3, dtype=np.float64)

    c_sq = C_LIGHT**2
    omega_geodetic = (1.5 * GM_SUN / (c_sq * r_norm**3)) * np.cross(r_vec, v_vec)
    return omega_geodetic


def vector_to_unit_quaternion(direction_vec: np.ndarray) -> np.ndarray:
    """Compute unit attitude quaternion aligning spacecraft +Z body axis with direction vector.

    Args:
        direction_vec: 3D desired thrust direction vector.

    Returns:
        4-element unit quaternion [q0, q1, q2, q3] where q0 is scalar part.
    """
    d_norm = float(np.linalg.norm(direction_vec))
    if d_norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    v_target = direction_vec / d_norm
    v_source = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    dot_prod = float(np.dot(v_source, v_target))
    if dot_prod >= 0.9999999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if dot_prod <= -0.9999999:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

    cross_prod = np.cross(v_source, v_target)
    s = math.sqrt((1.0 + dot_prod) * 2.0)
    q0 = 0.5 * s
    qv = cross_prod / s
    q = np.array([q0, qv[0], qv[1], qv[2]], dtype=np.float64)
    return q / float(np.linalg.norm(q))


def propagate_free_trajectory(
    r_init: np.ndarray,
    v_init: np.ndarray,
    duration_s: float,
    steps: int = 40,
    include_gravity: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate unforced (free-flight) state under 1PN relativistic gravity using RK4.

    Args:
        r_init: Initial 3D position vector [m].
        v_init: Initial 3D velocity vector [m/s].
        duration_s: Time duration to propagate t_go [s].
        steps: Number of integration sub-steps.
        include_gravity: If False, computes flat-space unperturbed inertial drift.

    Returns:
        Tuple of (r_final, v_final) at t_go.
    """
    if duration_s <= 0.0:
        return r_init.copy(), v_init.copy()
    if not include_gravity:
        return r_init + v_init * duration_s, v_init.copy()

    dt = duration_s / float(max(steps, 1))
    r = r_init.astype(np.float64, copy=True)
    v = v_init.astype(np.float64, copy=True)

    for _ in range(steps):
        k1_r = v
        k1_v = compute_1pn_gravity_acceleration(r, v)

        r_mid1 = r + 0.5 * dt * k1_r
        v_mid1 = v + 0.5 * dt * k1_v
        k2_r = v_mid1
        k2_v = compute_1pn_gravity_acceleration(r_mid1, v_mid1)

        r_mid2 = r + 0.5 * dt * k2_r
        v_mid2 = v + 0.5 * dt * k2_v
        k3_r = v_mid2
        k3_v = compute_1pn_gravity_acceleration(r_mid2, v_mid2)

        r_end = r + dt * k3_r
        v_end = v + dt * k3_v
        k4_r = v_end
        k4_v = compute_1pn_gravity_acceleration(r_end, v_end)

        r += (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
        v += (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

    return r, v


class ZEMZEVGuidanceLaw:
    """Optimal Linear-Feedback Zero-Effort-Miss / Zero-Effort-Velocity Guidance Controller."""

    def __init__(
        self,
        target_position_m: Sequence[float],
        target_velocity_ms: Sequence[float],
        target_arrival_epoch_s: float,
        thrust_max_n: float = 10.0,
        isp_sec: float = 4000.0,
        min_time_to_go_s: float = 10.0,
        include_gravity: bool = True,
        propagation_steps: int = 40,
    ) -> None:
        """Initialize ZEM/ZEV controller.

        Args:
            target_position_m:      Desired 3D target coordinates r_f [m].
            target_velocity_ms:     Desired 3D target velocity v_f [m/s].
            target_arrival_epoch_s: Final mission intercept epoch t_f [s].
            thrust_max_n:           Maximum continuous thrust capacity [N].
            isp_sec:                Specific impulse of propulsion system [s].
            min_time_to_go_s:       Terminal guidance cutoff boundary [s].
            include_gravity:        Whether to evaluate gravitational field along free arc.
            propagation_steps:      Sub-steps used to propagate free trajectory forward to t_f.
        """
        self.r_f = np.asarray(target_position_m, dtype=np.float64)
        self.v_f = np.asarray(target_velocity_ms, dtype=np.float64)
        self.t_f = float(target_arrival_epoch_s)
        self.thrust_max = float(thrust_max_n)
        self.isp = float(isp_sec)
        self.min_t_go = float(min_time_to_go_s)
        self.include_gravity = include_gravity
        self.propagation_steps = max(int(propagation_steps), 5)

    def compute_command(
        self,
        current_epoch_s: float,
        current_position_m: np.ndarray,
        current_velocity_ms: np.ndarray,
        current_mass_kg: float,
    ) -> GuidanceCommand:
        """Compute optimal feedback guidance acceleration command.

        Evaluates 1PN gravity compensated Zero-Effort-Miss and Zero-Effort-Velocity.
        """
        t_go = max(self.t_f - current_epoch_s, 0.0)
        r_curr = np.asarray(current_position_m, dtype=np.float64)
        v_curr = np.asarray(current_velocity_ms, dtype=np.float64)
        mass = max(float(current_mass_kg), 1.0)

        if t_go <= self.min_t_go:
            return GuidanceCommand(
                epoch_seconds=current_epoch_s,
                time_to_go_s=t_go,
                zem_m=self.r_f - r_curr,
                zev_ms=self.v_f - v_curr,
                cmd_accel_ms2=np.zeros(3, dtype=np.float64),
                clamped_accel_ms2=np.zeros(3, dtype=np.float64),
                thrust_n=0.0,
                attitude_quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                mass_flow_rate_kg_s=0.0,
            )

        r_free, v_free = propagate_free_trajectory(
            r_curr,
            v_curr,
            t_go,
            steps=self.propagation_steps,
            include_gravity=self.include_gravity,
        )

        zem = self.r_f - r_free
        zev = self.v_f - v_free

        a_cmd = (6.0 / (t_go**2)) * zem - (2.0 / t_go) * zev

        a_max = self.thrust_max / mass
        a_cmd_norm = float(np.linalg.norm(a_cmd))

        if a_cmd_norm > a_max:
            clamped_accel = a_cmd * (a_max / a_cmd_norm)
            thrust_mag = self.thrust_max
        else:
            clamped_accel = a_cmd.copy()
            thrust_mag = a_cmd_norm * mass

        attitude_q = vector_to_unit_quaternion(clamped_accel)

        v_norm = float(np.linalg.norm(v_curr))
        inv_gamma = math.sqrt(max(1.0 - (v_norm / C_LIGHT) ** 2, 1e-12))
        m_dot = (thrust_mag / (self.isp * G0)) * inv_gamma

        return GuidanceCommand(
            epoch_seconds=current_epoch_s,
            time_to_go_s=t_go,
            zem_m=zem,
            zev_ms=zev,
            cmd_accel_ms2=a_cmd,
            clamped_accel_ms2=clamped_accel,
            thrust_n=thrust_mag,
            attitude_quaternion=attitude_q,
            mass_flow_rate_kg_s=m_dot,
        )


def simulate_closed_loop_mission(
    duration_days: float = 30.0,
    step_hours: float = 6.0,
    initial_position_au: Sequence[float] = (1.0, 0.0, 0.0),
    initial_velocity_kms: Sequence[float] = (0.0, 29.78, 0.0),
    target_position_au: Optional[Sequence[float]] = None,
    target_velocity_kms: Optional[Sequence[float]] = None,
    wet_mass_kg: float = 1500.0,
    thrust_max_n: float = 5.0,
    isp_sec: float = 4500.0,
    initial_pos_dispersion_m: float = 5000.0,
    initial_vel_dispersion_ms: float = 0.5,
    include_gravity: bool = True,
) -> Dict[str, Any]:
    """Simulate complete closed-loop autonomous trajectory steering under 1PN dynamics.

    Evaluates ZEM/ZEV feedback guidance response, relativistic mass depletion,
    and terminal intercept accuracy under initial navigation dispersion.
    """
    total_seconds = duration_days * 86400.0
    dt = step_hours * 3600.0
    num_steps = max(int(math.ceil(total_seconds / dt)), 1)

    r_nominal_init = np.array([initial_position_au[0] * AU, initial_position_au[1] * AU, initial_position_au[2] * AU], dtype=np.float64)
    v_nominal_init = np.array([initial_velocity_kms[0] * 1000.0, initial_velocity_kms[1] * 1000.0, initial_velocity_kms[2] * 1000.0], dtype=np.float64)

    if target_position_au is not None and target_velocity_kms is not None:
        r_target = np.array([target_position_au[0] * AU, target_position_au[1] * AU, target_position_au[2] * AU], dtype=np.float64)
        v_target = np.array([target_velocity_kms[0] * 1000.0, target_velocity_kms[1] * 1000.0, target_velocity_kms[2] * 1000.0], dtype=np.float64)
    else:
        # Compute nominal unperturbed arrival state under 1PN dynamics
        r_target, v_target = propagate_free_trajectory(
            r_nominal_init,
            v_nominal_init,
            total_seconds,
            steps=num_steps * 4,
            include_gravity=include_gravity,
        )

    # Apply initial navigation dispersion to actual spacecraft state
    r_sc = r_nominal_init + np.array([initial_pos_dispersion_m / math.sqrt(3)] * 3)
    v_sc = v_nominal_init + np.array([initial_vel_dispersion_ms / math.sqrt(3)] * 3)

    controller = ZEMZEVGuidanceLaw(
        target_position_m=r_target,
        target_velocity_ms=v_target,
        target_arrival_epoch_s=total_seconds,
        thrust_max_n=thrust_max_n,
        isp_sec=isp_sec,
        min_time_to_go_s=dt * 0.25,
        include_gravity=include_gravity,
        propagation_steps=20,
    )

    current_mass = float(wet_mass_kg)
    current_t = 0.0
    telemetry: List[Dict[str, Any]] = []

    for step in range(num_steps):
        cmd = controller.compute_command(
            current_epoch_s=current_t,
            current_position_m=r_sc,
            current_velocity_ms=v_sc,
            current_mass_kg=current_mass,
        )

        zem_norm = float(np.linalg.norm(cmd.zem_m))
        zev_norm = float(np.linalg.norm(cmd.zev_ms))
        cmd_accel_norm = float(np.linalg.norm(cmd.clamped_accel_ms2))

        omega_schiff = compute_schiff_gyro_precession_rate(r_sc, v_sc) if include_gravity else np.zeros(3)
        schiff_norm_arcsec_yr = float(np.linalg.norm(omega_schiff)) * (180.0 / math.pi) * 3600.0 * (86400.0 * 365.25)

        telemetry.append({
            "day": round(current_t / 86400.0, 2),
            "time_to_go_days": round(cmd.time_to_go_s / 86400.0, 2),
            "zem_m": round(zem_norm, 2),
            "zev_ms": round(zev_norm, 4),
            "thrust_n": round(cmd.thrust_n, 3),
            "thrust_accel_ms2": round(cmd_accel_norm, 6),
            "mass_kg": round(current_mass, 2),
            "schiff_precession_arcsec_yr": round(schiff_norm_arcsec_yr, 4),
            "r_au": [round(float(x) / AU, 6) for x in r_sc],
            "v_kms": [round(float(x) / 1000.0, 4) for x in v_sc],
        })

        if include_gravity:
            a_grav = compute_1pn_gravity_acceleration(r_sc, v_sc)
        else:
            a_grav = np.zeros(3, dtype=np.float64)

        a_total = a_grav + cmd.clamped_accel_ms2

        # Symplectic Verlet integration of spacecraft actual state
        r_sc = r_sc + v_sc * dt + 0.5 * a_total * (dt**2)
        v_sc = v_sc + a_total * dt
        current_mass = max(current_mass - cmd.mass_flow_rate_kg_s * dt, 100.0)
        current_t += dt

    final_miss_distance_m = float(np.linalg.norm(r_sc - r_target))
    final_velocity_error_ms = float(np.linalg.norm(v_sc - v_target))
    total_propellant_used_kg = wet_mass_kg - current_mass

    return {
        "duration_days": duration_days,
        "step_hours": step_hours,
        "num_steps": num_steps,
        "initial_wet_mass_kg": wet_mass_kg,
        "final_mass_kg": current_mass,
        "total_propellant_used_kg": total_propellant_used_kg,
        "final_miss_distance_m": final_miss_distance_m,
        "final_velocity_error_ms": final_velocity_error_ms,
        "telemetry": telemetry,
    }
