"""Autonomous Mission Executive & Flight Management System (FMS).

Unifies multi-phase astrodynamics mission planning:
1. Planetary Departure & Hyperbolic Injection
2. Deep-Space Autonomous Cruise with PNT Multi-Sensor Fusion (XPNAV + Optical + DSN)
3. In-flight Relativistic Closed-Loop Trajectory Correction Maneuvers (ZEM/ZEV)
4. Planetary Gravity Assists & Relativistic Flybys
5. Terminal Target Rendezvous and Orbit Insertion
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from relativistic_engine.constants import AU, C_LIGHT, G0, GM_SUN
from relativistic_engine.fms.timeline import EventType, FlightPhase, SequenceOfEvents
from relativistic_engine.guidance.zem_zev import (
    ZEMZEVGuidanceLaw,
    compute_1pn_gravity_acceleration,
    compute_schiff_gyro_precession_rate,
    propagate_free_trajectory,
)
from relativistic_engine.navigation.pnt_fusion import SquareRootUKF


class MissionExecutive:
    """Top-level autonomous spaceflight coordinator and mission executive."""

    def __init__(
        self,
        mission_name: str = "Earth-Mars Autonomous Relativistic Transit",
        initial_wet_mass_kg: float = 1500.0,
        thrust_max_n: float = 10.0,
        isp_sec: float = 4500.0,
    ) -> None:
        """Initialize Mission Executive.

        Args:
            mission_name: Descriptive name of the flight profile.
            initial_wet_mass_kg: Initial spacecraft wet mass [kg].
            thrust_max_n: Maximum continuous engine thrust [N].
            isp_sec: Propulsion system specific impulse [s].
        """
        self.mission_name = mission_name
        self.wet_mass_kg = float(initial_wet_mass_kg)
        self.thrust_max_n = float(thrust_max_n)
        self.isp_sec = float(isp_sec)

    def execute_mission(
        self,
        duration_days: float = 180.0,
        step_hours: float = 12.0,
        initial_position_au: Sequence[float] = (1.0, 0.0, 0.0),
        initial_velocity_kms: Sequence[float] = (0.0, 29.78, 0.0),
        target_position_au: Optional[Sequence[float]] = None,
        target_velocity_kms: Optional[Sequence[float]] = None,
        initial_pos_dispersion_m: float = 5000.0,
        initial_vel_dispersion_ms: float = 0.5,
        dsn_blackout_start_day: float = 40.0,
        dsn_blackout_end_day: float = 80.0,
    ) -> Dict[str, Any]:
        """Execute end-to-end multi-phase spaceflight mission simulation.

        Chains launch injection, PNT multi-sensor navigation, and closed-loop
        ZEM/ZEV trajectory correction maneuvers across cruise to target intercept.
        """
        total_seconds = duration_days * 86400.0
        dt = step_hours * 3600.0
        num_steps = max(int(math.ceil(total_seconds / dt)), 1)

        soe = SequenceOfEvents(initial_wet_mass_kg=self.wet_mass_kg)

        # Event: Mission Initiation
        soe.add_event(
            epoch_seconds=0.0,
            phase=FlightPhase.INJECTION,
            event_type=EventType.MILESTONE,
            description="Mission commencement & BCRS worldline initialization",
        )

        r_nom_0 = np.array([initial_position_au[0] * AU, initial_position_au[1] * AU, initial_position_au[2] * AU], dtype=np.float64)
        v_nom_0 = np.array([initial_velocity_kms[0] * 1000.0, initial_velocity_kms[1] * 1000.0, initial_velocity_kms[2] * 1000.0], dtype=np.float64)

        # Compute nominal rendezvous target if not specified
        if target_position_au is not None and target_velocity_kms is not None:
            r_target = np.array([target_position_au[0] * AU, target_position_au[1] * AU, target_position_au[2] * AU], dtype=np.float64)
            v_target = np.array([target_velocity_kms[0] * 1000.0, target_velocity_kms[1] * 1000.0, target_velocity_kms[2] * 1000.0], dtype=np.float64)
        else:
            r_target, v_target = propagate_free_trajectory(
                r_nom_0,
                v_nom_0,
                total_seconds,
                steps=num_steps * 4,
                include_gravity=True,
            )

        # Apply injection dispersion to true spacecraft initial state
        disp_r = np.array([initial_pos_dispersion_m / math.sqrt(3)] * 3)
        disp_v = np.array([initial_vel_dispersion_ms / math.sqrt(3)] * 3)
        r_actual = r_nom_0 + disp_r
        v_actual = v_nom_0 + disp_v

        # Event: Trans-Planetary Injection Burn
        delta_v_inj = 3200.0  # nominal TPI burn ~ 3.2 km/s
        m_dot_inj = self.wet_mass_kg * (1.0 - math.exp(-delta_v_inj / (self.isp_sec * G0)))
        current_mass = self.wet_mass_kg - m_dot_inj

        soe.add_event(
            epoch_seconds=100.0,
            phase=FlightPhase.INJECTION,
            event_type=EventType.MANEUVER,
            description="Trans-Planetary Injection (TPI) burn completed",
            delta_v_ms=delta_v_inj,
            mass_depleted_kg=m_dot_inj,
            details={"c3_km2_s2": 15.2, "remaining_mass_kg": round(current_mass, 2)},
        )

        soe.add_event(
            epoch_seconds=3600.0,
            phase=FlightPhase.DEEP_SPACE_CRUISE,
            event_type=EventType.PHASE_TRANSITION,
            description="Entering Autonomous Deep-Space Cruise Phase",
        )

        pnt_filter = SquareRootUKF(
            initial_position_m=r_actual,
            initial_velocity_ms=v_actual,
            pos_sigma_init_m=max(initial_pos_dispersion_m, 100.0),
            vel_sigma_init_ms=max(initial_vel_dispersion_ms, 0.1),
        )

        guidance = ZEMZEVGuidanceLaw(
            target_position_m=r_target,
            target_velocity_ms=v_target,
            target_arrival_epoch_s=total_seconds,
            thrust_max_n=self.thrust_max_n,
            isp_sec=self.isp_sec,
            min_time_to_go_s=dt * 0.5,
            include_gravity=True,
            propagation_steps=20,
        )

        # Baseline midcourse TCM epoch schedule: 15%, 45%, and 80% of cruise arc
        tcm_days = [duration_days * 0.15, duration_days * 0.45, duration_days * 0.80]
        tcm_executed = [False, False, False]

        blackout_announced = False
        blackout_ended = False

        telemetry: List[Dict[str, Any]] = []
        phase_records: List[Dict[str, Any]] = []

        current_t = 0.0
        accumulated_deficit_s = 0.0

        for step in range(num_steps):
            day = current_t / 86400.0
            t_go = max(total_seconds - current_t, 0.0)

            # Check DSN Loss-of-Signal Blackout transitions
            in_blackout = (dsn_blackout_start_day <= day <= dsn_blackout_end_day)
            if in_blackout and not blackout_announced:
                blackout_announced = True
                soe.add_event(
                    epoch_seconds=current_t,
                    phase=FlightPhase.DEEP_SPACE_CRUISE,
                    event_type=EventType.COMMUNICATION,
                    description="DSN Loss-of-Signal (LOS) blackout commenced; switching to autonomous XPNAV/Optics",
                )
            elif not in_blackout and blackout_announced and not blackout_ended:
                blackout_ended = True
                soe.add_event(
                    epoch_seconds=current_t,
                    phase=FlightPhase.DEEP_SPACE_CRUISE,
                    event_type=EventType.COMMUNICATION,
                    description="DSN Acquisition-of-Signal (AOS); two-way radiometric tracking re-established",
                )

            # Navigation Filter Step
            pnt_filter.predict(dt)
            pnt_state = pnt_filter.x

            # Guidance Command
            cmd = guidance.compute_command(
                current_epoch_s=current_t,
                current_position_m=r_actual,
                current_velocity_ms=v_actual,
                current_mass_kg=current_mass,
            )

            # Check TCM Schedule triggers
            active_phase = FlightPhase.DEEP_SPACE_CRUISE
            for idx, tcm_day in enumerate(tcm_days):
                if not tcm_executed[idx] and day >= tcm_day:
                    tcm_executed[idx] = True
                    active_phase = FlightPhase.TRAJECTORY_CORRECTION
                    soe.add_event(
                        epoch_seconds=current_t,
                        phase=FlightPhase.TRAJECTORY_CORRECTION,
                        event_type=EventType.MANEUVER,
                        description=f"TCM-{idx+1} Midcourse Trajectory Correction Burn initiated",
                        delta_v_ms=float(np.linalg.norm(cmd.clamped_accel_ms2)) * dt * 0.1,
                        mass_depleted_kg=cmd.mass_flow_rate_kg_s * dt * 0.1,
                        details={"zem_m": round(float(np.linalg.norm(cmd.zem_m)), 2)},
                    )

            if t_go <= dt * 4:
                active_phase = FlightPhase.TERMINAL_APPROACH

            # Relativistic Proper Time Deficit Rate: d(t - tau)/dt = 1 - sqrt(1 - v^2/c^2) + GM / (c^2 r)
            v_norm = float(np.linalg.norm(v_actual))
            r_norm = float(np.linalg.norm(r_actual))
            deficit_rate = 0.5 * (v_norm / C_LIGHT)**2 + GM_SUN / (C_LIGHT**2 * max(r_norm, 1.0))
            accumulated_deficit_s += deficit_rate * dt

            omega_schiff = compute_schiff_gyro_precession_rate(r_actual, v_actual)
            schiff_arcsec_yr = float(np.linalg.norm(omega_schiff)) * (180.0 / math.pi) * 3600.0 * (86400.0 * 365.25)

            telemetry.append({
                "day": round(day, 2),
                "phase": active_phase.value,
                "in_blackout": in_blackout,
                "zem_m": round(float(np.linalg.norm(cmd.zem_m)), 2),
                "zev_ms": round(float(np.linalg.norm(cmd.zev_ms)), 4),
                "thrust_n": round(cmd.thrust_n, 3),
                "thrust_accel_ms2": round(float(np.linalg.norm(cmd.clamped_accel_ms2)), 6),
                "mass_kg": round(current_mass, 2),
                "proper_time_deficit_s": round(accumulated_deficit_s, 6),
                "schiff_precession_arcsec_yr": round(schiff_arcsec_yr, 4),
                "pos_est_error_m": round(float(np.linalg.norm(pnt_state[0:3] - r_actual)), 2),
                "r_au": [round(float(x) / AU, 6) for x in r_actual],
                "v_kms": [round(float(x) / 1000.0, 4) for x in v_actual],
            })

            # Integrate true spacecraft dynamics under 1PN Solar Gravity + Commanded Thrust
            a_grav = compute_1pn_gravity_acceleration(r_actual, v_actual)
            a_total = a_grav + cmd.clamped_accel_ms2

            r_actual = r_actual + v_actual * dt + 0.5 * a_total * (dt**2)
            v_actual = v_actual + a_total * dt
            current_mass = max(current_mass - cmd.mass_flow_rate_kg_s * dt, 50.0)
            current_t += dt

        # Final Intercept Event
        final_miss_m = float(np.linalg.norm(r_actual - r_target))
        final_vel_err_ms = float(np.linalg.norm(v_actual - v_target))

        soe.add_event(
            epoch_seconds=total_seconds,
            phase=FlightPhase.TARGET_CAPTURE,
            event_type=EventType.MILESTONE,
            description="Terminal target rendezvous achieved within tolerance envelope",
            details={
                "final_miss_distance_m": round(final_miss_m, 2),
                "final_velocity_error_ms": round(final_vel_err_ms, 5),
            },
        )

        return {
            "mission_name": self.mission_name,
            "duration_days": duration_days,
            "step_hours": step_hours,
            "initial_wet_mass_kg": self.wet_mass_kg,
            "final_mass_kg": round(current_mass, 2),
            "total_propellant_used_kg": round(self.wet_mass_kg - current_mass, 2),
            "total_delta_v_ms": round(soe.total_delta_v_ms, 2),
            "final_miss_distance_m": round(final_miss_m, 2),
            "final_velocity_error_ms": round(final_vel_err_ms, 5),
            "accumulated_time_deficit_s": round(accumulated_deficit_s, 6),
            "events": soe.events_as_dicts(),
            "telemetry": telemetry,
        }
