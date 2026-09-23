"""Monte Carlo non-linear uncertainty validation ensemble.

Stochastically samples initial state errors, navigation covariances, and engine
thrust tolerances to validate linearized State Transition Matrix (STM) propagation.

Authoritative Standards:
- ISO/IEC Guide 98-3:2008 / JCGM 101:2008 (Propagation of distributions using Monte Carlo).
- Principle 5: Deterministic physics (reproducible seed control).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
from jplephem.spk import SPK

from relativistic_engine.numerical.trajectory import (
    ThrustFunction,
    propagate_trajectory_3d,
)
from relativistic_engine.trajectory.profiles import TwoStageThrustProfile


@dataclass(frozen=True)
class MonteCarloResult:
    """Statistical results from a Monte Carlo trajectory ensemble.

    Attributes
    ----------
    num_samples : int
        Total number of Monte Carlo runs.
    nominal_final_position : np.ndarray
        Nominal unperturbed arrival position [x, y, z] in meters.
    nominal_final_velocity : np.ndarray
        Nominal unperturbed arrival velocity [vx, vy, vz] in m/s.
    mean_position : np.ndarray
        Sample mean arrival position [x, y, z] in meters.
    mean_velocity : np.ndarray
        Sample mean arrival velocity [vx, vy, vz] in m/s.
    covariance_position : np.ndarray
        3x3 empirical sample covariance of arrival position in m^2.
    covariance_velocity : np.ndarray
        3x3 empirical sample covariance of arrival velocity in (m/s)^2.
    std_position : np.ndarray
        1-sigma position standard deviation vector [sx, sy, sz] in meters.
    std_velocity : np.ndarray
        1-sigma velocity standard deviation vector [sx, sy, sz] in m/s.
    std_time_deficit : float
        1-sigma standard deviation of coordinate time deficit Delta t - Delta tau in seconds.
    """

    num_samples: int
    nominal_final_position: np.ndarray
    nominal_final_velocity: np.ndarray
    mean_position: np.ndarray
    mean_velocity: np.ndarray
    covariance_position: np.ndarray
    covariance_velocity: np.ndarray
    std_position: np.ndarray
    std_velocity: np.ndarray
    std_time_deficit: float


def run_monte_carlo_ensemble(
    r0: np.ndarray | Sequence[float],
    v0: np.ndarray | Sequence[float],
    t_span: tuple[float, float],
    *,
    thrust_profile: Optional[TwoStageThrustProfile] = None,
    sigma_r0: float = 0.0,
    sigma_v0: float = 0.0,
    thrust_relative_std: float = 0.0,
    num_samples: int = 100,
    seed: int = 42,
    epoch_jd_tdb: Optional[float] = None,
    spk: Optional[SPK] = None,
    gravitational_bodies: Optional[Sequence[str]] = ("sun",),
    include_1pn: bool = True,
    rtol: float = 1e-7,
    atol: float = 1e-8,
) -> MonteCarloResult:
    """Run an ensemble of perturbed relativistic trajectories.

    Parameters
    ----------
    r0 : Sequence[float]
        Nominal initial position in meters.
    v0 : Sequence[float]
        Nominal initial velocity in m/s.
    t_span : tuple[float, float]
        Flight duration interval (0, T) in seconds.
    thrust_profile : Optional[TwoStageThrustProfile]
        Nominal thrust profile.
    sigma_r0 : float
        Isotropic 1-sigma standard deviation for initial position in meters.
    sigma_v0 : float
        Isotropic 1-sigma standard deviation for initial velocity in m/s.
    thrust_relative_std : float
        Fractional standard deviation sigma_alpha / alpha of engine thrust magnitude.
    num_samples : int
        Number of stochastic trials (default: 100).
    seed : int
        PRNG seed for deterministic reproducibility.
    epoch_jd_tdb : Optional[float]
        Departure epoch in Julian Date (TDB).
    spk : Optional[SPK]
        NASA/JPL ephemeris kernel.
    gravitational_bodies : Optional[Sequence[str]]
        Bodies included in gravitational field.
    include_1pn : bool
        Whether to evaluate 1PN general relativistic solar acceleration.
    rtol : float
        ODE relative tolerance.
    atol : float
        ODE absolute tolerance.

    Returns
    -------
    MonteCarloResult
        Statistical summary of arrival states and empirical covariances.
    """
    rng = np.random.default_rng(seed)
    r_nom = np.asarray(r0, dtype=np.float64)
    v_nom = np.asarray(v0, dtype=np.float64)

    # 1. Run nominal trajectory
    nom_traj = propagate_trajectory_3d(
        r0=r_nom,
        v0=v_nom,
        t_span=t_span,
        thrust_func=thrust_profile,
        epoch_jd_tdb=epoch_jd_tdb,
        spk=spk,
        gravitational_bodies=gravitational_bodies,
        include_1pn=include_1pn,
        rtol=rtol,
        atol=atol,
    )
    r_final_nom = nom_traj.r[-1]
    v_final_nom = nom_traj.v[-1]

    # 2. Run ensemble
    positions = np.zeros((num_samples, 3), dtype=np.float64)
    velocities = np.zeros((num_samples, 3), dtype=np.float64)
    deficits = np.zeros(num_samples, dtype=np.float64)

    for i in range(num_samples):
        # Sample initial state perturbations
        dr0 = rng.normal(0.0, sigma_r0, size=3) if sigma_r0 > 0.0 else np.zeros(3)
        dv0 = rng.normal(0.0, sigma_v0, size=3) if sigma_v0 > 0.0 else np.zeros(3)

        r_sample = r_nom + dr0
        v_sample = v_nom + dv0

        # Sample thrust perturbation
        if thrust_profile is not None and thrust_relative_std > 0.0:
            scale_alpha = 1.0 + rng.normal(0.0, thrust_relative_std)
            sample_thrust = TwoStageThrustProfile(
                t1=thrust_profile.t1,
                t2=thrust_profile.t2,
                a1=thrust_profile.a1 * scale_alpha,
                a2=thrust_profile.a2 * scale_alpha,
            )
        else:
            sample_thrust = thrust_profile

        traj_sample = propagate_trajectory_3d(
            r0=r_sample,
            v0=v_sample,
            t_span=t_span,
            thrust_func=sample_thrust,
            epoch_jd_tdb=epoch_jd_tdb,
            spk=spk,
            gravitational_bodies=gravitational_bodies,
            include_1pn=include_1pn,
            rtol=rtol,
            atol=atol,
        )

        positions[i, :] = traj_sample.r[-1]
        velocities[i, :] = traj_sample.v[-1]
        deficits[i] = traj_sample.time_deficit[-1]

    # 3. Statistical reduction
    mean_r = np.mean(positions, axis=0)
    mean_v = np.mean(velocities, axis=0)

    cov_r = np.cov(positions, rowvar=False)
    cov_v = np.cov(velocities, rowvar=False)

    std_r = np.sqrt(np.diag(cov_r))
    std_v = np.sqrt(np.diag(cov_v))
    std_deficit = float(np.std(deficits, ddof=1))

    return MonteCarloResult(
        num_samples=num_samples,
        nominal_final_position=r_final_nom,
        nominal_final_velocity=v_final_nom,
        mean_position=mean_r,
        mean_velocity=mean_v,
        covariance_position=cov_r,
        covariance_velocity=cov_v,
        std_position=std_r,
        std_velocity=std_v,
        std_time_deficit=std_deficit,
    )
