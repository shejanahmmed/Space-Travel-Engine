"""Numerical subpackage for ODE integration, trajectory solving, and convergence verification."""

from relativistic_engine.numerical.integrator import (
    RelativisticRocketODE,
    integrate_relativistic_trajectory,
    integrate_brachistochrone,
    IntegrationResult,
)
from relativistic_engine.numerical.trajectory import (
    TrajectoryResult3D,
    propagate_trajectory_3d,
)
from relativistic_engine.numerical.batch_propagator import (
    BatchIntegratorBackend,
    BatchTrajectoryResult,
    propagate_batch_worldlines,
)

__all__ = [
    "RelativisticRocketODE",
    "integrate_relativistic_trajectory",
    "integrate_brachistochrone",
    "IntegrationResult",
    "TrajectoryResult3D",
    "propagate_trajectory_3d",
    "BatchIntegratorBackend",
    "BatchTrajectoryResult",
    "propagate_batch_worldlines",
]
