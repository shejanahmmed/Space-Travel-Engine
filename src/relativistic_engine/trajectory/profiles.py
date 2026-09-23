"""Thrust steering profiles and guidance laws for relativistic spaceflight.

Defines acceleration profiles for 3D continuous-thrust trajectories:
- Two-stage brachistochrone (boost-brake) guidance.
- Target-relative line-of-sight tracking guidance.
- Parameterized thrust direction steering.

Authoritative References:
- Lawden, D. F. (1963), "Optimal Trajectories for Space Navigation", Butterworths.
- Battin, R. H. (1999), "An Introduction to the Mathematics and Methods of Astrodynamics", AIAA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import math
import numpy as np


@dataclass(frozen=True)
class TwoStageThrustProfile:
    """Two-stage continuous proper acceleration profile (boost and brake).

    Stage 1: t in [0, t1), proper acceleration a1.
    Stage 2: t in [t1, t1 + t2], proper acceleration a2.

    Attributes
    ----------
    t1 : float
        Duration of Stage 1 (boost) in seconds.
    t2 : float
        Duration of Stage 2 (brake) in seconds.
    a1 : np.ndarray
        Constant proper acceleration vector [ax, ay, az] in m/s^2 for Stage 1.
    a2 : np.ndarray
        Constant proper acceleration vector [ax, ay, az] in m/s^2 for Stage 2.
    """

    t1: float
    t2: float
    a1: np.ndarray
    a2: np.ndarray

    @property
    def total_duration(self) -> float:
        """Total flight time in seconds."""
        return self.t1 + self.t2

    def __call__(self, t: float, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Evaluate proper acceleration vector at coordinate time t."""
        if t <= self.t1:
            return self.a1
        elif t <= self.total_duration:
            return self.a2
        else:
            return np.zeros(3, dtype=np.float64)


def create_brachistochrone_steering(
    accel_magnitude: float,
    t1: float,
    t2: float,
    dir1: np.ndarray | list[float],
    dir2: np.ndarray | list[float],
) -> TwoStageThrustProfile:
    """Create a two-stage brachistochrone profile with normalized direction vectors.

    Parameters
    ----------
    accel_magnitude : float
        Scalar proper acceleration |a| in m/s^2.
    t1 : float
        Duration of Stage 1 in seconds.
    t2 : float
        Duration of Stage 2 in seconds.
    dir1 : list[float]
        Stage 1 thrust direction vector.
    dir2 : list[float]
        Stage 2 thrust direction vector.

    Returns
    -------
    TwoStageThrustProfile
        Configured callable profile.
    """
    u1 = np.asarray(dir1, dtype=np.float64)
    u2 = np.asarray(dir2, dtype=np.float64)

    norm1 = np.linalg.norm(u1)
    norm2 = np.linalg.norm(u2)

    if norm1 == 0.0 or norm2 == 0.0:
        raise ValueError("Direction vectors must have non-zero norm")

    a1 = accel_magnitude * (u1 / norm1)
    a2 = accel_magnitude * (u2 / norm2)

    return TwoStageThrustProfile(t1=float(t1), t2=float(t2), a1=a1, a2=a2)
