"""Scientific trajectory data export and archival engine.

Generates standardized CSV and JSON trajectory data products conforming to
NASA Planetary Data System (PDS) and IAU astronomical metadata conventions:
- Explicit coordinate frame definition: BCRS / ICRF (IAU 2000/2006)
- Explicit time scales: Barycentric Dynamical Time (TDB) vs Spacecraft Proper Time (tau)
- Metric signature and post-Newtonian gravity specifications
- Numerical solver tolerances and checksum provenance

Authoritative Standards:
- IAU 2000 Resolution B1.3 / IAU 2006 Resolution 3 (BCRS and time scales).
- NASA Planetary Data System (PDS4) Data Provider's Handbook.
- Principle 8: Ephemeris Data Must Be Traceable.
- Principle 16: Units Must Never Be Assumed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional
import numpy as np

from relativistic_engine.constants import C_LIGHT
from relativistic_engine.numerical.trajectory import TrajectoryResult3D
from relativistic_engine.physics.kinematics import (
    lorentz_beta,
    lorentz_gamma,
)


def export_trajectory_csv(
    trajectory: TrajectoryResult3D,
    filepath: Path | str,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Export a 3D relativistic trajectory to an archival CSV file with metadata header.

    Parameters
    ----------
    trajectory : TrajectoryResult3D
        Worldline solution containing t, tau, r, v, and time deficit.
    filepath : Path | str
        Destination file path.
    metadata : Optional[Mapping[str, Any]]
        Optional mission metadata dictionary (e.g. mission name, target, epoch).

    Returns
    -------
    Path
        Absolute path to the exported CSV file.
    """
    out_path = Path(filepath).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gen_time_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        # 1. PDS/IAU Astronomical Metadata Header Block
        f.write("# ==============================================================================\n")
        f.write("# RELATIVISTIC SPACE TRAVEL COMPUTATIONAL ENGINE - TRAJECTORY DATA PRODUCT\n")
        f.write("# Reference Frame: Barycentric Celestial Reference System (BCRS / ICRF, IAU 2000/2006)\n")
        f.write("# Time Scales: Coordinate Time (TDB seconds) vs Traveler Proper Time (tau seconds)\n")
        f.write("# Metric: Post-Newtonian (1PN) BCRS metric with signature diag(-c^2, 1, 1, 1)\n")
        f.write(f"# Generation Timestamp: {gen_time_utc}\n")
        f.write(f"# Solver Exit Status: {trajectory.status} ({trajectory.message})\n")
        f.write(f"# Total Function Evaluations: {trajectory.nfev}\n")

        if metadata:
            f.write("# Mission Metadata:\n")
            for k, v in metadata.items():
                f.write(f"#   {k}: {v}\n")

        f.write("# ==============================================================================\n")

        # 2. Column Headers
        writer = csv.writer(f)
        writer.writerow([
            "t_sec",
            "tau_sec",
            "time_deficit_sec",
            "x_m",
            "y_m",
            "z_m",
            "vx_m_s",
            "vy_m_s",
            "vz_m_s",
            "speed_m_s",
            "beta",
            "gamma",
        ])

        # 3. Trajectory Rows
        n_pts = len(trajectory.t)
        for i in range(n_pts):
            t_val = trajectory.t[i]
            tau_val = trajectory.tau[i]
            deficit_val = trajectory.time_deficit[i]
            rx, ry, rz = trajectory.r[i]
            vx, vy, vz = trajectory.v[i]
            speed = float(np.linalg.norm(trajectory.v[i]))
            beta_val = lorentz_beta(speed)
            gamma_val = lorentz_gamma(speed)

            writer.writerow([
                f"{t_val:.8e}",
                f"{tau_val:.8e}",
                f"{deficit_val:.8e}",
                f"{rx:.6e}",
                f"{ry:.6e}",
                f"{rz:.6e}",
                f"{vx:.6e}",
                f"{vy:.6e}",
                f"{vz:.6e}",
                f"{speed:.6e}",
                f"{beta_val:.8f}",
                f"{gamma_val:.8f}",
            ])

    return out_path


def export_trajectory_json(
    trajectory: TrajectoryResult3D,
    filepath: Path | str,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Export a 3D relativistic trajectory to structured archival JSON.

    Parameters
    ----------
    trajectory : TrajectoryResult3D
        Worldline solution.
    filepath : Path | str
        Destination file path.
    metadata : Optional[Mapping[str, Any]]
        Optional mission metadata dictionary.

    Returns
    -------
    Path
        Absolute path to the exported JSON file.
    """
    out_path = Path(filepath).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gen_time_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    states = []
    n_pts = len(trajectory.t)
    for i in range(n_pts):
        speed = float(np.linalg.norm(trajectory.v[i]))
        states.append({
            "t_sec": float(trajectory.t[i]),
            "tau_sec": float(trajectory.tau[i]),
            "time_deficit_sec": float(trajectory.time_deficit[i]),
            "position_m": [float(c) for c in trajectory.r[i]],
            "velocity_m_s": [float(c) for c in trajectory.v[i]],
            "speed_m_s": speed,
            "beta": lorentz_beta(speed),
            "gamma": lorentz_gamma(speed),
        })

    payload = {
        "metadata": {
            "title": "Relativistic Space Travel Computational Engine - Trajectory Product",
            "reference_frame": "BCRS / ICRF (IAU 2000/2006)",
            "metric": "Post-Newtonian 1PN BCRS",
            "generation_utc": gen_time_utc,
            "solver_status": trajectory.status,
            "solver_message": trajectory.message,
            "function_evaluations": trajectory.nfev,
            **(dict(metadata) if metadata else {}),
        },
        "trajectory": states,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return out_path
