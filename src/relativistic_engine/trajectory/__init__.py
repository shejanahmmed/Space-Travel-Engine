"""Trajectory subpackage for rendezvous solving, guidance profiles, and mission analysis."""

from relativistic_engine.trajectory.profiles import (
    TwoStageThrustProfile,
    create_brachistochrone_steering,
)
from relativistic_engine.trajectory.rendezvous import (
    RendezvousMode,
    RendezvousSolution,
    solve_interplanetary_rendezvous,
)
from relativistic_engine.trajectory.interstellar import (
    InterstellarMissionResult,
    solve_interstellar_brachistochrone,
)
from relativistic_engine.trajectory.tour import (
    PlanetaryTour,
    TourLeg,
    solve_planetary_tour,
)
from relativistic_engine.trajectory.mission_reconstruction import (
    HistoricalEncounterResult,
    reconstruct_voyager2_jupiter_flyby,
    reconstruct_cassini_jupiter_flyby,
)

__all__ = [
    "TwoStageThrustProfile",
    "create_brachistochrone_steering",
    "RendezvousMode",
    "RendezvousSolution",
    "solve_interplanetary_rendezvous",
    "InterstellarMissionResult",
    "solve_interstellar_brachistochrone",
    "PlanetaryTour",
    "TourLeg",
    "solve_planetary_tour",
    "HistoricalEncounterResult",
    "reconstruct_voyager2_jupiter_flyby",
    "reconstruct_cassini_jupiter_flyby",
]

