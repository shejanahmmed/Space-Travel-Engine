"""Sequence of Events (SOE) and Mission Timeline Engine.

Manages time-tagged flight milestones, propulsion maneuvers, navigation sensor status,
and propellant mass depletion accounting across multi-phase spaceflight operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class FlightPhase(str, Enum):
    """Discrete operational phases of an interplanetary / deep-space flight profile."""

    INJECTION = "INJECTION"
    DEEP_SPACE_CRUISE = "DEEP_SPACE_CRUISE"
    TRAJECTORY_CORRECTION = "TRAJECTORY_CORRECTION"
    PLANETARY_FLYBY = "PLANETARY_FLYBY"
    TERMINAL_APPROACH = "TERMINAL_APPROACH"
    TARGET_CAPTURE = "TARGET_CAPTURE"


class EventType(str, Enum):
    """Categorized mission event classifications."""

    MANEUVER = "MANEUVER"
    NAVIGATION = "NAVIGATION"
    COMMUNICATION = "COMMUNICATION"
    PHASE_TRANSITION = "PHASE_TRANSITION"
    MILESTONE = "MILESTONE"


@dataclass(frozen=True)
class MissionEvent:
    """A discrete, time-tagged operational mission event.

    Attributes:
        epoch_seconds: Flight coordinate time from mission commencement [s].
        epoch_days:    Flight coordinate time in Julian days [d].
        phase:         Active flight phase.
        event_type:    Classification of event.
        description:   Human- and machine-readable description.
        delta_v_ms:    Delta-v impulse applied or required [m/s].
        mass_depleted_kg: Propellant mass consumed [kg].
        details:       Supplementary state telemetry dictionary.
    """

    epoch_seconds: float
    epoch_days: float
    phase: FlightPhase
    event_type: EventType
    description: str
    delta_v_ms: float = 0.0
    mass_depleted_kg: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to standard dictionary."""
        return {
            "epoch_seconds": round(self.epoch_seconds, 2),
            "epoch_days": round(self.epoch_days, 4),
            "phase": self.phase.value,
            "event_type": self.event_type.value,
            "description": self.description,
            "delta_v_ms": round(self.delta_v_ms, 3),
            "mass_depleted_kg": round(self.mass_depleted_kg, 3),
            "details": self.details,
        }


class SequenceOfEvents:
    """Chronologically ordered container for mission flight events and fuel ledgers."""

    def __init__(self, initial_wet_mass_kg: float) -> None:
        """Initialize Sequence of Events container.

        Args:
            initial_wet_mass_kg: Spacecraft wet mass at mission start [kg].
        """
        self.initial_mass_kg = float(initial_wet_mass_kg)
        self._events: List[MissionEvent] = []

    def add_event(
        self,
        epoch_seconds: float,
        phase: FlightPhase,
        event_type: EventType,
        description: str,
        delta_v_ms: float = 0.0,
        mass_depleted_kg: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
    ) -> MissionEvent:
        """Append an event to the chronological timeline.

        Args:
            epoch_seconds: Flight time in seconds.
            phase: Current operational phase.
            event_type: Categorical event type.
            description: Narrative summary.
            delta_v_ms: Delta-v expenditure in m/s.
            mass_depleted_kg: Fuel mass expended in kg.
            details: Contextual payload metadata.

        Returns:
            Constructed MissionEvent.
        """
        ev = MissionEvent(
            epoch_seconds=float(epoch_seconds),
            epoch_days=float(epoch_seconds) / 86400.0,
            phase=phase,
            event_type=event_type,
            description=description,
            delta_v_ms=float(delta_v_ms),
            mass_depleted_kg=float(mass_depleted_kg),
            details=details or {},
        )
        self._events.append(ev)
        self._events.sort(key=lambda x: x.epoch_seconds)
        return ev

    @property
    def total_delta_v_ms(self) -> float:
        """Total impulsive and continuous Delta-v consumed across all events [m/s]."""
        return sum(e.delta_v_ms for e in self._events)

    @property
    def total_propellant_used_kg(self) -> float:
        """Total propellant mass expended across mission timeline [kg]."""
        return sum(e.mass_depleted_kg for e in self._events)

    @property
    def final_mass_kg(self) -> float:
        """Calculated final spacecraft mass after all maneuvers [kg]."""
        return max(self.initial_mass_kg - self.total_propellant_used_kg, 1.0)

    def events_as_dicts(self) -> List[Dict[str, Any]]:
        """Return all events as serialized dictionaries."""
        return [e.to_dict() for e in self._events]
