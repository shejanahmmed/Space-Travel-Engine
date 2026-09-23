"""Ephemeris subpackage for JPL planetary kernel loading and BCRS state vector evaluation."""

from relativistic_engine.ephemeris.jpl_loader import (
    load_jpl_ephemeris,
    get_default_ephemeris_path,
    download_de440s_kernel,
)
from relativistic_engine.ephemeris.barycentric import (
    CelestialBodyState,
    get_body_barycentric_state,
    get_relative_state,
    SUPPORTED_BODIES,
)

from relativistic_engine.ephemeris.interstellar import (
    StarCatalogEntry,
    StarStateBCRS,
    INTERSTELLAR_CATALOG,
    get_star_barycentric_state,
)

__all__ = [
    "load_jpl_ephemeris",
    "get_default_ephemeris_path",
    "download_de440s_kernel",
    "CelestialBodyState",
    "get_body_barycentric_state",
    "get_relative_state",
    "SUPPORTED_BODIES",
    "StarCatalogEntry",
    "StarStateBCRS",
    "INTERSTELLAR_CATALOG",
    "get_star_barycentric_state",
]

