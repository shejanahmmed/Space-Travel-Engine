"""JSON-LD reproducibility manifest generator.

Satisfies Project Principle 17: every major computation must record its
complete provenance -- inputs, physical constants with authoritative sources,
ephemeris metadata, solver configuration, and a SHA-256 digest of the output
-- so that an independent researcher can reproduce the result.

The produced document conforms to JSON-LD 1.1 with a minimal schema.org
context augmented by IVOA and QUDT namespaces for unit annotation.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from relativistic_engine.constants import (
    C_LIGHT,
    G_NEWTON,
    AU,
    GM_SUN,
    GM_EARTH,
)

# Computational engine build metadata and capability manifest
_ENGINE_VERSION = "1.0.0"
_TEST_SUITE_COUNT = 315

# Authoritative constant provenance table
_CONSTANTS_PROVENANCE: dict[str, dict[str, Any]] = {
    "C_LIGHT_mps": {
        "value": C_LIGHT,
        "unit": "m s^-1",
        "source": "BIPM 1983 -- exact SI definition of the metre",
        "uncertainty": "exact",
    },
    "G_NEWTON_m3kg-1s-2": {
        "value": G_NEWTON,
        "unit": "m^3 kg^-1 s^-2",
        "source": "CODATA 2018 (NIST SP 961)",
        "relative_uncertainty": 2.2e-5,
    },
    "AU_m": {
        "value": AU,
        "unit": "m",
        "source": "IAU 2012 Resolution B2 -- exact definition",
        "uncertainty": "exact",
    },
    "GM_SUN_m3s-2": {
        "value": GM_SUN,
        "unit": "m^3 s^-2",
        "source": "IAU 2015 Resolution B3 / JPL DE440",
        "relative_uncertainty": 1e-10,
    },
    "GM_EARTH_m3s-2": {
        "value": GM_EARTH,
        "unit": "m^3 s^-2",
        "source": "IERS Conventions 2010, Table 1.1",
        "relative_uncertainty": 1e-9,
    },
}

_EPHEMERIS_METADATA: dict[str, Any] = {
    "kernel": "de440s.bsp",
    "de_number": 440,
    "time_scale": "TDB",
    "coordinate_frame": "BCRS / ICRF J2000",
    "source": "NASA/JPL Solar System Dynamics -- https://ssd.jpl.nasa.gov/planets/eph_export.html",
}

_SOLVER_METADATA: dict[str, Any] = {
    "integrator": "DOP853",
    "order": 8,
    "rtol": 1e-12,
    "atol": 1e-12,
    "velocity_clamp_mps": "c - 1e-12",
    "reference": "Dormand & Prince (1980), J. Comput. Appl. Math. 6(1):19-26",
}


def _sha256_of(obj: Any) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generate_manifest(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    computation_id: str | None = None,
    endpoint: str | None = None,
) -> dict[str, Any]:
    """Return a JSON-LD provenance manifest for a single engine computation."""
    cid = computation_id or str(uuid.uuid4())
    created = datetime.now(tz=timezone.utc).isoformat()
    output_hash = _sha256_of(outputs)

    return {
        "@context": {
            "@vocab": "https://schema.org/",
            "astro": "https://www.ivoa.net/rdf/",
            "qudt": "http://qudt.org/vocab/unit/",
        },
        "@type": "Dataset",
        "identifier": cid,
        "dateCreated": created,
        "engine_version": _ENGINE_VERSION,
        "test_suite_count": _TEST_SUITE_COUNT,
        "api_endpoint": endpoint or "unspecified",
        "ephemeris": _EPHEMERIS_METADATA,
        "constants": _CONSTANTS_PROVENANCE,
        "solver": _SOLVER_METADATA,
        "inputs": inputs,
        "outputs": {
            "result_sha256": output_hash,
            **outputs,
        },
    }
