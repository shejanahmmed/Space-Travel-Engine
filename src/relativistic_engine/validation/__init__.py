"""Validation and scientific benchmarking subpackage."""

from relativistic_engine.validation.horizons_validator import (
    HorizonsReferenceVector,
    HorizonsValidationReport,
    HORIZONS_DE440_BENCHMARKS,
    validate_ephemeris_against_horizons,
)
from relativistic_engine.validation.report_generator import (
    ProofBenchmarkRecord,
    ValidationCertificate,
    run_full_benchmark_suite,
    generate_validation_certificate,
)

__all__ = [
    "HorizonsReferenceVector",
    "HorizonsValidationReport",
    "HORIZONS_DE440_BENCHMARKS",
    "validate_ephemeris_against_horizons",
    "ProofBenchmarkRecord",
    "ValidationCertificate",
    "run_full_benchmark_suite",
    "generate_validation_certificate",
]
