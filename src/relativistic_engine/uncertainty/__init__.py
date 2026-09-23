"""Uncertainty subpackage for error propagation, variational STMs, and GUM reporting."""

from relativistic_engine.uncertainty.formatter import (
    format_uncertainty,
    UncertainQuantity,
)
from relativistic_engine.uncertainty.variational import (
    evaluate_gravity_gradient,
    evaluate_dynamics_jacobian,
    propagate_with_stm,
    STMPropagationResult,
)
from relativistic_engine.uncertainty.monte_carlo import (
    run_monte_carlo_ensemble,
    MonteCarloResult,
)
from relativistic_engine.uncertainty.batch_monte_carlo import (
    evaluate_batch_monte_carlo,
    BatchMonteCarloResult,
)

__all__ = [
    "format_uncertainty",
    "UncertainQuantity",
    "evaluate_gravity_gradient",
    "evaluate_dynamics_jacobian",
    "propagate_with_stm",
    "STMPropagationResult",
    "run_monte_carlo_ensemble",
    "MonteCarloResult",
    "evaluate_batch_monte_carlo",
    "BatchMonteCarloResult",
]
