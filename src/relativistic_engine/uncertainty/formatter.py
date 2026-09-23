"""GUM-compliant scientific uncertainty and significant-figure reporting.

Enforces Principle 4 (Never Hide Uncertainty):
- Suppresses spurious floating-point precision.
- Formats quantities according to the BIPM/ISO Guide to the Expression of
  Uncertainty in Measurement (JCGM 100:2008 / ISO/IEC Guide 98-3).
- Ensures nominal values are rounded to the same precision as their uncertainties.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


def _round_to_n_sig_figs(x: float, n: int = 2) -> tuple[float, int]:
    """Round a positive number to n significant figures and return (rounded_val, decimals).

    Decimals is the number of digits after the decimal point (may be negative).
    """
    if x <= 0.0 or not math.isfinite(x):
        return x, 0

    exponent = math.floor(math.log10(x))
    decimals = n - 1 - exponent
    factor = 10.0 ** decimals
    rounded = round(x * factor) / factor
    return rounded, decimals


def format_uncertainty(
    value: float,
    uncertainty: float,
    unit: str = "",
    *,
    sig_figs: int = 2,
    use_scientific: Optional[bool] = None,
) -> str:
    """Format a physical quantity and its standard uncertainty to defensible digits.

    Follows JCGM 100:2008 Section 7.2.2:
    - Uncertainty is rounded to `sig_figs` significant digits (default: 2).
    - Value is rounded to the same decimal place as the rounded uncertainty.

    Examples:
        format_uncertainty(4.498064, 0.0231, "days") -> "(4.498 ± 0.023) days"
        format_uncertainty(123456.7, 45.2, "m", use_scientific=False) -> "(123457 ± 45) m"
        format_uncertainty(1.905335e6, 2500.0, "m/s") -> "(1.9053 ± 0.0025) x 10^6 m/s"

    Args:
        value: Nominal physical value.
        uncertainty: 1-sigma standard uncertainty (must be >= 0).
        unit: Optional physical unit string.
        sig_figs: Number of significant figures for uncertainty (default: 2).
        use_scientific: Whether to use scientific notation. If None, auto-selects
            scientific notation for |value| >= 1e6 or 0 < |value| < 1e-3.

    Returns:
        Formatted string representing value ± uncertainty.
    """
    if uncertainty < 0.0:
        raise ValueError(f"Uncertainty must be non-negative, got {uncertainty}")

    if not math.isfinite(value) or not math.isfinite(uncertainty):
        return f"{value} ± {uncertainty} {unit}".strip()

    if uncertainty == 0.0:
        return f"{value:.6g} (exact) {unit}".strip()

    val_abs = abs(value) if value != 0.0 else uncertainty

    if use_scientific is True:
        sci = True
    elif use_scientific is False:
        sci = False
    else:
        # Default auto-threshold: 1e6 or 1e-3
        sci = val_abs >= 1.0e6 or (val_abs < 1.0e-3 and val_abs > 0.0)

    if sci:
        scale_exp = math.floor(math.log10(val_abs))
        scale = 10.0 ** scale_exp
        scaled_val = value / scale
        scaled_unc = uncertainty / scale

        rounded_unc, dec = _round_to_n_sig_figs(scaled_unc, sig_figs)
        dec = max(0, dec)
        rounded_val = round(scaled_val, dec)

        val_str = f"{rounded_val:.{dec}f}"
        unc_str = f"{rounded_unc:.{dec}f}"
        unit_str = f" {unit}" if unit else ""
        return f"({val_str} ± {unc_str}) x 10^{scale_exp}{unit_str}"

    else:
        rounded_unc, dec = _round_to_n_sig_figs(uncertainty, sig_figs)
        if dec >= 0:
            rounded_val = round(value, dec)
            val_str = f"{rounded_val:.{dec}f}"
            unc_str = f"{rounded_unc:.{dec}f}"
        else:
            rounded_val = round(value, dec)
            val_str = f"{int(round(rounded_val)):d}"
            unc_str = f"{int(round(rounded_unc)):d}"

        unit_str = f" {unit}" if unit else ""
        return f"({val_str} ± {unc_str}){unit_str}"


@dataclass(frozen=True)
class UncertainQuantity:
    """A physical quantity with associated standard uncertainty.

    Attributes
    ----------
    value : float
        Nominal central value in SI units.
    uncertainty : float
        1-sigma standard uncertainty in SI units.
    unit : str
        Unit descriptor string (e.g. 'm', 's', 'm/s', 'days').
    """

    value: float
    uncertainty: float
    unit: str = ""

    def __str__(self) -> str:
        return format_uncertainty(self.value, self.uncertainty, self.unit)

    def __repr__(self) -> str:
        return f"UncertainQuantity(value={self.value}, uncertainty={self.uncertainty}, unit='{self.unit}')"

    @property
    def relative_uncertainty(self) -> float:
        """Fractional / relative uncertainty sigma / |value|."""
        if self.value == 0.0:
            return float("inf") if self.uncertainty > 0.0 else 0.0
        return self.uncertainty / abs(self.value)
