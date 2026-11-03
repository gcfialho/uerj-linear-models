"""Multiple linear regression from first principles."""

from linear_models.ols import (
    OLSResult,
    breusch_pagan,
    design_matrix,
    fit,
    normal_equations,
    pearson,
    residual_diagnostics,
)

__all__ = [
    "OLSResult",
    "breusch_pagan",
    "design_matrix",
    "fit",
    "normal_equations",
    "pearson",
    "residual_diagnostics",
]
__version__ = "1.0.0"
