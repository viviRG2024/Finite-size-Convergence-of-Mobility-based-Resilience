"""Analysis modules: nested windows, multi-scale CV/H, scaling fits, tests."""

from .nested_windows import build_nested_windows, clip_segments_to_bbox
from .multi_scale import (
    analyse_multi_scale,
    analyse_single_scale,
    compute_cv_per_hour,
    compute_structural_entropy_per_hour,
)
from .scaling_fit import (
    fit_local_power_law,
    compute_alpha_beta_per_hour,
)
from .statistical_tests import (
    bootstrap_correlation_ci,
    bootstrap_scaling_exponents,
    permutation_test_optimal_scale,
    ks_test_correlation_curves,
)

__all__ = [
    "build_nested_windows",
    "clip_segments_to_bbox",
    "analyse_multi_scale",
    "analyse_single_scale",
    "compute_cv_per_hour",
    "compute_structural_entropy_per_hour",
    "fit_local_power_law",
    "compute_alpha_beta_per_hour",
    "bootstrap_correlation_ci",
    "bootstrap_scaling_exponents",
    "permutation_test_optimal_scale",
    "ks_test_correlation_curves",
]
