"""Optional process-side visualisation (off by default; enabled via ``--viz``)."""

from .diagnostics import plot_city_convergence_heatmap, plot_window_diagnostics

__all__ = ["plot_window_diagnostics", "plot_city_convergence_heatmap"]
