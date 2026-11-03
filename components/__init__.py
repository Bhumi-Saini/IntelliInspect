"""UI components package for IntelliInspect."""
from .dashboard import render_dashboard
from .inspection import render_inspection
from .history import render_history
from .performance import render_performance
from .about import render_about

__all__ = [
    "render_dashboard",
    "render_inspection",
    "render_history",
    "render_performance",
    "render_about"
]
