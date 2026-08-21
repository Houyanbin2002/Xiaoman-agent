"""Dashboard management API composition.

The package keeps the historical public import surface small while the route
implementations remain grouped by the runtime capability they manage.
"""

from .contracts import DashboardRuntimeServices
from .registration import register_dashboard_management

__all__ = ["DashboardRuntimeServices", "register_dashboard_management"]
