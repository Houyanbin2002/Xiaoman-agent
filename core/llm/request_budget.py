"""Optional execution-owned hook before a provider transport retry."""

from contextvars import ContextVar
from typing import Callable

before_transport_retry: ContextVar[Callable[[], None] | None] = ContextVar(
    "before_transport_retry", default=None
)
