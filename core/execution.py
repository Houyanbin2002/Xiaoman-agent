"""Execution outcome shared by nested executors and durable task scheduling."""


class IncompleteExecutionError(RuntimeError):
    """Execution stopped without completion; do not blindly replay side effects."""

    def __init__(self, reason: str, partial_output: str = "") -> None:
        self.reason = reason
        self.partial_output = partial_output
        super().__init__(f"任务未完成（{reason}）。{partial_output}")
