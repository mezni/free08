from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    """Mutable execution state for a single agent run."""

    goal: str
    iteration: int = 0
    completed: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)

    def record(self, event: dict[str, Any]) -> None:
        self.history.append(event)

    def next_iteration(self) -> None:
        self.iteration += 1

    def complete(self) -> None:
        self.completed = True