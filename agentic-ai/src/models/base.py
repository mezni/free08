from abc import ABC, abstractmethod

from agents.decision import AgentDecision
from state.state import AgentState


class Model(ABC):
    """Abstraction over the model used by an agent."""

    @abstractmethod
    def decide(self, state: AgentState) -> AgentDecision:
        """Choose the next action given the current state."""
        raise NotImplementedError