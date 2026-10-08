from agents.decision import (
    AgentDecision,
    ToolCall,
)
from models.base import Model
from state.state import AgentState


class FakeModel(Model):
    """
    Deterministic model used to demonstrate the agent loop.

    First:
        request search_knowledge
    Then:
        use the observation to produce the final answer
    """

    def decide(self, state: AgentState) -> AgentDecision:
        tool_results = [
            event
            for event in state.history
            if event.get("type") == "tool_result"
        ]

        if not tool_results:
            return AgentDecision(
                tool_call=ToolCall(
                    name="search_knowledge",
                    arguments={"query": state.goal},
                )
            )

        latest_result = tool_results[-1]["result"]
        return AgentDecision(
            final_answer=(
                "I researched the goal and found the following "
                f"information:\n\n{latest_result}"
            )
        )