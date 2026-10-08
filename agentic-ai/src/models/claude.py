
from anthropic import Anthropic
from dotenv import load_dotenv

from agents.decision import AgentDecision, ToolCall
from models.base import Model
from state.state import AgentState

load_dotenv()

MODEL = "claude-haiku-4-5"


class ClaudeModel(Model):
    def __init__(self, model_name: str = MODEL) -> None:
        self.model_name = model_name
        self.client = Anthropic(api_key=None)

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