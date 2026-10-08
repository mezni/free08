from collections.abc import Callable
from typing import Any

from models.base import Model
from state.state import AgentState

ToolFunction = Callable[..., Any]


class AgentRuntime:
    """Runs the agent decision-action-observation loop."""

    def __init__(
        self,
        model: Model,
        tools: dict[str, ToolFunction],
        max_iterations: int = 5,
    ) -> None:
        self.model = model
        self.tools = tools
        self.max_iterations = max_iterations

    def run(self, goal: str) -> str:
        state = AgentState(goal=goal)

        while not state.completed:
            if state.iteration >= self.max_iterations:
                return (
                    "Agent stopped because the maximum "
                    "number of iterations was reached."
                )

            decision = self.model.decide(state)
            state.next_iteration()
            state.record(
                {
                    "type": "model_decision",
                    "iteration": state.iteration,
                    "has_tool_call": decision.tool_call is not None,
                }
            )

            # Model has completed the task.
            if decision.final_answer is not None:
                state.complete()
                state.record(
                    {
                        "type": "agent_completed",
                        "answer": decision.final_answer,
                    }
                )
                return decision.final_answer

            # Otherwise, our AgentDecision invariant guarantees
            # the decision contains a tool call.
            tool_call = decision.tool_call
            if tool_call is None:
                raise RuntimeError(
                    "Invalid agent decision: no action available."
                )

            state.record(
                {
                    "type": "tool_requested",
                    "tool": tool_call.name,
                    "arguments": tool_call.arguments,
                }
            )

            tool = self.tools.get(tool_call.name)
            if tool is None:
                state.record(
                    {
                        "type": "tool_error",
                        "tool": tool_call.name,
                        "error": "Unknown tool",
                    }
                )
                continue

            try:
                result = tool(**tool_call.arguments)
            except Exception as exc:
                state.record(
                    {
                        "type": "tool_error",
                        "tool": tool_call.name,
                        "error": str(exc),
                    }
                )
                continue

            state.record(
                {
                    "type": "tool_result",
                    "tool": tool_call.name,
                    "result": result,
                }
            )

        return "Agent stopped without producing a final answer."