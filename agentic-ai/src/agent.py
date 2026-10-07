import os
from dataclasses import dataclass, field
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-4-5"

TOOL_SCHEMAS: list[dict[str, Any]] = []


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentDecision:
    final_answer: str | None = None
    tool_call: ToolCall | None = None


@dataclass
class AgentState:
    goal: str
    iterations: int = 0
    completed: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)


# 1. Python implementation of the tool
def search_knowledge(query: str) -> str:
    """Simulates searching a database or vector store."""
    return f"Knowledge Base Results for '{query}': Found 3 matching articles on system design."


# 2. Registry mapping string names to executable functions
TOOLS = {"search_knowledge": search_knowledge}


# 3. Schema sent to Claude API
CLAUDE_TOOLS = [
    {
        "name": "search_knowledge",
        "description": "Searches the internal knowledge base for articles and documentation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search term or subject to look up."}
            },
            "required": ["query"],
        },
    }
]


class ClaudeModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def _add_user_message(self, messages: list[dict[str, Any]], text: str) -> None:
        messages.append({"role": "user", "content": text})

    def _add_assistant_message(self, messages: list[dict[str, Any]], text: str) -> None:
        messages.append({"role": "assistant", "content": text})

    def run(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[str] | None = None,
    ) -> str | AgentDecision:
        request: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": 1024,
            "messages": messages,
        }
        if system is not None:
            request["system"] = system
        if tools:
            request["tools"] = [{"name": t} for t in tools]

        response = self.client.messages.create(**request)

        text = "".join(block.text for block in response.content if block.type == "text")

        for block in response.content:
            if block.type == "tool_use":
                tool_name = block.name
                tool_args = block.input
                if tool_name in TOOLS:
                    result = TOOLS[tool_name](**tool_args)
                    return AgentDecision(
                        tool_call=ToolCall(name=tool_name, arguments=tool_args),
                        final_answer=result,
                    )
                return AgentDecision(tool_call=ToolCall(name=tool_name, arguments=tool_args))

        if response.stop_reason == "end_turn":
            return AgentDecision(final_answer=text)

        return AgentDecision(final_answer=text)


class Agent:
    def __init__(self, model: ClaudeModel, max_iterations: int = 5) -> None:
        self.model = model
        self.max_iterations = max_iterations

    def run(self, goal: str) -> str:
        pass


if __name__ == "__main__":
    model = ClaudeModel(MODEL)
    agent = Agent(model=model, max_iterations=5)

    goal = (
        "Search the knowledge base for microservice deployment best practices "
        "and summarize the result."
    )
    print(f"Goal: {goal}\n" + "-" * 50)

    output = agent.run(goal)
    print("-" * 50)
    print("Final Output:\n", output if output else "(Agent.run is stub – no output)")
