from types import SimpleNamespace
from typing import Any

import pytest

from agent import Agent, AgentDecision, AgentState, ClaudeModel


class FakeMessages:
    def __init__(self) -> None:
        self.kwargs: dict = {}
        self.response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="model output")],
            stop_reason="end_turn",
        )

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> ClaudeModel:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    model = ClaudeModel("claude-haiku-4-5")
    fake = FakeMessages()
    model.client = SimpleNamespace(messages=fake)
    return model


def test_init_stores_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    model = ClaudeModel("claude-haiku-4-5")
    assert model.model_name == "claude-haiku-4-5"
    assert model.client is not None


def test__add_user_message(model: ClaudeModel) -> None:
    messages: list[dict[str, Any]] = []
    model._add_user_message(messages, "hello")
    assert messages == [{"role": "user", "content": "hello"}]


def test__add_assistant_message(model: ClaudeModel) -> None:
    messages: list[dict[str, Any]] = []
    model._add_assistant_message(messages, "hi there")
    assert messages == [{"role": "assistant", "content": "hi there"}]


def test_claude_model_run_returns_agent_decision_on_end_turn(model: ClaudeModel) -> None:
    model.client.messages.response.content = [
        SimpleNamespace(type="text", text="final answer")
    ]
    model.client.messages.response.stop_reason = "end_turn"

    decision = model.run(messages=[{"role": "user", "content": "test"}])
    assert isinstance(decision, AgentDecision)
    assert decision.final_answer == "final answer"
    assert decision.tool_call is None


def test_claude_model_run_returns_agent_decision_on_tool_use(model: ClaudeModel) -> None:
    model.client.messages.response.content = [
        SimpleNamespace(type="tool_use", name="test_tool", input={"param": "value"})
    ]
    model.client.messages.response.stop_reason = "tool_use"

    decision = model.run(messages=[{"role": "user", "content": "test"}])
    assert isinstance(decision, AgentDecision)
    assert decision.final_answer is None
    assert decision.tool_call is not None
    assert decision.tool_call.name == "test_tool"
    assert decision.tool_call.arguments == {"param": "value"}


def test_claude_model_run_passes_system_prompt(model: ClaudeModel) -> None:
    model.run(messages=[{"role": "user", "content": "test"}], system="You are a test.")
    assert model.client.messages.kwargs["system"] == "You are a test."


def test_claude_model_run_passes_tools(model: ClaudeModel) -> None:
    tools_list = ["search_knowledge"]
    model.run(messages=[{"role": "user", "content": "test"}], tools=tools_list)
    assert model.client.messages.kwargs["tools"] == [{"name": "search_knowledge"}]


def test_agent_init() -> None:
    mock_model = SimpleNamespace(name="mock_model")
    agent = Agent(model=mock_model, max_iterations=10)
    assert agent.model == mock_model
    assert agent.max_iterations == 10


def test_agent_run_stub() -> None:
    mock_model = SimpleNamespace(name="mock_model")
    agent = Agent(model=mock_model)
    assert agent.run("goal") is None  # empty run returns None


def test_agent_state_init() -> None:
    state = AgentState(goal="test goal")
    assert state.goal == "test goal"
    assert state.iterations == 0
    assert state.completed is False
    assert state.history == []


def test_agent_state_with_custom_history() -> None:
    history = [{"role": "user", "content": "hi"}]
    state = AgentState(goal="test goal", history=history)
    assert state.history == history