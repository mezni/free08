from models.fake import FakeModel
from runtime.runtime import AgentRuntime
from tools.simple import search_knowledge


def main() -> None:
    model = FakeModel()
    tools = {
        "search_knowledge": search_knowledge,
    }
    runtime = AgentRuntime(
        model=model,
        tools=tools,
        max_iterations=5,
    )
    result = runtime.run(
        "What are important security principles "
        "for an agentic AI system?"
    )
    print(result)


if __name__ == "__main__":
    main()