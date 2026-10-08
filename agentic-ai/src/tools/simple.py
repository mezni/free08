def search_knowledge(query: str) -> str:
    """Return simulated knowledge for Day 1."""
    knowledge = {
        "rag": (
            "Enterprise RAG should combine retrieval, "
            "authorization-aware access, grounding, and evaluation."
        ),
        "security": (
            "Agent tools should follow least privilege and "
            "sensitive actions should be controlled by policy."
        ),
    }

    query_lower = query.lower()
    for topic, content in knowledge.items():
        if topic in query_lower:
            return content

    return f"No knowledge found for query: {query}"