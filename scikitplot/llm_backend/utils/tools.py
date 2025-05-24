"""tools."""

# llm_backend/utils/tools.py
from typing import TypedDict


def search_wikipedia(query: str) -> str:
    """
    Simulate a search on Wikipedia for a given query.

    Args:
        query (str): The topic to search on Wikipedia.

    Returns
    -------
        str: A summary string simulating Wikipedia content.
    """
    # Simulate an API call (replace with actual API logic if needed)
    return f"Summary of '{query}' from Wikipedia."


# Add OpenAI-compatible function schema as a typed attribute
class SearchWikipediaSchema(TypedDict):
    """SearchWikipediaSchema."""

    type: str
    properties: dict
    required: list[str]


search_wikipedia.openai_schema: "type[SearchWikipediaSchema]" = {  # type: ignore  # noqa: PGH003
    "type": "object",
    "properties": {"query": {"type": "string", "description": "The topic to search"}},
    "required": ["query"],
}
