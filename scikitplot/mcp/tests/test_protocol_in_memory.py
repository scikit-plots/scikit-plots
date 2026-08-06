# SPDX-License-Identifier: BSD-3-Clause
"""Self-contained protocol tests using the real MCP SDK with no network port."""
from __future__ import annotations

import pytest

mcp = pytest.importorskip("mcp")
from mcp import Client  # noqa: E402

from scikitplot.mcp._demo import builtin_demo_retriever  # noqa: E402
from scikitplot.mcp._server import create_server  # noqa: E402


@pytest.fixture
def protocol_server():
    retriever = builtin_demo_retriever()
    return create_server(
        retriever,
        document_reader=retriever.get,
        health_path=None,
    )


@pytest.mark.asyncio
async def test_real_sdk_tool_discovery_and_idempotent_call(protocol_server) -> None:
    async with Client(protocol_server) as client:
        tools = await client.list_tools()
        search = next(tool for tool in tools.tools if tool.name == "search_docs")
        assert search.input_schema["type"] == "object"
        assert search.input_schema["additionalProperties"] is False
        assert search.output_schema["type"] == "object"
        assert search.output_schema["additionalProperties"] is False

        arguments = {"query": "transport", "k": 2}
        first = await client.call_tool("search_docs", arguments)
        second = await client.call_tool("search_docs", arguments)

    assert not first.is_error
    assert not second.is_error
    assert first.structured_content == second.structured_content

    content = first.structured_content
    assert content is not None
    assert content["count"] == len(content["passages"]) == len(content["citations"])
    assert content["count"] >= 1
    assert content["message"] is None
    assert content["security"]["untrusted_content"] is True


@pytest.mark.asyncio
async def test_real_sdk_empty_result_contract(protocol_server) -> None:
    async with Client(protocol_server) as client:
        result = await client.call_tool(
            "search_docs",
            {"query": "zzzz_nonexistent_canary_query_8d9c45", "k": 3},
        )

    assert not result.is_error
    content = result.structured_content
    assert content is not None
    assert content["count"] == 0
    assert content["passages"] == []
    assert content["citations"] == []
    assert content["message"]


@pytest.mark.asyncio
async def test_real_sdk_default_limit_matches_explicit_default(protocol_server) -> None:
    async with Client(protocol_server) as client:
        implicit = await client.call_tool("search_docs", {"query": "transport"})
        explicit = await client.call_tool(
            "search_docs",
            {"query": "transport", "k": 5},
        )

    assert not implicit.is_error
    assert not explicit.is_error
    assert implicit.structured_content == explicit.structured_content


@pytest.mark.asyncio
async def test_real_sdk_resource_read_is_repeatable(protocol_server) -> None:
    async with Client(protocol_server) as client:
        first = await client.read_resource("docs://chunk/transport")
        second = await client.read_resource("docs://chunk/transport")

    first_text = [item.text for item in first.contents]
    second_text = [item.text for item in second.contents]
    assert first_text == second_text
    assert first_text and "Transport choices" in first_text[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"query": None, "k": 2},
        {"query": True, "k": 2},
        {"query": ["transport"], "k": 2},
        {"query": "transport", "k": True},
        {"query": "transport", "k": False},
        {"query": "transport", "k": "2"},
        {"query": "transport", "k": 2.0},
        {"query": "transport", "k": None},
        {"query": "transport", "k": []},
        {"query": "transport", "k": 2, "unexpected": "value"},
    ],
)
async def test_real_sdk_rejects_coercible_input_types(protocol_server, arguments) -> None:
    """Exercise validation before Python can coerce ``bool`` into ``int``."""
    async with Client(protocol_server) as client:
        try:
            result = await client.call_tool("search_docs", arguments)
        except Exception:
            return

    assert result.is_error, f"coercible invalid input unexpectedly succeeded: {result!r}"
