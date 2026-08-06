# tests/test_server.py
# SPDX-License-Identifier: BSD-3-Clause
"""MCP server-shell tests using a lightweight SDK double."""
from __future__ import annotations

import asyncio
import inspect
import json
import sys
import types
from pathlib import Path
from typing import get_type_hints

import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, create_model
from starlette.requests import Request

_PKG_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PKG_ROOT))

from scikitplot.mcp._demo import builtin_demo_retriever  # noqa: E402
from scikitplot.mcp._server import SearchDocsOutput, SearchService, create_server  # noqa: E402
from scikitplot.mcp._version import __version__  # noqa: E402


class FakeToolAnnotations:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeArgumentModelBase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class FakeToolManager:
    def __init__(self):
        self._tools = {}

    def get_tool(self, name):
        return self._tools.get(name)


class FakeMCPServer:
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self.tools = {}
        self.resources = {}
        self.routes = []
        self._tool_manager = FakeToolManager()

    def tool(self, **kwargs):
        def decorator(func):
            self.tools[kwargs["name"]] = (func, kwargs)
            hints = get_type_hints(func, include_extras=True)
            fields = {}
            for parameter in inspect.signature(func).parameters.values():
                annotation = hints.get(parameter.name, object)
                default = (
                    ...
                    if parameter.default is inspect.Parameter.empty
                    else parameter.default
                )
                fields[parameter.name] = (annotation, default)
            argument_model = create_model(
                f"{func.__name__}Arguments",
                __base__=FakeArgumentModelBase,
                **fields,
            )
            metadata = types.SimpleNamespace(arg_model=argument_model)
            registered = types.SimpleNamespace(
                fn=func,
                fn_metadata=metadata,
                parameters=argument_model.model_json_schema(by_alias=True),
            )
            self._tool_manager._tools[kwargs["name"]] = registered
            return func

        return decorator

    def resource(self, uri, **kwargs):
        def decorator(func):
            self.resources[uri] = (func, kwargs)
            return func

        return decorator

    def custom_route(self, path, methods, **kwargs):
        def decorator(func):
            self.routes.append((path, methods, kwargs, func))
            return func

        return decorator


def _install_fake_sdk(monkeypatch):
    mcp_module = types.ModuleType("mcp")
    server_module = types.ModuleType("mcp.server")
    server_module.MCPServer = FakeMCPServer
    types_module = types.ModuleType("mcp_types")
    types_module.ToolAnnotations = FakeToolAnnotations
    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.server", server_module)
    monkeypatch.setitem(sys.modules, "mcp_types", types_module)


def _request(path="/healthz"):
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8000),
        }
    )


def test_health_route_is_registered_and_minimal(monkeypatch):
    _install_fake_sdk(monkeypatch)
    retriever = builtin_demo_retriever()
    server = create_server(retriever, document_reader=retriever.get, health_path="/healthz")

    assert "search_docs" in server.tools
    assert "docs://chunk/{doc_id}" in server.resources
    assert len(server.routes) == 1
    path, methods, kwargs, handler = server.routes[0]
    assert path == "/healthz"
    assert methods == ["GET"]
    assert kwargs["include_in_schema"] is False

    response = asyncio.run(handler(_request()))
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    payload = json.loads(response.body)
    assert payload == {
        "status": "ok",
        "service": "scikitplot-docs",
        "version": __version__,
    }


def test_registered_tool_schema_forbids_unknown_arguments(monkeypatch):
    _install_fake_sdk(monkeypatch)
    server = create_server(builtin_demo_retriever(), health_path=None)
    registered = server._tool_manager.get_tool("search_docs")

    assert registered.parameters["additionalProperties"] is False
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        registered.fn_metadata.arg_model.model_validate(
            {"query": "transport", "k": 2, "unexpected": "value"}
        )


def test_registered_tool_arguments_are_strict_at_sdk_boundary(monkeypatch):
    """Reject values that Pydantic's normal ``int`` mode would coerce."""
    _install_fake_sdk(monkeypatch)
    server = create_server(builtin_demo_retriever(), health_path=None)
    tool, _metadata = server.tools["search_docs"]
    hints = get_type_hints(tool, include_extras=True)

    query_adapter = TypeAdapter(hints["query"])
    k_adapter = TypeAdapter(hints["k"])

    assert query_adapter.validate_python("transport") == "transport"
    assert k_adapter.validate_python(2) == 2

    for invalid_query in (True, 123, 1.5):
        with pytest.raises(ValidationError):
            query_adapter.validate_python(invalid_query)

    for invalid_k in (True, False, "2", 2.0):
        with pytest.raises(ValidationError):
            k_adapter.validate_python(invalid_k)


def test_health_route_can_be_disabled(monkeypatch):
    _install_fake_sdk(monkeypatch)
    server = create_server(builtin_demo_retriever(), health_path=None)
    assert server.routes == []


@pytest.mark.parametrize("bad", ["healthz", "/", "/health?full=1", "/health#fragment", "/health\\x"])
def test_invalid_health_path_rejected(monkeypatch, bad):
    _install_fake_sdk(monkeypatch)
    with pytest.raises(ValueError, match="health_path"):
        create_server(builtin_demo_retriever(), health_path=bad)


def test_search_service_empty_result_contract_is_consistent():
    class EmptyRetriever:
        def search(self, query, k=5):
            return []

    output = SearchService(EmptyRetriever()).search("missing", 3)
    assert output.count == 0
    assert output.passages == []
    assert output.citations == []
    assert output.message == "No matching documentation was found for this query."


def test_search_service_rejects_direct_type_coercion():
    service = SearchService(builtin_demo_retriever())
    with pytest.raises(ValueError, match="query must be a string"):
        service.search(123, 2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="k must be an integer"):
        service.search("transport", "2")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="k must be an integer"):
        service.search("transport", True)


@pytest.mark.parametrize("bad", [True, False, "4", 4.0, None])
def test_search_service_rejects_coerced_max_concurrency(bad):
    with pytest.raises(TypeError, match="max_concurrency must be an integer"):
        SearchService(builtin_demo_retriever(), max_concurrency=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [True, False, "0.1", None])
def test_search_service_rejects_coerced_acquire_timeout(bad):
    with pytest.raises(TypeError, match="acquire_timeout_seconds must be a number"):
        SearchService(
            builtin_demo_retriever(),
            acquire_timeout_seconds=bad,  # type: ignore[arg-type]
        )


def test_search_service_is_deterministic_for_immutable_backend():
    service = SearchService(builtin_demo_retriever())
    first = service.search("transport", 2).model_dump(mode="json")
    second = service.search("transport", 2).model_dump(mode="json")
    assert first == second
    assert first["count"] == len(first["passages"]) == len(first["citations"])


def test_non_empty_output_rejects_status_message():
    with pytest.raises(ValidationError, match="non-empty result must not include"):
        SearchDocsOutput.model_validate(
            {
                "query": "transport",
                "count": 1,
                "passages": ["UNTRUSTED REFERENCE DATA: example"],
                "citations": [
                    {
                        "n": 1,
                        "source_uri": "https://example.test/docs",
                        "title": "Example",
                        "anchor": "",
                        "doc_id": "example",
                        "score": 1.0,
                    }
                ],
                "message": "this must not coexist with results",
                "security": {
                    "untrusted_content": True,
                    "notice": "UNTRUSTED REFERENCE DATA: example",
                },
            }
        )


def test_output_models_reject_unknown_fields():
    payload = SearchService(builtin_demo_retriever()).search("transport", 1).model_dump()
    payload["unexpected"] = "value"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SearchDocsOutput.model_validate(payload)
