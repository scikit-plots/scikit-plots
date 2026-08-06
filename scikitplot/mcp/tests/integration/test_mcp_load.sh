#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Repeatable bounded load test. It performs no writes and leaves no state behind.
set -Eeuo pipefail

: "${SCIKITPLOT_MCP_TEST_URL:=http://127.0.0.1:8000/mcp}"
: "${SCIKITPLOT_MCP_LOAD_REQUESTS:=200}"
: "${SCIKITPLOT_MCP_LOAD_CONCURRENCY:=20}"
: "${SCIKITPLOT_MCP_LOAD_TIMEOUT:=120}"

export SCIKITPLOT_MCP_TEST_URL
export SCIKITPLOT_MCP_LOAD_REQUESTS
export SCIKITPLOT_MCP_LOAD_CONCURRENCY
export SCIKITPLOT_MCP_LOAD_TIMEOUT

python - <<'PY'
import importlib.metadata
import platform

print("Python:", platform.python_version())

for package in ("mcp", "pydantic", "starlette", "uvicorn", "httpx", "pytest"):
    try:
        print(f"{package}: {importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package}: not installed")
PY

python - <<'PY'
from __future__ import annotations

import asyncio
import os
import statistics
import time
from typing import Any

from mcp import Client

URL = os.environ["SCIKITPLOT_MCP_TEST_URL"]
REQUESTS = int(os.environ["SCIKITPLOT_MCP_LOAD_REQUESTS"])
CONCURRENCY = int(os.environ["SCIKITPLOT_MCP_LOAD_CONCURRENCY"])
TIMEOUT = float(os.environ["SCIKITPLOT_MCP_LOAD_TIMEOUT"])

if not 1 <= REQUESTS <= 10_000:
    raise SystemExit("SCIKITPLOT_MCP_LOAD_REQUESTS must be between 1 and 10000")
if not 1 <= CONCURRENCY <= 256:
    raise SystemExit("SCIKITPLOT_MCP_LOAD_CONCURRENCY must be between 1 and 256")
if not 1 <= TIMEOUT <= 3600:
    raise SystemExit("SCIKITPLOT_MCP_LOAD_TIMEOUT must be between 1 and 3600 seconds")


def assert_contract(content: dict[str, Any], expected_query: str) -> None:
    assert content["query"] == expected_query
    assert content["count"] == len(content["passages"]) == len(content["citations"])
    assert content["count"] <= 3
    assert content["security"]["untrusted_content"] is True


async def main() -> None:
    semaphore = asyncio.Semaphore(CONCURRENCY)
    durations: list[float] = []
    failures: list[str] = []
    baseline: dict[str, Any] | None = None

    async with Client(URL) as client:
        first = await client.call_tool("search_docs", {"query": "transport", "k": 3})
        if first.is_error or not isinstance(first.structured_content, dict):
            raise RuntimeError(f"baseline call failed: {first!r}")
        baseline = first.structured_content
        assert_contract(baseline, "transport")

        async def request(number: int) -> None:
            query = "transport" if number % 2 == 0 else "security"
            async with semaphore:
                started = time.perf_counter()
                try:
                    result = await client.call_tool(
                        "search_docs",
                        {"query": query, "k": 3},
                    )
                    if result.is_error or not isinstance(result.structured_content, dict):
                        raise RuntimeError("tool returned an error or no structured content")
                    assert_contract(result.structured_content, query)
                    if query == "transport" and result.structured_content != baseline:
                        raise AssertionError("same read-only query returned a different result")
                except Exception as exc:  # collect all failures before reporting
                    failures.append(f"request {number}: {type(exc).__name__}: {exc}")
                finally:
                    durations.append(time.perf_counter() - started)

        await asyncio.wait_for(
            asyncio.gather(*(request(number) for number in range(REQUESTS))),
            timeout=TIMEOUT,
        )

    ordered = sorted(durations)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
        return ordered[index]

    print(f"url:         {URL}")
    print(f"requests:    {REQUESTS}")
    print(f"concurrency: {CONCURRENCY}")
    print(f"failures:    {len(failures)}")
    print(f"mean:        {statistics.mean(durations):.4f}s")
    print(f"p50:         {percentile(0.50):.4f}s")
    print(f"p95:         {percentile(0.95):.4f}s")
    print(f"p99:         {percentile(0.99):.4f}s")

    if failures:
        print("\nFirst failures:")
        for failure in failures[:20]:
            print(f"- {failure}")
        raise SystemExit(1)


asyncio.run(main())
PY
