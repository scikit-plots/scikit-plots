"""
Serve a Real Hamlet Corpus over MCP with Annoy
==============================================

.. currentmodule:: scikitplot.mcp

This example connects three public scikit-plots subsystems in one realistic,
offline-first workflow:

``scikitplot.corpus``
    provides :data:`~scikitplot.corpus.HAMLET_TEXT` and the deterministic
    :class:`~scikitplot.corpus.HashEmbedder`;

``scikitplot.annoy``
    provides the explicitly requested native ANN backend;

``scikitplot.mcp``
    exposes the resulting corpus through the read-only ``search_docs`` MCP tool
    and ``docs://chunk/{doc_id}`` resource surface.

The automated path mirrors two terminals in CI:

**Terminal 1 — server**

.. code-block:: bash

   scikitplot mcp --docker \
       --host 127.0.0.1 \
       --corpus-annoy /tmp/scikitplot-mcp-hamlet \
       --hash-dimension 256 \
       --annoy-n-trees 10

**Terminal 2 — client**

.. code-block:: python

   from mcp import Client

   async with Client("http://127.0.0.1:8000/mcp") as client:
       result = await client.call_tool(
           "search_docs",
           {"query": "sleep dream death", "k": 3},
       )

The gallery does not open a server during a normal documentation build.
Set ``SCIKITPLOT_GALLERY_RUN_MCP_DOCKER=1`` to execute the real Streamable
HTTP subprocess round trip in CI or an intentionally provisioned environment.

The local Corpus + Annoy retriever and CLI ``--self-test`` do not require the
MCP SDK. If the optional native Annoy capability is unavailable, those sections
report a specific ``SKIP``. Once Annoy imports successfully, unexpected index
or query failures remain visible.

.. code-block:: text

    The showcase implements this actual flow:

    HAMLET_TEXT
        ↓
    temporary local corpus
        ↓
    HashEmbedder
        ↓
    Corpus RetrievalIndex
        ↓
    Annoy backend
        ↓
    CorpusAnnoyRetriever
        ↓
    scikitplot mcp --docker
        ↓
    MCP Client context
        ↓
    search_docs(...)
        ↓
    print results + citations
        ↓
    clean server shutdown

.. code-block:: text

    The same embedder instance/contract is used for both document vectors and query vectors:

    HAMLET documents
        ↓
    HashEmbedder
        ↓
    Annoy vectors

    query
        ↓
    same HashEmbedder
        ↓
    Annoy nearest-neighbor query

.. code-block:: text

    The example itself:

    Popen(scikitplot mcp --docker ...)
            ↓
    poll /healthz
            ↓
    wait until healthy
            ↓
    with server context
            ↓
    run MCP client subprocess
            ↓
    async with Client(...)
            ↓
    call search_docs
            ↓
    print bounded results
            ↓
    terminate server
            ↓
    kill only if graceful shutdown times out
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from scikitplot.corpus import HAMLET_TEXT, HashEmbedder
from scikitplot.mcp import (
    CorpusAnnoyRetriever,
    server_runtime_status,
)


# %%
# 1. Create a real local corpus
# -----------------------------
# The corpus is a normal directory because the MCP Corpus+Annoy profile accepts
# the same local documentation/source shape used by :class:`CorpusBuilder`.

_WORKSPACE = tempfile.TemporaryDirectory(prefix="scikitplot-mcp-hamlet-")
_CORPUS_DIR = Path(_WORKSPACE.name) / "corpus"
_CORPUS_DIR.mkdir()

_HAMLET_PATH = _CORPUS_DIR / "hamlet.txt"
_HAMLET_PATH.write_text(HAMLET_TEXT, encoding="utf-8")

_QUERY = "sleep dream death"
_HASH_DIMENSION = 256
_ANNOY_N_TREES = 10

print("Corpus directory:", _CORPUS_DIR)
print("Corpus file:", _HAMLET_PATH.name)
print("Characters:", len(HAMLET_TEXT))
print("Query:", _QUERY)


# %%
# 2. Check the optional Annoy capability
# --------------------------------------
# ``scikitplot.annoy`` is optional at runtime because native-extension support
# varies by platform/build.  ImportError is therefore a capability SKIP.
# Other exception types are not hidden.


def _probe_annoy() -> tuple[bool, str]:
    try:
        from scikitplot.annoy import Index  # noqa: F401
    except ImportError as exc:
        return False, f"native Annoy capability is unavailable ({exc})"
    return True, "Annoy is importable"


annoy_ready, annoy_reason = _probe_annoy()
print("Annoy:", annoy_reason)


# %%
# 3. Build and query Corpus + Annoy directly
# ------------------------------------------
# This is the same retriever that the MCP server will own.  HashEmbedder keeps
# the path deterministic and avoids a model/network dependency.

retriever = None

if not annoy_ready:
    print(f"[SKIP] Direct Corpus+Annoy retrieval: {annoy_reason}")
else:
    retriever = CorpusAnnoyRetriever.from_corpus_annoy(
        str(_CORPUS_DIR),
        embedder=HashEmbedder(dimension=_HASH_DIMENSION),
        backend="annoy",
        metric="angular",
        n_trees=_ANNOY_N_TREES,
        strict=True,
    )

    hits = retriever.search(_QUERY, k=3)

    print(f"Direct hits: {len(hits)}")
    for rank, hit in enumerate(hits, start=1):
        print(
            f"[{rank}] score={hit.score:.4f} "
            f"doc_id={hit.doc_id} source={hit.source_uri!r}"
        )
        print(f"    {hit.text[:180]!r}")

    if hits:
        resource_doc = retriever.get(hits[0].doc_id)
        print("Resource lookup:", resource_doc.doc_id if resource_doc else None)


# %%
# 4. Exercise the centralized CLI without starting a server
# ----------------------------------------------------------
# ``--self-test`` loads the exact configured backend, performs one bounded
# read-only search, validates the MCP result contract, prints JSON, and exits.
# It deliberately stays on the SDK-free retrieval tier.


def _scikitplot_cli() -> list[str]:
    executable = shutil.which("scikitplot")
    if executable:
        return [executable]
    return [sys.executable, "-m", "scikitplot"]


_CLI = _scikitplot_cli()

self_test_payload = None

if not annoy_ready:
    print(f"[SKIP] CLI Corpus+Annoy self-test: {annoy_reason}")
else:
    command = [
        *_CLI,
        "mcp",
        "--corpus-annoy",
        str(_CORPUS_DIR),
        "--hash-dimension",
        str(_HASH_DIMENSION),
        "--annoy-n-trees",
        str(_ANNOY_N_TREES),
        "--annoy-metric",
        "angular",
        "--self-test",
        "--self-test-query",
        _QUERY,
        "--self-test-require-match",
        "--log-level",
        "WARNING",
    ]

    print("CLI self-test command:")
    print(" ".join(command))

    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    self_test_payload = json.loads(completed.stdout)
    print("Self-test count:", self_test_payload["count"])

    for citation in self_test_payload["citations"][:3]:
        print(
            "  citation:",
            citation["doc_id"],
            f"score={citation['score']:.4f}",
            citation["source_uri"],
        )


# %%
# 5. Define a bounded server subprocess context
# ---------------------------------------------
# The server is always terminated, and then killed only if graceful termination
# exceeds the bounded cleanup timeout.


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(
    url: str,
    process: subprocess.Popen,
    log_path: Path,
    *,
    timeout: float = 30.0,
) -> None:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(
                f"MCP server exited before becoming healthy "
                f"(code={returncode}).\n{log_text[-4000:]}"
            )

        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if response.status == 200 and payload.get("status") == "ok":
                return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.2)

    raise TimeoutError(f"MCP server did not become healthy within {timeout:.0f}s")


@contextmanager
def _running_mcp_server():
    port = _free_loopback_port()
    mcp_url = f"http://127.0.0.1:{port}/mcp"
    health_url = f"http://127.0.0.1:{port}/healthz"
    log_path = Path(_WORKSPACE.name) / "mcp-server.log"

    command = [
        *_CLI,
        "mcp",
        "--docker",
        # Keep the automated gallery/CI server loopback-only even though the
        # Docker profile normally defaults to 0.0.0.0.
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--corpus-annoy",
        str(_CORPUS_DIR),
        "--hash-dimension",
        str(_HASH_DIMENSION),
        "--annoy-n-trees",
        str(_ANNOY_N_TREES),
        "--annoy-metric",
        "angular",
        "--log-level",
        "WARNING",
    ]

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            _wait_for_health(health_url, process, log_path)
            print("Server healthy:", health_url)
            yield mcp_url
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

            print("Server exit code:", process.returncode)


# %%
# 6. Query the real Docker-profile server through MCP Client
# ----------------------------------------------------------
# The client is executed in a small child Python process.  That keeps the
# generated Jupyter notebook safe from ``asyncio.run()``/already-running-loop
# conflicts while still exercising the official SDK's async context manager.

_RUN_LIVE = os.environ.get(
    "SCIKITPLOT_GALLERY_RUN_MCP_DOCKER", "1"
).strip().casefold() in {
    "1",
    "true",
    "yes",
    "on",
}

server_status = server_runtime_status()

if not _RUN_LIVE:
    print(
        "[SKIP] Live MCP HTTP round trip: "
        "set SCIKITPLOT_GALLERY_RUN_MCP_DOCKER=1 to opt in."
    )
elif not annoy_ready:
    print(f"[SKIP] Live MCP HTTP round trip: {annoy_reason}")
elif not server_status["server_available"]:
    print(
        "[SKIP] Live MCP HTTP round trip: "
        f"{server_status['reason']}. Install scikit-plots[mcp]."
    )
else:
    client_script = Path(_WORKSPACE.name) / "mcp_client.py"
    client_script.write_text(
        """\
import asyncio
import json
import sys

from mcp import Client


async def main():
    url, query = sys.argv[1], sys.argv[2]

    async with Client(url) as client:
        result = await client.call_tool(
            "search_docs",
            {"query": query, "k": 3},
        )

    content = result.structured_content
    print(json.dumps(content, indent=2, sort_keys=True))


asyncio.run(main())
""",
        encoding="utf-8",
    )

    with _running_mcp_server() as mcp_url:
        completed = subprocess.run(
            [sys.executable, str(client_script), mcp_url, _QUERY],
            check=True,
            capture_output=True,
            text=True,
        )

        live_payload = json.loads(completed.stdout)

        print("MCP query:", live_payload["query"])
        print("MCP result count:", live_payload["count"])

        for citation, passage in zip(
            live_payload["citations"][:3],
            live_payload["passages"][:3],
        ):
            print(
                f"  [{citation['n']}] score={citation['score']:.4f} "
                f"doc_id={citation['doc_id']} "
                f"source={citation['source_uri']!r}"
            )
            print("     ", passage[:220].replace("\n", " "))


# %%
# 7. CI / Docker usage
# --------------------
# The complete live round trip can be enabled in CI with:
#
# .. code-block:: bash
#
#    export SCIKITPLOT_GALLERY_RUN_MCP_DOCKER=1
#    python galleries/examples/mcp/plot_mcp_corpus_annoy_hamlet_script.py
#
# Or run the two terminals manually:
#
# .. code-block:: bash
#
#    mkdir -p /tmp/scikitplot-mcp-hamlet
#    python - <<'PY'
#    from pathlib import Path
#    from scikitplot.corpus import HAMLET_TEXT
#
#    Path("/tmp/scikitplot-mcp-hamlet/hamlet.txt").write_text(
#        HAMLET_TEXT,
#        encoding="utf-8",
#    )
#    PY
#
#    scikitplot mcp --docker \
#        --corpus-annoy /tmp/scikitplot-mcp-hamlet \
#        --hash-dimension 256 \
#        --annoy-n-trees 10
#
# A second terminal/client can then connect to ``http://127.0.0.1:8000/mcp``.


# %%
# 8. Cleanup
# ----------
# No persistent index or corpus state is required by this showcase.

_WORKSPACE.cleanup()
print("Temporary Hamlet MCP workspace cleaned.")


# %%
#
# .. tags::
#
#    model-workflow: corpus
#    model-workflow: mcp
#    plot-type: text
#    level: advanced
#    purpose: showcase
