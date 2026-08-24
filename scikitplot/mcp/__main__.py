# scikitplot/mcp/__main__.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Command-line entry point for the runnable MCP documentation server.

The CLI has two deployment profiles:

* default: local ``stdio`` transport;
* ``--docker``: Streamable HTTP on ``0.0.0.0`` with a public ``/healthz``
  liveness/readiness route.

``--docker`` is an explicit acknowledgement that the process is reachable on a
container network. It does not add authentication; publish the port only to a
trusted interface or place the server behind an authenticated reverse proxy.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, TextIO
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, build_opener

from ._capabilities import (  # SDK-free (no pydantic)
    server_capabilities,
    server_runtime_status,
)
from ._core import SearchCoordinator  # SDK-free (no pydantic)
from ._demo import InMemoryBm25Retriever, builtin_demo_retriever


# Server-tier factories exposed as module attributes so callers/tests reference and
# patch them here, but the pydantic + MCP-SDK import happens only when they are
# CALLED. This keeps --help / --print-effective-config / --list-capabilities
# base-install safe (no server-tier import), while --self-test / server startup
# import the server tier on demand.
def SearchService(*args, **kwargs):  # noqa: N802 (proxy keeps the class name)
    """Lazy proxy for :class:`scikitplot.mcp._server.SearchService`."""
    from ._server import (  # ruff: ignore[import-outside-top-level]
        SearchService as _SearchService,
    )

    return _SearchService(*args, **kwargs)


def create_server(*args, **kwargs):
    """Lazy proxy for :func:`scikitplot.mcp._server.create_server`."""
    from ._server import (  # ruff: ignore[import-outside-top-level]
        create_server as _create_server,
    )

    return _create_server(*args, **kwargs)


# NOTE: SearchService / create_server are imported lazily, only in the
# branches that build the server tier, so --help / --print-effective-config /
# --list-capabilities stay base-install safe (no pydantic/SDK import).

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
_REMOTE_BIND_HOSTS = {"0.0.0.0", "::"}  # ruff: ignore[hardcoded-bind-all-interfaces]
_DOC_ID_RE = re.compile(r"\A[A-Za-z0-9._:-]{1,200}\Z")
logger = logging.getLogger(__name__)

__all__ = [
    "RuntimeConfig",
    "main",
]


@dataclass(frozen=True)
class RuntimeConfig:
    """Validated effective CLI configuration."""

    transport: str
    docs_jsonl: str | None
    corpus_annoy: str | None
    corpus_embedding_model: str | None
    hash_dimension: int
    annoy_metric: str
    annoy_n_trees: int
    host: str
    port: int
    path: str
    health_path: str | None
    max_concurrency: int
    max_request_body: int
    stateful_http: bool
    allow_unauthenticated_remote: bool
    docker: bool
    log_level: str
    probe: bool
    probe_timeout: float
    self_test: bool
    self_test_query: str
    self_test_require_match: bool
    self_test_expected_doc_id: str | None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or probe the scikitplot documentation MCP server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default=None,
        help="MCP transport; defaults to streamable-http with --docker, otherwise stdio",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help=(
            "Use container defaults: streamable-http, host 0.0.0.0, and an enabled "
            "health route. This explicitly permits the container-network bind but "
            "does not add authentication."
        ),
    )
    parser.add_argument(
        "--docs-jsonl",
        help="Optional bounded JSONL corpus; built-in demo is the default",
    )
    parser.add_argument(
        "--corpus-annoy",
        default=None,
        metavar="PATH",
        help=(
            "Build a local scikitplot.corpus retriever from PATH and force the "
            "Annoy vector backend. Uses the deterministic HashEmbedder unless "
            "--corpus-embedding-model is supplied."
        ),
    )
    parser.add_argument(
        "--corpus-embedding-model",
        default=None,
        help=(
            "Optional model-backed embedding name for --corpus-annoy. "
            "Omit for the deterministic offline HashEmbedder."
        ),
    )
    parser.add_argument(
        "--hash-dimension",
        type=int,
        default=None,
        help="HashEmbedder dimension used by --corpus-annoy when no model is selected",
    )
    parser.add_argument(
        "--annoy-metric",
        default=None,
        help="Annoy metric used by --corpus-annoy",
    )
    parser.add_argument(
        "--annoy-n-trees",
        type=int,
        default=None,
        help="Annoy tree count used by --corpus-annoy",
    )
    parser.add_argument("--host", default=None, help="HTTP bind host")
    parser.add_argument("--port", type=int, default=None, help="HTTP bind port")
    parser.add_argument("--path", default=None, help="Streamable HTTP endpoint path")
    parser.add_argument(
        "--health-path",
        default=None,
        help="Unauthenticated HTTP health route; ignored by stdio",
    )
    parser.add_argument(
        "--no-health-endpoint",
        action="store_true",
        help="Disable the HTTP health route",
    )
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--max-request-body", type=int, default=None)
    parser.add_argument(
        "--stateful-http",
        action="store_true",
        help="Keep HTTP sessions; stateless is default",
    )
    parser.add_argument(
        "--allow-unauthenticated-remote",
        action="store_true",
        help="Explicitly permit a non-local bind without authentication",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="GET the configured health endpoint and exit instead of starting a server",
    )
    parser.add_argument(
        "--probe-timeout",
        type=float,
        default=None,
        help="Health-probe timeout in seconds",
    )
    parser.add_argument(
        "--print-effective-config",
        action="store_true",
        help="Print validated effective configuration as JSON and exit",
    )
    parser.add_argument(
        "--list-capabilities",
        action="store_true",
        help="Print the read-only tool/resource inventory as JSON and exit "
        "(does not import the MCP SDK or start a server)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Load the configured backend, run one deterministic read-only search, "
            "validate the output contract, print JSON, and exit"
        ),
    )
    parser.add_argument(
        "--self-test-query",
        default=None,
        help="Query used by --self-test",
    )
    parser.add_argument(
        "--self-test-require-match",
        action="store_true",
        help="Fail --self-test when the query returns zero passages",
    )
    parser.add_argument(
        "--self-test-expected-doc-id",
        default=None,
        help="Fail --self-test unless this exact citation doc_id is returned",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default=None,
    )
    return parser


def _env_value(environ: Mapping[str, str], name: str, default: str) -> str:
    value = environ.get(name)
    return value.strip() if value and value.strip() else default


def _env_int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer") from exc


def _env_float(environ: Mapping[str, str], name: str, default: float) -> float:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a number") from exc


def _env_bool(environ: Mapping[str, str], name: str, default: bool = False) -> bool:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"{name} must be one of: 1/0, true/false, yes/no, on/off")


def _normalize_path(value: str, option: str) -> str:
    path = value.strip()
    if not path.startswith("/"):
        raise SystemExit(f"{option} must start with '/'")
    if any(char in path for char in ("?", "#", "\\")):
        raise SystemExit(
            f"{option} must be a plain URL path without '?', '#', or backslashes"
        )
    if "//" in path:
        raise SystemExit(f"{option} must not contain empty path segments ('//')")
    if path != "/":
        path = path.rstrip("/")
    return path


def _resolve_config(  # ruff: ignore[too-many-branches]
    args: argparse.Namespace,
    *,
    environ: Mapping[str, str] | None = None,
) -> RuntimeConfig:
    env = os.environ if environ is None else environ

    env_docker = _env_bool(env, "SCIKITPLOT_MCP_DOCKER", False)
    docker = bool(args.docker or env_docker)

    default_transport = "streamable-http" if docker or args.probe else "stdio"
    transport = args.transport or _env_value(
        env, "SCIKITPLOT_MCP_TRANSPORT", default_transport
    )
    if transport not in {"stdio", "streamable-http"}:
        raise SystemExit(
            "SCIKITPLOT_MCP_TRANSPORT must be 'stdio' or 'streamable-http'"
        )

    corpus_annoy_raw = args.corpus_annoy
    if corpus_annoy_raw is None:
        corpus_annoy_raw = env.get("SCIKITPLOT_MCP_CORPUS_ANNOY")
    corpus_annoy = (
        corpus_annoy_raw.strip()
        if isinstance(corpus_annoy_raw, str) and corpus_annoy_raw.strip()
        else None
    )

    corpus_embedding_model_raw = args.corpus_embedding_model
    if corpus_embedding_model_raw is None:
        corpus_embedding_model_raw = env.get("SCIKITPLOT_MCP_CORPUS_EMBEDDING_MODEL")
    corpus_embedding_model = (
        corpus_embedding_model_raw.strip()
        if isinstance(corpus_embedding_model_raw, str)
        and corpus_embedding_model_raw.strip()
        else None
    )
    hash_dimension = (
        args.hash_dimension
        if args.hash_dimension is not None
        else _env_int(env, "SCIKITPLOT_MCP_HASH_DIMENSION", 256)
    )
    annoy_metric = (
        args.annoy_metric or _env_value(env, "SCIKITPLOT_MCP_ANNOY_METRIC", "angular")
    ).strip()
    annoy_n_trees = (
        args.annoy_n_trees
        if args.annoy_n_trees is not None
        else _env_int(env, "SCIKITPLOT_MCP_ANNOY_N_TREES", 10)
    )

    default_host = (
        "0.0.0.0"  # ruff: ignore[hardcoded-bind-all-interfaces]
        if docker
        else "127.0.0.1"
    )
    host = (args.host or _env_value(env, "SCIKITPLOT_MCP_HOST", default_host)).strip()
    if not host or any(char.isspace() for char in host) or "/" in host:
        raise SystemExit(
            "--host must be a hostname or IP address without whitespace or a URL scheme"
        )

    port = (
        args.port
        if args.port is not None
        else _env_int(env, "SCIKITPLOT_MCP_PORT", 8000)
    )
    path = _normalize_path(
        args.path or _env_value(env, "SCIKITPLOT_MCP_PATH", "/mcp"),
        "--path",
    )

    if args.no_health_endpoint:
        health_path = None
    else:
        health_path = _normalize_path(
            args.health_path
            or _env_value(env, "SCIKITPLOT_MCP_HEALTH_PATH", "/healthz"),
            "--health-path",
        )

    max_concurrency = (
        args.max_concurrency
        if args.max_concurrency is not None
        else _env_int(env, "SCIKITPLOT_MCP_MAX_CONCURRENCY", 4)
    )
    max_request_body = (
        args.max_request_body
        if args.max_request_body is not None
        else _env_int(env, "SCIKITPLOT_MCP_MAX_REQUEST_BODY", 1024 * 1024)
    )
    probe_timeout = (
        args.probe_timeout
        if args.probe_timeout is not None
        else _env_float(env, "SCIKITPLOT_MCP_PROBE_TIMEOUT", 2.0)
    )
    allow_remote = bool(
        args.allow_unauthenticated_remote
        or docker
        or _env_bool(env, "SCIKITPLOT_MCP_ALLOW_UNAUTHENTICATED_REMOTE", False)
    )
    log_level = (
        args.log_level or _env_value(env, "SCIKITPLOT_MCP_LOG_LEVEL", "INFO").upper()
    )
    self_test = bool(
        args.self_test or _env_bool(env, "SCIKITPLOT_MCP_SELF_TEST", False)
    )
    self_test_query = (
        args.self_test_query
        or _env_value(env, "SCIKITPLOT_MCP_SELF_TEST_QUERY", "transport")
    ).strip()
    self_test_require_match = bool(
        args.self_test_require_match
        or _env_bool(env, "SCIKITPLOT_MCP_SELF_TEST_REQUIRE_MATCH", False)
    )
    expected_doc_id_raw = args.self_test_expected_doc_id
    if expected_doc_id_raw is None:
        expected_doc_id_raw = env.get("SCIKITPLOT_MCP_SELF_TEST_EXPECTED_DOC_ID")
    self_test_expected_doc_id = (
        expected_doc_id_raw.strip()
        if isinstance(expected_doc_id_raw, str) and expected_doc_id_raw.strip()
        else None
    )

    if args.docs_jsonl and corpus_annoy:
        raise SystemExit("--docs-jsonl and --corpus-annoy are mutually exclusive")
    if not 8 <= hash_dimension <= 65536:  # ruff: ignore[magic-value-comparison]
        raise SystemExit("--hash-dimension must be between 8 and 65536")
    if not 1 <= annoy_n_trees <= 10000:  # ruff: ignore[magic-value-comparison]
        raise SystemExit("--annoy-n-trees must be between 1 and 10000")
    if not annoy_metric or any(char.isspace() for char in annoy_metric):
        raise SystemExit(
            "--annoy-metric must be a non-empty metric name without whitespace"
        )

    if not 1 <= port <= 65535:  # ruff: ignore[magic-value-comparison]
        raise SystemExit("--port must be between 1 and 65535")
    if not 1 <= max_concurrency <= 128:  # ruff: ignore[magic-value-comparison]
        raise SystemExit("--max-concurrency must be between 1 and 128")
    if (
        not 1024  # ruff: ignore[magic-value-comparison]
        <= max_request_body
        <= 4 * 1024 * 1024
    ):
        raise SystemExit("--max-request-body must be between 1 KiB and 4 MiB")
    if not 0.05 <= probe_timeout <= 60.0:  # ruff: ignore[magic-value-comparison]
        raise SystemExit("--probe-timeout must be between 0.05 and 60 seconds")
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise SystemExit("SCIKITPLOT_MCP_LOG_LEVEL is invalid")
    if path == "/":
        raise SystemExit(
            "--path cannot be '/'; reserve the root for deployment infrastructure"
        )
    if health_path == "/":
        raise SystemExit("--health-path cannot be '/'")
    if health_path is not None and (
        health_path == path
        or health_path.startswith(path + "/")
        or path.startswith(health_path + "/")
    ):
        raise SystemExit("--health-path and --path must not overlap")
    if args.probe and transport != "streamable-http":
        raise SystemExit("--probe requires --transport streamable-http")
    if args.probe and health_path is None:
        raise SystemExit("--probe requires an enabled health endpoint")
    if args.probe and self_test:
        raise SystemExit("--probe and --self-test are mutually exclusive")
    if self_test and not self_test_query:
        raise SystemExit("--self-test-query must not be empty")
    if len(self_test_query) > 1024:  # ruff: ignore[magic-value-comparison]
        raise SystemExit("--self-test-query must be at most 1024 characters")
    if (self_test_require_match or self_test_expected_doc_id) and not self_test:
        raise SystemExit(
            "--self-test-require-match and --self-test-expected-doc-id require --self-test"
        )
    if self_test_expected_doc_id is not None and not _DOC_ID_RE.fullmatch(
        self_test_expected_doc_id
    ):
        raise SystemExit(
            "--self-test-expected-doc-id must use 1-200 letters, digits, dot, underscore, colon, or hyphen"
        )
    if transport == "streamable-http" and host not in _LOCAL_HOSTS and not allow_remote:
        raise SystemExit(
            "refusing unauthenticated non-local bind; use localhost, add production auth, "
            "pass --docker for an isolated container network, or explicitly pass "
            "--allow-unauthenticated-remote"
        )

    return RuntimeConfig(
        transport=transport,
        docs_jsonl=args.docs_jsonl,
        corpus_annoy=corpus_annoy,
        corpus_embedding_model=corpus_embedding_model,
        hash_dimension=hash_dimension,
        annoy_metric=annoy_metric,
        annoy_n_trees=annoy_n_trees,
        host=host,
        port=port,
        path=path,
        health_path=health_path,
        max_concurrency=max_concurrency,
        max_request_body=max_request_body,
        stateful_http=bool(args.stateful_http),
        allow_unauthenticated_remote=allow_remote,
        docker=docker,
        log_level=log_level,
        probe=bool(args.probe),
        probe_timeout=probe_timeout,
        self_test=self_test,
        self_test_query=self_test_query,
        self_test_require_match=self_test_require_match,
        self_test_expected_doc_id=self_test_expected_doc_id,
    )


def _connect_host(host: str) -> str:
    if host == "0.0.0.0":  # ruff: ignore[hardcoded-bind-all-interfaces]
        return "127.0.0.1"
    if host == "::":
        return "::1"
    return host


def _health_url(config: RuntimeConfig) -> str:
    if config.health_path is None:
        raise ValueError("health endpoint is disabled")
    host = _connect_host(config.host)
    url_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"http://{url_host}:{config.port}{config.health_path}"


def _probe_health(config: RuntimeConfig) -> int:
    url = _health_url(config)
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(  # noqa: S310 - validated local/admin URL
            url,
            timeout=config.probe_timeout,
        ) as response:
            status = int(getattr(response, "status", 0))
            payload = response.read(64 * 1024)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        message = f"UNHEALTHY {url}: {exc}"
        # print(message, file=sys.stderr)
        logger.exception(message, exc_info=exc)
        return 1

    try:
        body = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        message = f"UNHEALTHY {url}: invalid JSON response ({exc})"
        # print(message, file=sys.stderr)
        logger.exception(message, exc_info=exc)
        return 1

    if (
        status != 200  # ruff: ignore[magic-value-comparison]
        or not isinstance(body, dict)
        or body.get("status") != "ok"
    ):
        message = f"UNHEALTHY {url}: HTTP {status} payload={body!r}"
        print(message, file=sys.stderr)  # ruff: ignore[print]
        logger.warning(message)
        return 1

    message = f"HEALTHY {url}"
    print(message)  # ruff: ignore[print]
    logger.info(message)
    return 0


def _editable_install_hint() -> str | None:
    """Return a concise warning when a Meson editable loader is active.

    This cannot prevent a rebuild triggered while importing ``scikitplot``;
    Python resolves the parent package before this module executes. It still
    makes the risky runtime mode visible once startup reaches the CLI.
    """
    for finder in sys.meta_path:
        module_name = type(finder).__module__
        if "_scikit_plots_editable_loader" in module_name:
            return (
                "Meson editable-install loader detected. Do not use an editable install "
                "in a production Docker runtime because every fresh interpreter may invoke Ninja; "
                "install a built wheel instead."
            )
    return None


def _write_json(payload: object, *, stream: TextIO) -> None:
    """Write one deterministic JSON document to the selected output stream."""
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()


def _build_corpus_annoy_retriever(config: RuntimeConfig):
    """Build the optional Corpus+Annoy retriever without importing it at CLI import time."""
    if config.corpus_annoy is None:
        raise RuntimeError("corpus_annoy path is not configured")
    docs_path = Path(config.corpus_annoy)
    if not docs_path.exists():
        raise SystemExit(f"--corpus-annoy does not exist: {docs_path}")

    from ._corpus_annoy import (  # ruff: ignore[import-outside-top-level]
        CorpusAnnoyRetriever,
    )

    kwargs = {
        "metric": config.annoy_metric,
        "n_trees": config.annoy_n_trees,
        "backend": "annoy",
        "strict": True,
    }
    if config.corpus_embedding_model is not None:
        kwargs["embedding_model"] = config.corpus_embedding_model
    else:
        from scikitplot.corpus import (  # ruff: ignore[import-outside-top-level]
            HashEmbedder,
        )

        kwargs["embedder"] = HashEmbedder(dimension=config.hash_dimension)

    return CorpusAnnoyRetriever.from_corpus_annoy(str(docs_path), **kwargs)


def _load_retriever(config: RuntimeConfig):
    """Load exactly one configured retrieval backend."""
    if config.corpus_annoy:
        return _build_corpus_annoy_retriever(config)
    if config.docs_jsonl:
        docs_path = Path(config.docs_jsonl)
        if not docs_path.is_file():
            raise SystemExit(
                f"--docs-jsonl does not exist or is not a file: {docs_path}"
            )
        return InMemoryBm25Retriever.from_jsonl(docs_path)
    return builtin_demo_retriever()


def main(  # ruff: ignore[too-many-branches, undocumented-public-function]
    argv: list[str] | None = None,
    *,
    stdout: TextIO | None = None,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    config = _resolve_config(args)
    output_stream = sys.stdout if stdout is None else stdout

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    editable_hint = _editable_install_hint()
    if editable_hint:
        logger.warning(editable_hint)

    if args.print_effective_config:
        _write_json(asdict(config), stream=output_stream)
        return 0

    if args.list_capabilities:
        _write_json(server_capabilities(), stream=output_stream)
        return 0

    if config.probe:
        return _probe_health(config)

    retriever = _load_retriever(config)

    if config.self_test:
        # M05: the self-test deliberately uses the Tier-L coordinator, not the
        # pydantic wire adapter, so ``--self-test`` stays runnable on a base
        # install. This is what test_backend_self_test_*_avoids_server_creation
        # asserts by name.
        payload = SearchCoordinator(retriever, max_concurrency=1).search(
            config.self_test_query, 3
        )["structuredContent"]
        if payload["count"] != len(payload["passages"]) or payload["count"] != len(
            payload["citations"]
        ):
            raise SystemExit("self-test failed: inconsistent output contract")
        if config.self_test_require_match and payload["count"] == 0:
            raise SystemExit(
                "self-test failed: query returned no matching documentation"
            )
        if config.self_test_expected_doc_id is not None:
            returned_doc_ids = {
                citation["doc_id"]
                for citation in payload["citations"]
                if citation.get("doc_id")
            }
            if config.self_test_expected_doc_id not in returned_doc_ids:
                raise SystemExit(
                    "self-test failed: expected doc_id "
                    f"{config.self_test_expected_doc_id!r} was not returned"
                )
        _write_json(payload, stream=output_stream)
        return 0

    # M02-01: pre-flight the capability probe before touching the server tier.
    # ``create_server`` imports ``_server``, whose module-scope ``pydantic``
    # import raises before the SDK guard inside it can produce an actionable
    # message -- so a base install got a raw ModuleNotFoundError traceback from
    # the exact command every plugin bundle declares. This is what
    # ``server_runtime_status`` was written for (see _capabilities.py).
    status = server_runtime_status()
    if not status["server_available"]:
        logger.error(
            "cannot start the MCP server: %s (python=%s, sdk_status=%s, "
            "sdk_version=%s)",
            status["reason"],
            status["python"],
            status["sdk_status"],
            status["sdk_version"],
        )
        raise SystemExit(
            "the MCP server layer is unavailable: "
            f"{status['reason']}. Install the server extra with: "
            'pip install "scikit-plots[mcp]"   '
            "(the SDK-free retrieval tier remains usable without it)."
        )

    server = create_server(
        retriever,
        document_reader=retriever.get,
        max_concurrency=config.max_concurrency,
        log_level=config.log_level,
        health_path=(
            config.health_path if config.transport == "streamable-http" else None
        ),
    )

    if config.transport == "stdio":
        logger.info("starting MCP stdio transport")
        server.run()
        return 0

    if config.host not in _LOCAL_HOSTS:
        logger.warning(
            "binding unauthenticated MCP HTTP to %s; restrict Docker port publication or add an authenticated proxy",
            config.host,
        )

    logger.info(
        "starting MCP Streamable HTTP bind=%s:%d endpoint=%s health=%s stateful=%s",
        config.host,
        config.port,
        config.path,
        config.health_path or "disabled",
        config.stateful_http,
    )
    server.run(
        transport="streamable-http",
        host=config.host,
        port=config.port,
        streamable_http_path=config.path,
        json_response=True,
        stateless_http=not config.stateful_http,
        max_request_body_size=config.max_request_body,
    )
    return 0


# curl --fail --silent --show-error http://127.0.0.1:8000/healthz
# curl -i http://127.0.0.1:8000/mcp
if __name__ == "__main__":
    raise SystemExit(main())
