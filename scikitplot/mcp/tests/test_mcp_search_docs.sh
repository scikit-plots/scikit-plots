#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Idempotent end-to-end acceptance test for an already-running MCP server.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PACKAGE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

: "${SCIKITPLOT_MCP_TEST_URL:=http://127.0.0.1:8000/mcp}"
: "${SCIKITPLOT_MCP_HEALTH_URL:=http://127.0.0.1:8000/healthz}"
: "${SCIKITPLOT_MCP_CANARY_TOKEN:=MCP_CANARY_7F3A91C2}"
: "${SCIKITPLOT_MCP_CANARY_DOC_ID:=scikitplot-canary-001}"
: "${SCIKITPLOT_MCP_PARALLEL_REQUESTS:=20}"
: "${SCIKITPLOT_MCP_REPEATED_CONNECTIONS:=10}"

if [[ -z "${SCIKITPLOT_MCP_EXPECTED_VERSION:-}" ]]; then
  SCIKITPLOT_MCP_EXPECTED_VERSION="$({
    PACKAGE_DIR="${PACKAGE_DIR}" python - <<'PY'
import os
import runpy

version_file = os.path.join(os.environ["PACKAGE_DIR"], "_version.py")
print(runpy.run_path(version_file)["__version__"])
PY
  })"
fi

export SCIKITPLOT_MCP_RUN_LIVE=1
export SCIKITPLOT_MCP_TEST_URL
export SCIKITPLOT_MCP_HEALTH_URL
export SCIKITPLOT_MCP_CANARY_TOKEN
export SCIKITPLOT_MCP_CANARY_DOC_ID
export SCIKITPLOT_MCP_PARALLEL_REQUESTS
export SCIKITPLOT_MCP_REPEATED_CONNECTIONS
export SCIKITPLOT_MCP_EXPECTED_VERSION

python - <<'PY'
import json
import os
import urllib.request

url = os.environ["SCIKITPLOT_MCP_HEALTH_URL"]
request = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
with urllib.request.urlopen(request, timeout=5) as response:
    payload = json.loads(response.read().decode("utf-8"))
assert response.status == 200
assert payload.get("status") == "ok"
expected_version = os.environ["SCIKITPLOT_MCP_EXPECTED_VERSION"]
actual_version = payload.get("version")
assert actual_version == expected_version, (
    "stale or wrong MCP server image: "
    f"expected version {expected_version!r}, received {actual_version!r}. "
    "Rebuild and restart the server before running live acceptance tests."
)
print(f"PRECHECK PASS: {url} {payload}")
PY

exec python -m pytest -vv \
  "${SCRIPT_DIR}/integration/test_mcp_http_live.py"
