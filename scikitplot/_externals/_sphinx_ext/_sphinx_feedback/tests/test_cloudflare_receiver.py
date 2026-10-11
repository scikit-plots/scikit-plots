"""
The reference Cloudflare Worker receiver agrees with the Python webhook sender.

Notes
-----
The Worker runs under Node (WebCrypto, ``Request`` and ``Response`` are
globals from Node 18) against an in-memory stand-in for D1 that keeps the one
property the Worker relies on: ``INSERT ... ON CONFLICT DO NOTHING`` reports
``changes`` 1 or 0. Bodies and signatures are produced by the Python sender,
so a drift in either side fails here. Skipped where Node is not installed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from _sphinx_ext._sphinx_feedback._contracts import (
    build_feedback_event,
    feedback_event_request_hash,
)
from _sphinx_ext._sphinx_feedback._service._webhook import delivery_body, sign

WORKER = Path(__file__).resolve().parents[1] / "_service" / "receivers" / "cloudflare_worker.js"
NODE = shutil.which("node")
SECRET = "worker-secret"  # noqa: S105 - test value

HARNESS = r"""
import { pathToFileURL } from "node:url";
const [workerPath, casesJson] = process.argv.slice(1);
const worker = await import(pathToFileURL(workerPath).href);
const rows = new Map();
const db = {
  prepare(sql) {
    return {
      bind(...args) {
        return {
          async run() {
            if (rows.has(args[0])) return { meta: { changes: 0 } };
            rows.set(args[0], { request_hash: args[1], event_json: args[4] });
            return { meta: { changes: 1 } };
          },
          async first() { return rows.get(args[0]) || null; },
        };
      },
    };
  },
};
const env = { FEEDBACK_DB: db, FEEDBACK_WEBHOOK_TOKEN: process.env.SECRET };
const out = [];
for (const c of JSON.parse(casesJson)) {
  const response = await worker.default.fetch(
    new Request("https://ingest.example" + c.path, {
      method: c.method,
      headers: { "content-type": "application/json", "x-feedback-signature-256": c.signature },
      body: c.method === "POST" ? c.body : undefined,
    }),
    env,
  );
  out.push({ status: response.status, body: await response.json() });
}
console.log(JSON.stringify(out));
"""


def _event(rating):
    return build_feedback_event(
        {
            "contract": "page.feedback-request.v1",
            "action": "submit",
            "site_id": "docs",
            "page_id": "guide/install",
            "feedback_id": "feedback-" + "7" * 48,
            "rating": rating,
            "mode": "quick",
            "contributor": {"display_name": ""},
        }
    )


def _case(rating, *, secret=SECRET, path="/v1/events", method="POST"):
    event = _event(rating)
    body = delivery_body(event, feedback_event_request_hash(event)).decode("utf-8")
    return {
        "path": path,
        "method": method,
        "body": body,
        "signature": sign(secret, body.encode("utf-8")),
    }


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_worker_stores_once_replays_conflicts_and_checks_signatures():
    cases = [
        _case(1),
        _case(1),
        _case(-1),
        _case(1, secret="wrong"),
        _case(1, path="/elsewhere"),
        _case(1, method="GET"),
    ]
    # Inherit the environment: on Windows a child needs ``SystemRoot`` (and
    # friends) to initialise its system libraries, as the ``subprocess``
    # documentation states for ``env``; a two-variable environment failed the
    # Windows leg only. ``NODE_OPTIONS`` is dropped so a developer's shell
    # cannot inject flags into the harness.
    env = {key: value for key, value in os.environ.items() if key != "NODE_OPTIONS"}
    env["SECRET"] = SECRET
    result = subprocess.run(  # noqa: S603 - fixed argv
        [NODE, "--input-type=module", "-e", HARNESS, str(WORKER), json.dumps(cases)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        env=env,
        check=False,
    )
    # Report Node's own error: ``check=True`` showed only the exit status.
    assert result.returncode == 0, (
        f"node exited {result.returncode}\n--- stderr ---\n{result.stderr}"
    )
    replies = json.loads(result.stdout)
    statuses = [reply["status"] for reply in replies]
    assert statuses == [202, 202, 409, 401, 404, 405]
    assert replies[0]["body"]["status"] == "accepted"
    assert replies[1]["body"]["status"] == "replay"
    assert replies[0]["body"]["feedback_id"] == "feedback-" + "7" * 48
