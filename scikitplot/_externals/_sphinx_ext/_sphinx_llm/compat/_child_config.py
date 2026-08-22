# SPDX-License-Identifier: BSD-3-Clause
"""Child-only Sphinx extension that restores primary-build config semantics."""

from __future__ import annotations

import os
from pathlib import Path

from sphinx.errors import ExtensionError

from .primary_build_context import (
    SNAPSHOT_ENV,
    SNAPSHOT_SHA256_ENV,
    PrimaryBuildContextError,
    apply_snapshot_to_config,
    read_snapshot,
)


def setup(app):
    """Restore the primary build's effective config before later extensions load."""

    # Consume the handoff variables exactly once so child-spawned processes do
    # not inherit stale snapshot authority after this bootstrap has run.
    raw_path = os.environ.pop(SNAPSHOT_ENV, "")
    expected_sha256 = os.environ.pop(SNAPSHOT_SHA256_ENV, "")
    if not raw_path or not expected_sha256:
        raise ExtensionError(
            "The _sphinx_llm child config bootstrap was loaded without an "
            "integrity-checked primary-build snapshot"
        )

    try:
        snapshot = read_snapshot(Path(raw_path), expected_sha256)
    except PrimaryBuildContextError as exc:
        raise ExtensionError(str(exc)) from exc

    # This helper is injected as the first user extension.  Apply immediately
    # so later extension setup code sees primary values.  Apply again at the
    # earliest config-inited priority because Sphinx 5 calls Config.init_values()
    # after extension setup and can otherwise overwrite the early attributes.
    # Newer Sphinx versions also benefit from the explicit final parity pass.
    apply_snapshot_to_config(app.config, snapshot)
    app.connect(
        "config-inited",
        lambda _app, config: apply_snapshot_to_config(config, snapshot),
        priority=1,
    )

    return {
        "version": "1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
