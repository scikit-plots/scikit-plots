# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026, The scikit-plots developers.
# Modified by the scikit-plots project for downstream configuration parity.
# SPDX-License-Identifier: Apache-2.0
"""
Downstream Markdown generator preserving primary Sphinx configuration.

``build_markdown_files`` is intentionally kept structurally close to NVIDIA's
pinned implementation so upstream changes remain reviewable.  The downstream
changes are limited to:

* capture the primary effective config before ``config-inited`` transforms;
* inject a child bootstrap extension first;
* forward directly representable overrides with ``-D`` for early Sphinx use;
* transfer the full pickleable effective config through an integrity-checked,
  short-lived local snapshot rather than process-list-visible arguments.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from sphinx.errors import ExtensionError

from ..sphinx_llm import txt as upstream_txt
from .primary_build_context import (
    PrimaryBuildContextError,
    capture_primary_config,
    child_extensions_value,
    direct_override_cli_args,
    snapshot_environment,
    write_snapshot,
)


class ConfigParityMarkdownGenerator(upstream_txt.MarkdownGenerator):
    """NVIDIA generator plus a fail-closed primary-config transfer boundary."""

    def __init__(self, app):
        super().__init__(app)
        self._primary_config_snapshot = None
        self._config_snapshot_path: Path | None = None

    def setup(self):
        # Capture before Sphinx's built-in config-inited transforms (normally
        # priority ~790/800) so both primary and child apply those transforms once.
        self.app.connect("config-inited", self._capture_primary_config, priority=1)
        super().setup()

    def _capture_primary_config(self, app, config):
        try:
            self._primary_config_snapshot = capture_primary_config(config)
        except PrimaryBuildContextError as exc:
            raise ExtensionError(str(exc)) from exc

    def _cleanup_config_snapshot(self):
        path = self._config_snapshot_path
        self._config_snapshot_path = None
        if path is not None:
            path.unlink(missing_ok=True)

    def build_markdown_files(self, app=None, exception=None):
        """Start the Markdown sub-build with primary semantic config preserved."""

        if exception is not None:
            upstream_txt.logger.info(
                "Skipping markdown build because the primary build failed"
            )
            return

        self.md_build_dir.mkdir(exist_ok=True)
        self.md_build_logfile = (  # ruff: ignore[open-file-with-context-handler]
            tempfile.NamedTemporaryFile(  # ruff: ignore[open-file-with-context-handler]
                mode="w",
                delete=False,
                prefix="sphinx_llm_output_",
                suffix=".log",
            )
        )

        try:
            if self._primary_config_snapshot is None:
                # Defensive path for direct generator use outside normal Sphinx
                # event order; normal extension execution captures at config-inited.
                self._primary_config_snapshot = capture_primary_config(self.app.config)

            snapshot_path, snapshot_sha256 = write_snapshot(
                self._primary_config_snapshot
            )
            self._config_snapshot_path = snapshot_path

            sphinx_build_cmd = [
                sys.executable,
                "-m",
                "sphinx",
                "-b",
                upstream_txt.SphinxLlmMarkdownBuilder.name,
                "-t",
                "sphinx_llm_markdown",
                "-c",
                str(self.app.confdir),
                str(self.app.srcdir),
                str(self.md_build_dir),
            ]

            for tag in self.app.tags:
                sphinx_build_cmd += ["-t", tag]

            # Preserve directly representable overrides early enough for Sphinx
            # core initialization (language, needs_sphinx, etc.).  The child
            # bootstrap restores the complete pickleable effective config before
            # the remaining user extensions load.
            sphinx_build_cmd += direct_override_cli_args(self.app.config)
            extensions = list(self._primary_config_snapshot["extensions"])
            sphinx_build_cmd += [
                "-D",
                f"extensions={child_extensions_value(extensions)}",
            ]

            if not self.parallel:
                sphinx_build_cmd.extend(["-d", str(self.app.doctreedir)])

            env = os.environ.copy()
            env.update(snapshot_environment(snapshot_path, snapshot_sha256))

            upstream_txt.logger.info(
                "Spawning additional sphinx subprocess to build markdown files "
                "for llms.txt with primary configuration parity"
            )
            upstream_txt.logger.info(
                "Subprocess output available at: %s", self.md_build_logfile.name
            )

            try:
                with self.md_build_logfile:
                    self.md_build_process = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
                        sphinx_build_cmd,
                        stdout=self.md_build_logfile,
                        stderr=self.md_build_logfile,
                        env=env,
                    )
            except Exception as exc:  # ruff: ignore[blind-except]
                self._cleanup_config_snapshot()
                upstream_txt.logger.error(
                    "Failed to run sphinx-build subprocess: %s", exc
                )
        except PrimaryBuildContextError as exc:
            self._cleanup_config_snapshot()
            raise ExtensionError(
                f"Cannot preserve primary Sphinx configuration for Markdown: {exc}"
            ) from exc
        except Exception as exc:  # ruff: ignore[blind-except]
            self._cleanup_config_snapshot()
            upstream_txt.logger.error("Failed to generate markdown files: %s", exc)

    def combine_builds(self, app, exception):
        try:
            return super().combine_builds(app, exception)
        finally:
            self._cleanup_config_snapshot()
