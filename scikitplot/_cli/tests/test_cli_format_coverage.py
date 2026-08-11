# scikitplot/_cli/tests/test_cli_format_coverage.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Every command that accepts --format must support ALL declared formats.

This is a regression guard for the class of bug where a handler hardcodes a
subset of formats (e.g. show-versions raising KeyError on --format toml). It is
data-driven from the registry, so new commands/formats are covered automatically.
"""
import io
import json
import sys

import pytest

from scikitplot._cli.registry import BUILTIN_COMMANDS

# (command_name, fmt) for every command exposing a "fmt" param, across its choices.
_MATRIX = []
for _spec in BUILTIN_COMMANDS:
    _fmt = next((p for p in _spec.params if p.dest == "fmt"), None)
    if _fmt is not None and _fmt.choices:
        for _choice in _fmt.choices:
            _MATRIX.append((_spec.name, _choice))


@pytest.mark.parametrize("name,fmt", _MATRIX)
def test_command_supports_every_declared_format(name, fmt, monkeypatch):
    from scikitplot._cli._frontends import _argparse
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    code = _argparse.run([name, "--format", fmt])
    assert code == 0, (name, fmt)
    out = buf.getvalue()
    if fmt == "json":
        json.loads(out)
    elif fmt == "toml":
        tomllib = pytest.importorskip("tomllib")
        tomllib.loads(out)
    # yaml/text: exit-0 is sufficient (yaml may be absent -> CapabilityMissing,
    # which surfaces as a non-zero code and would fail above; installed here)
