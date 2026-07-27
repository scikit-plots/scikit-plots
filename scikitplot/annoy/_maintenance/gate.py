"""
Scoped-gate runner (documented harness, rule L-GATE-SCOPE — not a source change).

Stubs the unbuilt `scikitplot.api` namespace so the package's lazy __getattr__
short-circuits in a SCOPED build (only annoy/cexternals/memmap/random compiled),
then binds the real `get_config` the annoy extension imports at module load.
"""

import sys
import types

_api = types.ModuleType("scikitplot.api")
_api.__path__ = []
sys.modules["scikitplot.api"] = _api
import scikitplot

try:
    from scikitplot.config import config_context, get_config, set_config

    for _n, _v in (
        ("get_config", get_config),
        ("set_config", set_config),
        ("config_context", config_context),
    ):
        setattr(scikitplot, _n, _v)
except (
    Exception  # ruff: ignore[blind-except]
) as e:  # keep gate honest: surface, don't mask
    sys.stderr.write(f"[gate] config bind failed: {e!r}\n")
import pytest

sys.exit(
    pytest.main(
        [
            "-q",
            "--noconftest",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            *sys.argv[1:],
        ]
    )
)
