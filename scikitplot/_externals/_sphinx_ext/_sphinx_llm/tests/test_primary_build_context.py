# SPDX-License-Identifier: BSD-3-Clause
"""Dependency-light tests for primary-build context transfer helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat

import pytest

from scikitplot._externals._sphinx_ext._sphinx_llm.compat.primary_build_context import (
    CHILD_EXTENSION,
    PrimaryBuildContextError,
    apply_snapshot_to_config,
    capture_primary_config,
    child_extensions_value,
    direct_override_cli_args,
    encode_snapshot,
    read_snapshot,
    write_snapshot,
)


@dataclass
class _Config:
    overrides: dict
    extensions: list[str]

    def __post_init__(self):
        self.values = {"language": object(), "feature": object()}
        self.language = "fr"
        self.feature = ["a", "b"]


def test_direct_override_cli_args_are_deterministic():
    config = _Config(
        overrides={
            "language": "fr",
            "nitpicky": True,
            "source_encoding": "utf-8",
        },
        extensions=["demo.ext"],
    )
    assert direct_override_cli_args(config) == [
        "-D",
        "language=fr",
        "-D",
        "source_encoding=utf-8",
    ]



def test_effective_value_is_captured_after_override_mapping_was_consumed():
    config = _Config(overrides={}, extensions=[])
    config.values["feature_options"] = object()
    config.feature_options = {"mode": "override"}
    snapshot = capture_primary_config(config)
    assert snapshot["values"]["feature_options"] == {"mode": "override"}


def test_secret_shaped_override_never_enters_cli_arguments():
    config = _Config(
        overrides={"language": "fr", "api_token": "do-not-expose"},
        extensions=[],
    )
    args = direct_override_cli_args(config)
    assert args == ["-D", "language=fr"]
    assert "do-not-expose" not in " ".join(args)

def test_child_extension_is_first_and_deduplicated():
    value = child_extensions_value(["a", CHILD_EXTENSION, "b"])
    assert value.split(",") == [CHILD_EXTENSION, "a", "b"]



def test_child_extension_with_comma_fails_closed():
    with pytest.raises(PrimaryBuildContextError, match="containing commas"):
        child_extensions_value(["demo,invalid"])


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits only")
def test_snapshot_file_is_private_on_posix():
    config = _Config(overrides={"language": "fr"}, extensions=[])
    path, digest = write_snapshot(capture_primary_config(config))
    try:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        read_snapshot(path, digest)
    finally:
        path.unlink(missing_ok=True)

def test_snapshot_roundtrip_integrity_checks_and_deletes_file():
    config = _Config(overrides={"language": "fr"}, extensions=["demo.ext"])
    snapshot = capture_primary_config(config)
    path, digest = write_snapshot(snapshot)
    restored = read_snapshot(path, digest)
    assert restored["values"]["language"] == "fr"
    assert restored["extensions"] == ["demo.ext"]
    assert not path.exists()


def test_snapshot_digest_mismatch_fails_closed(tmp_path: Path):
    config = _Config(overrides={}, extensions=[])
    payload, _ = encode_snapshot(capture_primary_config(config))
    path = tmp_path / "snapshot.pkl"
    path.write_bytes(payload)
    with pytest.raises(PrimaryBuildContextError, match="digest mismatch"):
        read_snapshot(path, "0" * 64)
    assert not path.exists()


def test_unpickleable_direct_override_fails_closed():
    config = _Config(overrides={"language": lambda: "fr"}, extensions=[])
    with pytest.raises(PrimaryBuildContextError, match="Unpickleable direct"):
        capture_primary_config(config)


def test_apply_snapshot_restores_values_and_extensions():
    config = _Config(overrides={}, extensions=["child-only"])
    snapshot = {
        "schema_version": 1,
        "values": {"language": "tr", "feature": ["x"]},
        "extensions": ["primary.ext"],
        "skipped": {},
    }
    apply_snapshot_to_config(config, snapshot)
    assert config.language == "tr"
    assert config.feature == ["x"]
    assert config.extensions == ["primary.ext"]
