# SPDX-License-Identifier: BSD-3-Clause
"""
Primary Sphinx build context capture for the downstream Markdown sub-build.

The NVIDIA baseline deliberately runs Markdown as a separate Sphinx build.  This
module carries the *effective* primary configuration across that process boundary
without exposing configuration values on the command line.

Only a short-lived local file is used for the snapshot.  Its SHA-256 digest is
passed separately via the child environment and verified before deserialisation.
The child removes the file immediately after reading it.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import pickle
import stat
import tempfile
from pathlib import Path
from typing import Any

SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_ENV = "SCIKITPLOT_SPHINX_LLM_PRIMARY_CONFIG"
SNAPSHOT_SHA256_ENV = "SCIKITPLOT_SPHINX_LLM_PRIMARY_CONFIG_SHA256"
CHILD_EXTENSION = "scikitplot._externals._sphinx_ext._sphinx_llm.compat._child_config"
# These Sphinx core values are consumed before user extensions load, so they
# may need early ``-D`` propagation.  Everything else stays off the process
# command line and is restored by the integrity-checked child snapshot.
EARLY_CONFIG_OVERRIDE_KEYS = frozenset(
    {
        "needs_sphinx",
        "suppress_warnings",
        "language",
        "locale_dirs",
        "source_encoding",
        "gettext_allow_fuzzy_translations",
    }
)


class PrimaryBuildContextError(RuntimeError):
    """Raised when primary-build semantic context cannot be transferred safely."""


def _pickleable(value: object) -> bool:
    try:
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:  # ruff: ignore[blind-except]
        return False
    return True


def capture_primary_config(config: Any) -> dict[str, Any]:
    """
    Capture pickleable effective Sphinx configuration values.

    Sphinx extensions have all registered their config values by the time the
    primary ``config-inited`` event fires.  Capturing at that point gives us the
    effective values *before* later config-inited transforms run.  Values that
    cannot be pickled are intentionally left to the original ``conf.py`` in the
    child process; if such a value is also a direct programmatic override we fail
    closed because the child could otherwise silently diverge.
    """

    values: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    config_names = tuple(getattr(config, "values", {}).keys())

    for name in config_names:
        try:
            value = getattr(config, name)
        except (
            Exception  # ruff: ignore[blind-except]
        ) as exc:  # pragma: no cover - defensive  # ruff: ignore[blind-except]
            skipped[name] = f"read-error:{type(exc).__name__}"
            continue
        if _pickleable(value):
            values[name] = value
        else:
            skipped[name] = type(value).__qualname__

    extensions = list(getattr(config, "extensions", ()) or ())
    if not all(isinstance(item, str) for item in extensions):
        raise PrimaryBuildContextError(
            "Sphinx config.extensions contains a non-string entry and cannot be "
            "replayed safely in the Markdown subprocess"
        )

    direct_overrides = dict(getattr(config, "overrides", {}) or {})
    untransferable_direct = [
        name
        for name, value in direct_overrides.items()
        if name in skipped or not _pickleable(value)
    ]
    if untransferable_direct:
        joined = ", ".join(sorted(untransferable_direct))
        raise PrimaryBuildContextError(
            "Unpickleable direct Sphinx config override(s) cannot cross the "
            f"Markdown subprocess boundary safely: {joined}"
        )

    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "values": values,
        "extensions": extensions,
        "skipped": skipped,
    }


def serialize_direct_override(name: str, value: Any) -> str | None:
    """
    Return a Sphinx ``-D`` value when the conversion is lossless enough.

    These direct overrides are forwarded in addition to the effective-config
    snapshot because a few core Sphinx settings (for example ``language`` and
    ``needs_sphinx``) are consumed before user extensions are loaded.

    ``None`` means "let the early child bootstrap restore this value instead".
    """

    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, str) for item in value
    ):
        return ",".join(value)
    return None


def direct_override_cli_args(config: Any) -> list[str]:
    """Build deterministic ``-D`` arguments for directly representable overrides."""

    args: list[str] = []
    overrides = dict(getattr(config, "overrides", {}) or {})
    for name in sorted(overrides):
        if name not in EARLY_CONFIG_OVERRIDE_KEYS:
            continue
        value = serialize_direct_override(name, overrides[name])
        if value is not None:
            args.extend(["-D", f"{name}={value}"])
    return args


def child_extensions_value(extensions: list[str] | tuple[str, ...]) -> str:
    """Return the child extension list with the bootstrap extension first."""

    ordered = [CHILD_EXTENSION]
    ordered.extend(item for item in extensions if item != CHILD_EXTENSION)
    if any("," in item for item in ordered):
        raise PrimaryBuildContextError(
            "Sphinx extension names containing commas cannot be transferred with "
            "the supported -D extensions=... boundary"
        )
    return ",".join(ordered)


def encode_snapshot(snapshot: dict[str, Any]) -> tuple[bytes, str]:
    """Serialize a validated snapshot and return ``(payload, sha256)``."""

    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise PrimaryBuildContextError("Unsupported primary-config snapshot schema")
    payload = pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL)
    return payload, hashlib.sha256(payload).hexdigest()


def write_snapshot(snapshot: dict[str, Any]) -> tuple[Path, str]:
    """Write a private temporary snapshot for the Markdown child process."""

    payload, digest = encode_snapshot(snapshot)
    fd, raw_path = tempfile.mkstemp(prefix="scikitplot_sphinx_llm_", suffix=".pkl")
    path = Path(raw_path)
    try:
        if os.name == "posix":
            os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
    except Exception:
        # try:
        #     os.close(fd)
        # except OSError:
        #     pass
        with contextlib.suppress(OSError):
            os.close(fd)
        path.unlink(missing_ok=True)
        raise
    return path, digest


def read_snapshot(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Read, integrity-check, validate, and remove a child configuration snapshot."""

    try:
        payload = path.read_bytes()
    finally:
        # The snapshot may contain sensitive build configuration.  Remove it as
        # soon as the child has read the bytes, including on validation failure.
        path.unlink(missing_ok=True)

    actual = hashlib.sha256(payload).hexdigest()
    if not expected_sha256 or actual != expected_sha256:
        raise PrimaryBuildContextError(
            "Primary-config snapshot digest mismatch; refusing to deserialize it"
        )

    try:
        snapshot = pickle.loads(payload)  # noqa: S301 - integrity-checked local handoff
    except Exception as exc:
        raise PrimaryBuildContextError(
            "Primary-config snapshot could not be deserialized"
        ) from exc

    if not isinstance(snapshot, dict):
        raise PrimaryBuildContextError("Primary-config snapshot must be a mapping")
    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise PrimaryBuildContextError("Unsupported primary-config snapshot schema")
    if not isinstance(snapshot.get("values"), dict):
        raise PrimaryBuildContextError("Primary-config snapshot values are invalid")
    if not isinstance(snapshot.get("extensions"), list) or not all(
        isinstance(item, str) for item in snapshot["extensions"]
    ):
        raise PrimaryBuildContextError("Primary-config snapshot extensions are invalid")
    return snapshot


def apply_snapshot_to_config(config: Any, snapshot: dict[str, Any]) -> None:
    """Apply captured effective values to a child Sphinx ``Config`` object."""

    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise PrimaryBuildContextError("Unsupported primary-config snapshot schema")
    values = snapshot.get("values")
    extensions = snapshot.get("extensions")
    if not isinstance(values, dict):
        raise PrimaryBuildContextError("Primary-config snapshot values are invalid")
    if not isinstance(extensions, list) or not all(
        isinstance(item, str) for item in extensions
    ):
        raise PrimaryBuildContextError("Primary-config snapshot extensions are invalid")
    for name, value in values.items():
        setattr(config, name, value)
    config.extensions = list(extensions)


def snapshot_environment(path: Path, digest: str) -> dict[str, str]:
    """Return the two environment variables required by the child bootstrap."""

    return {
        SNAPSHOT_ENV: str(path),
        SNAPSHOT_SHA256_ENV: digest,
    }
