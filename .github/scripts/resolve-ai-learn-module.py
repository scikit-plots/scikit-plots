# .github/scripts/resolve-ai-learn-module.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Resolve the trusted AI Learn package namespace used by publication CI.

The repository has existed in two source layouts:

* ``scikitplot._externals._sphinx_ext._sphinx_ai_learn`` (canonical/current)
* ``_sphinx_ext._sphinx_ai_learn`` (legacy/standalone)

Resolution is deliberately allow-listed and deterministic.  The canonical
fully-qualified namespace wins whenever it is present; the legacy namespace is
used only when the canonical namespace is absent.  A present-but-broken
canonical package fails closed instead of silently switching implementations.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path

PREFERRED_MODULE = "scikitplot._externals._sphinx_ext._sphinx_ai_learn"
FALLBACK_MODULE = "_sphinx_ext._sphinx_ai_learn"
_ALLOWED_MODULES = (PREFERRED_MODULE, FALLBACK_MODULE)


def _find_spec(module_name: str):
    """Return a spec while distinguishing absence from a broken parent import."""
    try:
        return importlib.util.find_spec(module_name)
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").strip()
        if missing and (
            module_name == missing or module_name.startswith(missing + ".")
        ):
            return None
        raise SystemExit(
            f"Failed while probing AI Learn module {module_name!r}: "
            f"missing dependency {missing or '<unknown>'!r}."
        ) from exc
    except Exception as exc:
        raise SystemExit(
            f"Failed while probing AI Learn module {module_name!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def resolve_module() -> str:
    """Resolve and validate the deterministic AI Learn publication module."""
    configured_default = os.environ.get(
        "AI_LEARN_MODULE_DEFAULT",
        PREFERRED_MODULE,
    ).strip()
    if configured_default != PREFERRED_MODULE:
        raise SystemExit(
            "AI_LEARN_MODULE_DEFAULT must remain the canonical namespace: "
            + PREFERRED_MODULE
        )

    preferred_spec = _find_spec(PREFERRED_MODULE)
    if preferred_spec is not None:
        selected = PREFERRED_MODULE
    else:
        fallback_spec = _find_spec(FALLBACK_MODULE)
        if fallback_spec is not None:
            selected = FALLBACK_MODULE
        else:
            raise SystemExit(
                "AI Learn publication module was not found. Expected one of: "
                + ", ".join(_ALLOWED_MODULES)
            )

    try:
        package = importlib.import_module(selected)
    except Exception as exc:  # fail closed with the real import defect
        raise SystemExit(
            f"Resolved AI Learn module {selected!r}, but importing it failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if not callable(getattr(package, "materialize", None)):
        raise SystemExit(
            f"Resolved AI Learn module {selected!r} does not expose callable materialize()."
        )
    if _find_spec(f"{selected}._publication_request_cli") is None:
        raise SystemExit(
            f"Resolved AI Learn module {selected!r} is missing _publication_request_cli."
        )

    return selected


def _append_line(path_value: str | None, line: str) -> None:
    """Append one UTF-8 line to a GitHub Actions environment/summary file."""
    if not path_value:
        return
    path = Path(path_value)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line.rstrip("\n") + "\n")


def main() -> int:
    """Run."""
    selected = resolve_module()
    _append_line(os.environ.get("GITHUB_ENV"), f"AI_LEARN_MODULE={selected}")
    _append_line(
        os.environ.get("GITHUB_STEP_SUMMARY"),
        f"AI Learn Python namespace: `{selected}`",
    )
    print(  # ruff: ignore[print]
        selected,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
