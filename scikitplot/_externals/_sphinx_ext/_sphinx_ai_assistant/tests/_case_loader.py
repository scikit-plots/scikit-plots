"""
Load non-collected case fragments into one canonical test-module owner.

A production Python module has one collected pytest owner (``test_<module>.py``).
Large contracts may keep feature-sized case fragments under a hidden ``_cases``
package; those files are not collected directly.  This loader imports them and
exports only pytest-visible tests/fixtures into the canonical owner, preserving
one collection path without forcing unrelated helper namespaces into one giant
module.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any, MutableMapping, Sequence


def _is_test_object(name: str, value: Any, module_name: str) -> bool:
    if getattr(value, "__module__", None) != module_name:
        return False
    return (name.startswith("test_") and callable(value)) or (
        name.startswith("Test") and isinstance(value, type)
    )


def _is_fixture(value: Any) -> bool:
    return (
        getattr(value, "_pytestfixturefunction", None) is not None
        or getattr(value, "_fixture_function_marker", None) is not None
    )


def export_case_tests(
    namespace: MutableMapping[str, Any],
    *,
    package: str,
    case_package: str,
    cases: Sequence[str],
) -> None:
    """
    Export case tests/fixtures into one canonical pytest module.

    Case modules may define ``pytestmark``; those marks are applied to each
    exported test object so fixture/skip semantics remain case-local after the
    physical move. Duplicate pytest-visible names fail closed instead of
    silently overriding coverage.
    """
    import pytest

    for case in cases:
        module = import_module(f".{case_package}.{case}", package)
        marks = getattr(module, "pytestmark", ())
        if not isinstance(marks, (list, tuple)):
            marks = (marks,)

        for name, value in vars(module).items():
            if not _is_fixture(value):
                continue
            prior = namespace.get(name)
            if prior is not None and prior is not value:
                raise RuntimeError(
                    f"duplicate fixture name {name!r} while loading {case_package}.{case}"
                )
            namespace[name] = value

        for name, value in vars(module).items():
            if not _is_test_object(name, value, module.__name__):
                continue
            if name in namespace:
                raise RuntimeError(
                    f"duplicate test name {name!r} while loading {case_package}.{case}"
                )
            for mark in marks:
                if mark:
                    value = mark(value)
            namespace[name] = value
