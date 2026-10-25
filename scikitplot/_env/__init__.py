# scikitplot/_env/__init__.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Environment builder: one request, the same environment on any machine.

Describe the environment once (packages, dependency files, Python version,
target) and build it with whichever tool the host has: uv, venv + pip,
pipenv, micromamba, mamba or conda, for this machine, another Conda platform
or the browser (JupyterLite / Xeus, Pyodide).

The flow is the same from Python, the command line (``scikitplot env``), CI
and an AI agent::

    request  ->  plan (reads, changes nothing)  ->  approvals  ->  execute
                                                                  -> receipt

>>> from scikitplot import _env
>>> p = _env.plan(name="demo", packages=["requests"])  # doctest: +SKIP
>>> print(p.render())  # doctest: +SKIP
>>> _env.create(name="demo", packages=["requests"])  # doctest: +SKIP

Notes
-----
**User notes.**

* ``plan`` shows every command first; ``create`` runs exactly that plan and
  does nothing when the environment already matches (receipt digest).
* Anything that installs a tool, edits a shell start-up file or deletes an
  environment needs an approval: ``approve=["bootstrap"]`` /
  ``--approve bootstrap``, or an answer at a terminal. Configuration files
  cannot approve anything.
* ``network="offline"`` plans only steps that need no download.
* Errors carry a stable ``code`` and a ``hint``; ``--format json`` and
  ``--events FILE`` give machine-readable output for CI and agents.

**Developer notes.** Subpackages, each with one job: ``_core`` (data model,
planner, executor, approvals, events, lock), ``_config`` (documents and
presets), ``_sources`` (dependency files), ``_backends`` (tools as plan
steps), ``_targets`` (where the environment runs), ``_bootstrap`` (tool
installs, shell lines) and ``_xeus`` (the JupyterLite add-on). Importing
this package imports none of them until a name is used.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # the names below, for type checkers and linters (not imported)
    from ._api import batch, create, doctor, plan, remove, run, shell_init, validate
    from ._config._schema import load_request
    from ._core._decisions import Decider
    from ._core._errors import (
        ApprovalRequiredError,
        BackendUnavailableError,
        ConfigError,
        ConversionRefusedError,
        EnvError,
        ExecutionError,
        LockBusyError,
        NetworkForbiddenError,
        PlanningError,
        SourceError,
        TargetUnsupportedError,
    )
    from ._core._events import EventSink
    from ._core._executor import ExecutionResult
    from ._core._host import detect_host
    from ._core._plan import EnvironmentPlan, Step
    from ._core._request import EnvironmentRequest, Policy, SourceRef

__all__ = [
    "ApprovalRequiredError",
    "BackendUnavailableError",
    "ConfigError",
    "ConversionRefusedError",
    "Decider",
    "EnvError",
    "EnvironmentPlan",
    "EnvironmentRequest",
    "EventSink",
    "ExecutionError",
    "ExecutionResult",
    "LockBusyError",
    "NetworkForbiddenError",
    "PlanningError",
    "Policy",
    "SourceError",
    "SourceRef",
    "Step",
    "TargetUnsupportedError",
    "batch",
    "create",
    "detect_host",
    "doctor",
    "load_request",
    "plan",
    "remove",
    "run",
    "shell_init",
    "validate",
]

_LAZY = {
    **dict.fromkeys(
        (
            "batch",
            "create",
            "doctor",
            "plan",
            "remove",
            "run",
            "shell_init",
            "validate",
        ),
        "._api",
    ),
    **dict.fromkeys(
        (
            "ApprovalRequiredError",
            "BackendUnavailableError",
            "ConfigError",
            "ConversionRefusedError",
            "EnvError",
            "ExecutionError",
            "LockBusyError",
            "NetworkForbiddenError",
            "PlanningError",
            "SourceError",
            "TargetUnsupportedError",
        ),
        "._core._errors",
    ),
    "EnvironmentRequest": "._core._request",
    "Policy": "._core._request",
    "SourceRef": "._core._request",
    "EnvironmentPlan": "._core._plan",
    "Step": "._core._plan",
    "ExecutionResult": "._core._executor",
    "Decider": "._core._decisions",
    "EventSink": "._core._events",
    "detect_host": "._core._host",
    "load_request": "._config._schema",
}


def __getattr__(name: str) -> Any:
    """
    Import a public name on first use (keeps ``import scikitplot`` light).

    Parameters
    ----------
    name : str
        Attribute name.

    Returns
    -------
    object
        The attribute, cached in the module namespace.

    Raises
    ------
    AttributeError
        For a name the package does not export.
    """
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(_LAZY[name], __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
