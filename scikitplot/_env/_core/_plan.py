# scikitplot/_env/_core/_plan.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Immutable plans: every step the builder will take, shown before it is taken.

Notes
-----
**User notes.** A plan lists each command with its arguments, whether it uses
the network, what it writes and whether it needs an approval. ``scikitplot env
plan`` prints it; ``create`` executes exactly that plan.

**Developer notes.** A step is either a ``command`` (an argv run with prompts
off) or an ``action`` (a named function of the builder with plain-data
parameters, such as a bootstrap download). Both are data, so a plan is
serialisable, comparable and has a digest; the executor looks actions up in
one table.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import shlex
from typing import Any

__all__ = ["EnvironmentPlan", "Step", "digest_of", "redact", "redact_data"]

#: Schema identifier of a plan document.
PLAN_SCHEMA = "scikitplot.env.plan/v1"

_SECRET_URL = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)[^/@\s]+@")
_CONDA_TOKEN = re.compile(r"(?P<pre>://[^/\s]+(?:/[^\s]*)?/t/)[^/\s]+/")
_SECRET_QUERY = re.compile(
    r"(?i)([?&](?:token|access_token|api_key|key|password)=)[^&\s]+"
)


def digest_of(data: Any) -> str:
    """
    Return the SHA-256 of ``data`` as canonical JSON.

    Parameters
    ----------
    data : object
        JSON-serialisable data (other objects are converted with ``str``).

    Returns
    -------
    str
        Hex digest of the sorted, compact JSON text.

    Examples
    --------
    >>> from scikitplot._env._core._plan import digest_of
    >>> digest_of({"b": 1, "a": 2}) == digest_of({"a": 2, "b": 1})
    True
    """
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def redact(text: str) -> str:
    """
    Remove credentials from URLs in ``text``.

    Parameters
    ----------
    text : str
        Text that may hold URLs.

    Returns
    -------
    str
        URL user info (``user:password@`` or a bare ``token@``) becomes
        ``***@``; an anaconda.org ``/t/<token>/`` path part becomes
        ``/t/***/``; ``token=...``-style query values become ``***``.

    Examples
    --------
    >>> from scikitplot._env._core._plan import redact
    >>> redact("pip install -i https://me:s3cret@host/simple x")
    'pip install -i https://***@host/simple x'
    """
    text = _SECRET_URL.sub(lambda m: m.group("scheme") + "***@", text)
    text = _CONDA_TOKEN.sub(lambda m: m.group("pre") + "***/", text)
    return _SECRET_QUERY.sub(lambda m: m.group(1) + "***", text)


def redact_data(value: Any) -> Any:
    """
    Return ``value`` with :py:func:`redact` applied to every string in it.

    Parameters
    ----------
    value : object
        Plain data (mappings, sequences, scalars).

    Returns
    -------
    object
        The same structure (tuples become lists, unknown objects strings).

    Examples
    --------
    >>> from scikitplot._env._core._plan import redact_data
    >>> redact_data({"url": ["git+https://ghp_x@github.com/o/r"]})
    {'url': ['git+https://***@github.com/o/r']}
    """
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {str(k): redact_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_data(v) for v in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact(str(value))


@dataclasses.dataclass(frozen=True)
class Step:  # numpydoc ignore=PR02
    """
    One step of a plan.

    Parameters
    ----------
    id : str
        Unique id within the plan (``"create"``, ``"install-pypi"``).
    title : str
        One line for people.
    kind : {"command", "action"}, default="command"
        A command (``argv``) or a builder action (``action`` + ``params``).
    argv : tuple of str, default=()
        Command and arguments.
    action : str or None, default=None
        Action name for ``kind="action"``.
    params : tuple of (str, object), default=()
        Plain-data parameters of the action.
    env : tuple of (str, str), default=()
        Extra environment variables of the command.
    cwd : str or None, default=None
        Working directory of the command.
    network : bool, default=False
        Whether the step may use the network.
    writes : tuple of str, default=()
        Paths the step creates or changes.
    approval : str or None, default=None
        Approval kind needed before the step runs.
    """

    id: str
    title: str
    kind: str = "command"
    argv: tuple[str, ...] = ()
    action: str | None = None
    params: tuple[tuple[str, Any], ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    cwd: str | None = None
    network: bool = False
    writes: tuple[str, ...] = ()
    approval: str | None = None

    def __post_init__(self) -> None:  # noqa: D105
        if self.kind not in ("command", "action"):
            raise ValueError(f"step kind {self.kind!r} is not 'command' or 'action'")
        if self.kind == "command" and not self.argv:
            raise ValueError(f"command step {self.id!r} has no argv")
        if self.kind == "action" and not self.action:
            raise ValueError(f"action step {self.id!r} names no action")
        object.__setattr__(self, "argv", tuple(str(a) for a in self.argv))
        object.__setattr__(self, "writes", tuple(str(w) for w in self.writes))
        object.__setattr__(self, "params", tuple(self.params))
        object.__setattr__(self, "env", tuple((str(k), str(v)) for k, v in self.env))

    @property
    def params_dict(self) -> dict[str, Any]:  # numpydoc ignore=RT01
        """dict: Action parameters as a mapping."""
        return dict(self.params)

    def command_line(self) -> str:
        """
        Return the command as one redacted, shell-quoted line.

        Returns
        -------
        str
            ``argv`` joined with :py:func:`shlex.join`, or ``action(params)``.
        """
        if self.kind == "action":
            args = ", ".join(f"{k}={v!r}" for k, v in self.params)
            return redact(f"{self.action}({args})")
        return redact(shlex.join(self.argv))

    def to_dict(self) -> dict[str, Any]:
        """
        Return the step as a mapping (argv redacted).

        Returns
        -------
        dict
            Plain data (JSON-serialisable).
        """
        return {
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "argv": [redact(a) for a in self.argv],
            "action": self.action,
            "params": dict(self.params),
            "env": dict(self.env),
            "cwd": self.cwd,
            "network": self.network,
            "writes": list(self.writes),
            "approval": self.approval,
        }


@dataclasses.dataclass(frozen=True)
class EnvironmentPlan:  # numpydoc ignore=PR02
    """
    Everything the builder will do for one request.

    Parameters
    ----------
    request : dict
        The request, as :py:meth:`~.EnvironmentRequest.to_dict` with provenance.
    request_digest : str
        Digest of the request.
    operation : {"ensure", "recreate", "remove"}
        What the plan does.
    target : str
        Target kind.
    backend_requested, backend : str
        Backend asked for and backend chosen.
    backend_reason : str
        Why that backend.
    prefix : str
        Directory of the environment.
    python_executable : str or None
        Python of the environment once built (None for plan-only targets).
    exists : bool
        Whether a usable environment was already at ``prefix`` when planning.
    up_to_date : bool
        Whether its receipt records this request digest (nothing to do).
    steps : tuple of Step
        Steps in order.
    dependencies : tuple of dict
        Normalised dependencies (``name``, ``spec``, ``ecosystem``, ``origin``).
    notes : tuple of str
        Conversions and decisions people should read.
    host : dict
        The host description used for planning.
    executable : bool, default=True
        False for plan-only targets (``pyodide``); ``artifacts`` then holds
        what to use instead.
    artifacts : tuple of (str, str), default=()
        Generated texts of a plan-only target (``("micropip.py", ...)``).
    inputs : tuple of (str, str), default=()
        ``(path, sha256)`` of every dependency file read while planning.
    python : str or None, default=None
        Python version asked for, by the request or a dependency file.
    state_digest : str, default=""
        Digest of what the environment should contain: request digest,
        target, backend, prefix and ``inputs``. The receipt stores it, and a
        later plan with the same value is up to date. Unlike
        :py:attr:`digest` it does not depend on the steps, which differ
        between creating and updating.
    """

    request: dict
    request_digest: str
    operation: str
    target: str
    backend_requested: str
    backend: str
    backend_reason: str
    prefix: str
    python_executable: str | None
    exists: bool
    up_to_date: bool
    steps: tuple[Step, ...]
    dependencies: tuple[dict, ...]
    notes: tuple[str, ...]
    host: dict
    executable: bool = True
    artifacts: tuple[tuple[str, str], ...] = ()
    inputs: tuple[tuple[str, str], ...] = ()
    python: str | None = None
    state_digest: str = ""

    @property
    def uses_network(self) -> bool:  # numpydoc ignore=RT01
        """bool: Some step may use the network."""
        return any(s.network for s in self.steps)

    @property
    def approvals(self) -> tuple[str, ...]:  # numpydoc ignore=RT01
        """Tuple of str: Approval kinds the plan needs, in step order, once each."""
        seen: list[str] = []
        for step in self.steps:
            if step.approval and step.approval not in seen:
                seen.append(step.approval)
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        """
        Return the plan as a mapping of the shared data model.

        Returns
        -------
        dict
            The plan with ``schema`` and ``digest``.
        """
        data = {
            "schema": PLAN_SCHEMA,
            "operation": self.operation,
            "target": self.target,
            "backend_requested": self.backend_requested,
            "backend": self.backend,
            "backend_reason": self.backend_reason,
            "prefix": self.prefix,
            "python_executable": self.python_executable,
            "exists": self.exists,
            "up_to_date": self.up_to_date,
            "executable": self.executable,
            "uses_network": self.uses_network,
            "approvals": list(self.approvals),
            "steps": [s.to_dict() for s in self.steps],
            "dependencies": [dict(d) for d in self.dependencies],
            "notes": list(self.notes),
            "artifacts": dict(self.artifacts),
            "request_digest": self.request_digest,
            "state_digest": self.state_digest,
            "python": self.python,
            "inputs": [{"path": p, "sha256": h} for p, h in self.inputs],
            "request": self.request,
            "host": self.host,
        }
        data = redact_data(data)  # specs, URLs and notes may hold a token
        data["digest"] = self.digest
        return data

    @property
    def digest(self) -> str:  # numpydoc ignore=RT01
        """str: SHA-256 of the steps, prefix, backend, inputs and request digest."""
        core = {
            "request_digest": self.request_digest,
            "state_digest": self.state_digest,
            "backend": self.backend,
            "prefix": self.prefix,
            "operation": self.operation,
            "steps": [s.to_dict() for s in self.steps],
        }
        return digest_of(core)

    def render(self) -> str:
        """
        Return the plan as text for people.

        Returns
        -------
        str
            Several lines: summary, notes, then one block per step.
        """
        lines = [
            f"plan {self.digest[:12]}  {self.operation} {self.prefix}",
            f"  target   {self.target}",
            f"  backend  {self.backend}"
            + (
                f" (asked: {self.backend_requested})"
                if self.backend_requested != self.backend
                else ""
            )
            + f" - {self.backend_reason}",
        ]
        if self.up_to_date:
            lines.append("  status   up to date: the receipt matches this request")
        elif self.exists:
            lines.append("  status   exists: packages will be brought up to date")
        lines.extend(f"  note     {redact(note)}" for note in self.notes)
        if not self.executable:
            lines.append(
                "  plan only: this target is not built by running commands here"
            )
        for i, step in enumerate(self.steps, 1):
            flags = [
                f
                for f, on in (
                    ("network", step.network),
                    (f"needs approval: {step.approval}", step.approval),
                )
                if on
            ]
            lines.append(
                f"  {i}. {step.title}" + (f"  [{', '.join(flags)}]" if flags else "")
            )
            lines.append(f"     $ {step.command_line()}")
        for name, _ in self.artifacts:
            lines.append(f"  artifact {name}")
        return "\n".join(lines)
