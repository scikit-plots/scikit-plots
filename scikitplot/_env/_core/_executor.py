# scikitplot/_env/_core/_executor.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The executor: runs a plan exactly, under a lock, with every approval first.

Notes
-----
**User notes.** Before the first step runs, every approval the plan needs
is decided (asked once, or taken from ``approve=``); a refusal stops the run
with nothing changed. Commands run with no terminal input and prompts off,
their output is logged (``DEBUG``) and the last lines go into the error when
one fails. After success the receipt is written into the environment.

``PYTHONPATH`` and ``PYTHONHOME`` are removed from the commands'
environment: either one makes a new environment import the wrong packages.

**Developer notes.** Actions are the builder's own functions, looked up by
name in :py:data:`ACTIONS`; tests replace the runner and actions with
fakes. A failed step raises :py:class:`~._errors.ExecutionError` whose
``result`` attribute holds the partial :py:class:`ExecutionResult`.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import logging
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from ._decisions import Decider
from ._errors import ExecutionError, PlanningError
from ._events import EventSink
from ._lock import PrefixLock
from ._plan import EnvironmentPlan, Step, redact, redact_data
from ._receipt import is_environment, read_receipt, write_receipt
from ._types import Operation

__all__ = ["ACTIONS", "ExecutionResult", "StepResult", "execute"]

logger = logging.getLogger(__name__)

_DROPPED = ("PYTHONPATH", "PYTHONHOME")
_TAIL = 30


@dataclasses.dataclass(frozen=True)
class StepResult:  # numpydoc ignore=PR02
    """
    Outcome of one step.

    Parameters
    ----------
    id : str
        Step id.
    ok : bool
        Whether it succeeded.
    seconds : float
        Duration.
    returncode : int or None, default=None
        Exit status of a command.
    """

    id: str
    ok: bool
    seconds: float
    returncode: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Return the result as a mapping.

        Returns
        -------
        dict
            Plain data (JSON-serialisable).
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ExecutionResult:  # numpydoc ignore=PR02
    """
    Outcome of a plan.

    Parameters
    ----------
    plan : EnvironmentPlan
        The executed plan.
    ok : bool
        Whether every step succeeded.
    steps : tuple of StepResult
        Results in order (up to the failed step).
    receipt : str or None, default=None
        Path of the receipt written.
    approvals : tuple of dict, default=()
        How each approval was decided.
    """

    plan: EnvironmentPlan
    ok: bool
    steps: tuple[StepResult, ...]
    receipt: str | None = None
    approvals: tuple[dict, ...] = ()

    @property
    def changed(self) -> bool:  # numpydoc ignore=RT01
        """bool: Whether any step ran."""
        return bool(self.steps)

    def to_dict(self) -> dict[str, Any]:
        """
        Return the result as a mapping.

        Returns
        -------
        dict
            Plain data (JSON-serialisable).
        """
        return {
            "ok": self.ok,
            "changed": self.changed,
            "prefix": self.plan.prefix,
            "backend": self.plan.backend,
            "plan_digest": self.plan.digest,
            "receipt": self.receipt,
            "approvals": list(self.approvals),
            "steps": [s.to_dict() for s in self.steps],
        }


# --- actions ---------------------------------------------------------------


def _remove_tree(path: str) -> None:
    target = Path(path)
    if not target.exists():
        return
    if (
        target == Path(target.anchor)
        or target == Path.home()
        or not is_environment(target)
    ):
        raise ExecutionError(
            f"refusing to delete {target}: not an environment", step="remove"
        )

    def _writable(
        func: Callable, name: str, _exc: Any
    ) -> None:  # read-only files on Windows
        os.chmod(name, stat.S_IWRITE)
        func(name)

    if sys.version_info >= (3, 12):
        shutil.rmtree(target, onexc=_writable)
    else:  # pragma: no cover - Python < 3.12
        shutil.rmtree(target, onerror=_writable)


def _bootstrap_micromamba(
    tools_dir: str, subdir: str, offline: bool, url: str, sha256: str
) -> None:
    from .._bootstrap import install_micromamba, micromamba_archive  # noqa: PLC0415

    # ``url`` and ``sha256`` are in the plan so people see them; the pins in
    # the code decide, and a plan document that names others is refused.
    pinned = micromamba_archive(subdir)
    if (url, sha256) != (pinned["url"], pinned["sha256"]):
        raise ExecutionError(
            f"the plan names {url} / {sha256}, the pinned archive is"
            f" {pinned['url']} / {pinned['sha256']}",
            step="bootstrap-micromamba",
        )
    install_micromamba(tools_dir, subdir, offline=offline)


def _bootstrap_uv(
    tools_dir: str,
    requirement: str,
    python: str,
    run: Callable[..., Any] = subprocess.run,
) -> None:
    from .._bootstrap import UV_REQUIREMENT, UV_WHEEL_SHA256  # noqa: PLC0415

    if requirement != UV_REQUIREMENT:
        raise ExecutionError(
            f"the plan names {requirement!r}; the pinned requirement is"
            f" {UV_REQUIREMENT!r}",
            step="bootstrap-uv",
        )
    dest_dir = Path(tools_dir) / "bin"
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="skplt-uv-") as tmp:
        pinned = Path(tmp) / "uv-requirements.txt"
        pinned.write_text(
            requirement
            + "".join(f" --hash=sha256:{h}" for h in UV_WHEEL_SHA256)
            + "\n",
            encoding="utf-8",
        )
        target = Path(tmp) / "target"
        done = run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--no-deps",
                "--require-hashes",
                "--only-binary=:all:",
                "--target",
                str(target),
                "-r",
                str(pinned),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            check=False,
        )
        if done.returncode != 0:
            tail = "\n".join((done.stdout or "").splitlines()[-_TAIL:])
            raise ExecutionError(
                f"pip could not install {requirement}:\n{tail}",
                step="bootstrap-uv",
                returncode=done.returncode,
            )
        found = [
            p
            for d in ("bin", "Scripts")
            for n in ("uv", "uv.exe")
            for p in [target / d / n]
            if p.is_file()
        ]
        if len(found) != 1:
            raise ExecutionError(
                f"expected one uv executable from {requirement}, found"
                f" {[str(p) for p in found]}",
                step="bootstrap-uv",
            )
        dest = dest_dir / found[0].name
        shutil.copyfile(found[0], dest.with_suffix(dest.suffix + ".tmp"))
        os.chmod(dest.with_suffix(dest.suffix + ".tmp"), 0o755)  # noqa: S103 (a binary)
        os.replace(dest.with_suffix(dest.suffix + ".tmp"), dest)
    logger.info("installed %s at %s", requirement, dest)


def _xeus_pip_layer(
    prefix: str, requirements: Sequence[str], index_args: Sequence[str] = ()
) -> None:
    from .._xeus._pip import install_pure_python  # noqa: PLC0415

    install_pure_python(prefix, list(requirements), index_args=list(index_args))


#: ``name -> function(**params)`` of every plan action.
ACTIONS: Mapping[str, Callable[..., Any]] = MappingProxyType(
    {
        "remove_tree": _remove_tree,
        "bootstrap_micromamba": _bootstrap_micromamba,
        "bootstrap_uv": _bootstrap_uv,
        "xeus_pip_layer": _xeus_pip_layer,
    }
)


# --- running ---------------------------------------------------------------


def _run_command(
    argv: Sequence[str], *, cwd: str | None, env: Mapping[str, str]
) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603 (an argv list from the plan; no shell)
        list(argv),
        cwd=cwd,
        env=dict(env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        check=False,
    )


def _command_env(step: Step, environ: Mapping[str, str]) -> dict[str, str]:
    env = {k: v for k, v in environ.items() if k not in _DROPPED}
    env.update(step.env)
    return env


def _approval_message(plan: EnvironmentPlan, kind: str) -> str:
    titles = [s.title for s in plan.steps if s.approval == kind]
    return f"{kind}: " + "; ".join(titles)


def execute(
    plan: EnvironmentPlan,
    *,
    decider: Decider | None = None,
    events: EventSink | None = None,
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
    actions: Mapping[str, Callable[..., Any]] | None = None,
    environ: Mapping[str, str] | None = None,
    lock_timeout: float = 0.0,
) -> ExecutionResult:
    """
    Run ``plan``.

    Parameters
    ----------
    plan : EnvironmentPlan
        A plan from :py:func:`~._planner.plan_request`.
    decider : Decider or None, default=None
        Approval decisions; ``Decider(plan's interaction policy)`` when None.
    events : EventSink or None, default=None
        Event sink; logger only when None.
    runner : callable or None, default=None
        ``runner(argv, cwd=..., env=...)`` returning an object with
        ``returncode`` and ``stdout``.
    actions : mapping or None, default=None
        Action table; :py:data:`ACTIONS` when None.
    environ : mapping or None, default=None
        Base environment of commands; :py:data:`os.environ` when None.
    lock_timeout : float, default=0.0
        Seconds to wait for another run on the same prefix.

    Returns
    -------
    ExecutionResult
        The outcome.

    Raises
    ------
    PlanningError
        For a plan-only target.
    ApprovalRequiredError
        When an approval is refused (nothing has run).
    LockBusyError
        When another run holds the prefix.
    ExecutionError
        When a step fails; ``exc.result`` is the partial result.
    """
    if not plan.executable:
        raise PlanningError(
            f"target {plan.target!r} is plan-only; use the plan's artifacts",
            hint="see plan.artifacts",
        )
    events = EventSink() if events is None else events
    decider = (
        Decider(plan.request.get("policy", {}).get("interaction", "auto"))
        if decider is None
        else decider
    )
    runner = _run_command if runner is None else runner
    table = ACTIONS if actions is None else actions
    environ = os.environ if environ is None else environ
    events.emit(
        "plan.started",
        digest=plan.digest,
        prefix=plan.prefix,
        backend=plan.backend,
        operation=plan.operation,
        steps=len(plan.steps),
        up_to_date=plan.up_to_date,
    )
    for kind in plan.approvals:
        decider.require(kind, _approval_message(plan, kind))
        events.emit("approval.granted", kind=kind)
    results: list[StepResult] = []
    if not plan.steps:
        events.emit("plan.finished", ok=True, changed=False)
        return ExecutionResult(plan, True, (), approvals=tuple(decider.transcript))
    Path(plan.prefix).parent.mkdir(parents=True, exist_ok=True)
    with PrefixLock(plan.prefix, timeout=lock_timeout):
        if _done_meanwhile(plan):
            events.emit("plan.finished", ok=True, changed=False, concurrent=True)
            return ExecutionResult(plan, True, (), approvals=tuple(decider.transcript))
        for index, step in enumerate(plan.steps, 1):
            events.emit(
                "step.started",
                step=step.id,
                index=index,
                title=step.title,
                command=step.command_line(),
            )
            start = time.monotonic()
            try:
                returncode = _run_step(step, runner, table, environ)
            except ExecutionError as exc:
                seconds = round(time.monotonic() - start, 3)
                results.append(StepResult(step.id, False, seconds, exc.returncode))
                events.emit(
                    "step.failed",
                    step=step.id,
                    seconds=seconds,
                    returncode=exc.returncode,
                    error=redact(str(exc)),
                )
                exc.result = ExecutionResult(
                    plan, False, tuple(results), approvals=tuple(decider.transcript)
                )
                raise
            seconds = round(time.monotonic() - start, 3)
            results.append(StepResult(step.id, True, seconds, returncode))
            events.emit("step.finished", step=step.id, seconds=seconds)
        if plan.operation != Operation.REMOVE.value:
            try:
                _check_built(plan)
            except ExecutionError as exc:
                events.emit("plan.failed", error=redact(str(exc)))
                exc.result = ExecutionResult(
                    plan, False, tuple(results), approvals=tuple(decider.transcript)
                )
                raise
        receipt = None
        if plan.operation != Operation.REMOVE.value:
            receipt = str(
                write_receipt(  # redacted: a receipt is read by other tools
                    plan.prefix,
                    redact_data(
                        {
                            "python": plan.python,
                            "dependencies": [dict(d) for d in plan.dependencies],
                            "plan_digest": plan.digest,
                            "state_digest": plan.state_digest,
                            "backend": plan.backend,
                            "target": plan.target,
                            "python_executable": plan.python_executable,
                            "request": plan.request,
                            "inputs": [list(i) for i in plan.inputs],
                            "steps": [r.to_dict() for r in results],
                            "built": (
                                dt.datetime.now(dt.timezone.utc).isoformat(
                                    timespec="seconds"
                                )
                            ),
                        }
                    ),
                )
            )
    events.emit("plan.finished", ok=True, changed=True, receipt=receipt)
    return ExecutionResult(
        plan, True, tuple(results), receipt, approvals=tuple(decider.transcript)
    )


def _done_meanwhile(plan: EnvironmentPlan) -> bool:
    """
    Re-check, under the lock, the state the plan was made for.

    Parameters
    ----------
    plan : EnvironmentPlan
        The plan about to run.

    Returns
    -------
    bool
        True when another run has already built exactly this state (nothing
        to do).

    Raises
    ------
    ExecutionError
        With code ``plan_stale`` when the environment appeared or disappeared
        since the plan was made (the plan's steps would not fit it).
    """
    if plan.operation == Operation.REMOVE.value:
        return False  # deleting an already deleted tree is a no-op
    receipt = read_receipt(plan.prefix)
    if (
        plan.operation == Operation.ENSURE.value
        and receipt is not None
        and receipt.get("state_digest") == plan.state_digest
    ):
        return True
    python = plan.python_executable
    now = is_environment(plan.prefix) and (
        Path(python).exists() if python else (Path(plan.prefix) / "conda-meta").is_dir()
    )
    if now != plan.exists:
        raise ExecutionError(
            f"{plan.prefix} {'appeared' if now else 'disappeared'} while this run waited"
            " for the lock",
            step=plan.steps[0].id,
            code="plan_stale",
            hint="run the command again; it plans for the environment as it is now",
        )
    return False


def _check_built(plan: EnvironmentPlan) -> None:
    """
    Fail when the steps succeeded but left no environment (a tool's silent no-op).

    Parameters
    ----------
    plan : EnvironmentPlan
        The executed plan.

    Raises
    ------
    ExecutionError
        When the prefix or the planned Python does not exist.
    """
    if not Path(plan.prefix).is_dir():
        raise ExecutionError(
            f"every step succeeded but {plan.prefix} does not exist",
            step=plan.steps[-1].id,
        )
    if plan.python_executable and not Path(plan.python_executable).exists():
        raise ExecutionError(
            f"every step succeeded but {plan.python_executable} does not exist",
            step=plan.steps[-1].id,
            hint="the backend built a different layout; report"
            " it with the plan (--format json)",
        )


def _run_step(
    step: Step,
    runner: Callable[..., Any],
    table: Mapping[str, Callable[..., Any]],
    environ: Mapping[str, str],
) -> int | None:
    if step.kind == "action":
        if step.action not in table:
            raise ExecutionError(f"unknown action {step.action!r}", step=step.id)
        try:
            table[step.action](**step.params_dict)
        except ExecutionError as exc:
            exc.step = step.id
            raise
        except OSError as exc:
            raise ExecutionError(f"{step.title}: {exc}", step=step.id) from exc
        return None
    try:
        done = runner(step.argv, cwd=step.cwd, env=_command_env(step, environ))
    except OSError as exc:  # executable missing, permission denied
        raise ExecutionError(
            f"cannot run {step.argv[0]}: {exc}",
            step=step.id,
            hint="run 'scikitplot env doctor' to see the tools found",
        ) from exc
    output = done.stdout or ""
    for line in output.splitlines():
        logger.debug("[%s] %s", step.id, redact(line))
    if done.returncode != 0:
        tail = "\n".join(output.splitlines()[-_TAIL:])
        raise ExecutionError(
            f"step {step.id!r} failed with exit status {done.returncode}:\n{redact(tail)}",
            step=step.id,
            returncode=done.returncode,
            hint="the full output is logged at DEBUG level (command line: -vv)",
        )
    return done.returncode
