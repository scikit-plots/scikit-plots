# scikitplot/_ipymagic/_bridge.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The engine behind ``%scikitplot``: run any ``scikitplot`` command and return a result.

Notes
-----
**User notes.** :py:func:`run_command` is the magic without IPython:
``run_command("info --format json").data`` from a script, a test or an
agent. Where a command runs is decided by the CLI's own registry:

.. list-table::
   :header-rows: 1

   * - Command
     - Runs
   * - native (``info``, ``doctor``, ``show-versions`` ...)
     - in the kernel, isolated: its output is captured, the ``scikitplot``
       logger is restored afterwards
   * - delegated (``env``, ``mcp``, ``cleanprompt``), any command with cell
       input, or every command with ``isolate=True``
     - in a subprocess ``python -m scikitplot`` (same interpreter as the
       kernel), output shown line by line, stopped (with its children) by an
       interrupt or the timeout

A word that is not a command is refused with the closest names; a guess is
never run. With ``--format json`` the output is parsed into
:py:attr:`.CommandResult.data` and shown as JSON.

**Developer notes.** The routing reads :py:mod:`scikitplot._cli.registry`
only; no command, option or exit status is declared here, so a new CLI
command works in notebooks without a change to this module.
"""

from __future__ import annotations

import contextlib
import difflib
import io
import json
import logging
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import IO, Callable, Iterator, Sequence

from ._errors import CommandFailedError, MagicUsageError
from ._results import CommandResult
from ._settings import MagicSettings

__all__ = ["Route", "route", "run_command", "split_line", "wants_json"]

logger = logging.getLogger(__name__)

#: Characters kept per captured stream. Output streamed while a command runs
#: and in-process output are shown in full; output shown after a subprocess
#: ends is the capture, followed by a note when it was cut.
MAX_CAPTURE = 1_048_576

#: Exit status of a command stopped by the timeout (as GNU ``timeout``).
EXIT_TIMEOUT = 124

#: Exit status when the child process could not be started.
EXIT_NOT_STARTED = 127

#: Seconds a stopped child gets between terminate and kill.
STOP_GRACE = 5.0


def split_line(line: str) -> list[str]:
    r"""
    Split a magic's line into arguments the way a POSIX shell would.

    Parameters
    ----------
    line : str
        Text after the magic's name (IPython has already expanded
        ``{var}`` and ``$var``).

    Returns
    -------
    list of str
        Arguments.

    Raises
    ------
    MagicUsageError
        For an unterminated quote.

    Notes
    -----
    Quote a Windows path (``"C:\Users\me\env.toml"``): outside quotes a
    backslash escapes the next character, as in any POSIX shell.

    Examples
    --------
    >>> from scikitplot._ipymagic import split_line
    >>> split_line('env plan -n "my env" --package six')
    ['env', 'plan', '-n', 'my env', '--package', 'six']
    """
    try:
        return shlex.split(line, posix=True)
    except ValueError as exc:
        raise MagicUsageError(
            f"cannot split the arguments: {exc}", hint="close the quote"
        ) from None


@dataclass(frozen=True)
class Route:  # numpydoc ignore=PR02
    """
    Which command an argument list names, and whether the CLI delegates it.

    Parameters
    ----------
    command : str or None
        Canonical command name; None when the arguments name none (root
        ``--help``, ``--version``, or an unknown root option the CLI reports).
    delegated : bool
        The command is a pass-through to a submodule's own CLI.
    """

    command: str | None
    delegated: bool


def _names() -> list[str]:
    from .._cli import registry  # noqa: PLC0415

    return sorted({*registry.BY_NAME, *registry.BY_ALIAS})


def route(argv: Sequence[str]) -> Route:
    """
    Return the command ``argv`` names, as the CLI would find it.

    Parameters
    ----------
    argv : sequence of str
        Arguments after ``scikitplot``.

    Returns
    -------
    Route
        The command and how it runs.

    Raises
    ------
    MagicUsageError
        When the first word that is not an option is not a command or an
        alias (and no delegated command follows); the closest names are in
        the hint.

    Notes
    -----
    The delegated decision is the CLI's own (``_split_delegated``): the first
    word anywhere in ``argv`` that names a command decides, so ``-vq env``,
    ``--verb env`` and ``-v env`` all delegate, exactly as on the command line.
    The root parser takes no option with a value, so the first word that is
    not an option is the one argparse and click read as the command.

    Examples
    --------
    >>> from scikitplot._ipymagic import route
    >>> route(["-v", "env", "plan"])
    Route(command='env', delegated=True)
    """
    from .._cli import registry  # noqa: PLC0415

    first = next((spec for spec in map(registry.resolve, argv) if spec), None)
    if first is not None and first.delegate is not None:
        return Route(first.name, True)
    word = next((token for token in argv if not token.startswith("-")), None)
    if word is None:
        return Route(None, False)  # root --help/--version, or the CLI's error
    spec = registry.resolve(word)
    if spec is None:
        close = difflib.get_close_matches(word, _names(), n=3, cutoff=0.6)
        raise MagicUsageError(
            f"{word!r} is not a scikitplot command",
            hint=(
                f"did you mean {' or '.join(repr(c) for c in close)}?"
                if close
                else f"commands: {', '.join(_names())}"
            ),
        )
    return Route(spec.name, False)


def wants_json(argv: Sequence[str]) -> bool:
    """
    Return whether ``argv`` asks for JSON output (``--format json``).

    Parameters
    ----------
    argv : sequence of str
        Arguments.

    Returns
    -------
    bool
        True for ``--format json`` or ``--format=json`` anywhere.
    """
    for i, word in enumerate(argv):
        if word == "--format=json":
            return True
        if word == "--format" and i + 1 < len(argv) and argv[i + 1] == "json":
            return True
    return False


class _Capture:
    """A bounded text buffer (keeps the first :data:`MAX_CAPTURE` characters)."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._size = 0
        self.truncated = False

    def add(self, text: str) -> None:
        room = MAX_CAPTURE - self._size
        if room <= 0:
            self.truncated = self.truncated or bool(text)
            return
        if len(text) > room:
            text, self.truncated = text[:room], True
        self._parts.append(text)
        self._size += len(text)

    def value(self) -> str:
        return "".join(self._parts)


def _bounded(text: str) -> tuple[str, bool]:
    capture = _Capture()
    capture.add(text)
    return capture.value(), capture.truncated


# --- in process --------------------------------------------------------------


@contextlib.contextmanager
def _isolated_cli_state() -> Iterator[None]:
    """
    Restore, after a CLI run, everything the CLI configures for logging.

    Yields
    ------
    None
        Inside: the CLI may configure the ``scikitplot`` logger freely.

    Notes
    -----
    ``scikitplot._cli.logging.configure`` sets the ``scikitplot`` logger's
    level, adds a stderr handler once (module global ``_HANDLER``), turns
    propagation off and re-points the handler at the current ``sys.stderr``.
    In a kernel that would last for the session and hide the package's logs
    from the user's own handlers, so all of it is put back.
    """
    from .._cli import logging as cli_logging  # noqa: PLC0415

    target = logging.getLogger("scikitplot")
    level, handlers = target.level, list(target.handlers)
    propagate, disabled = target.propagate, target.disabled
    saved_handler = cli_logging._HANDLER  # noqa: SLF001
    saved_stream = getattr(saved_handler, "stream", None)
    try:
        yield
    finally:
        for handler in list(target.handlers):
            if handler not in handlers:
                target.removeHandler(handler)
        for handler in handlers:
            if handler not in target.handlers:
                target.addHandler(handler)
        target.setLevel(level)
        target.propagate, target.disabled = propagate, disabled
        cli_logging._HANDLER = saved_handler  # noqa: SLF001
        if saved_handler is not None:
            saved_handler.stream = saved_stream


def _exit_status(code: object, err: IO[str]) -> int:
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    err.write(f"{code}\n")  # SystemExit("message"): the interpreter would print it
    return 1


@contextlib.contextmanager
def _noting_interrupts() -> Iterator[list[int]]:
    """
    Record a SIGINT that arrives inside the block, then raise as usual.

    Yields
    ------
    list of int
        Gets an item for each interrupt; empty when none arrived.

    Notes
    -----
    ``scikitplot._cli.app.main`` turns ``KeyboardInterrupt`` into exit status
    130, which is right for a process but would let "Run all" continue in a
    kernel. The handler is replaced only while Python's default one is
    installed and only in the main thread (the only place SIGINT is
    delivered); otherwise nothing changes and nothing is recorded.
    """
    seen: list[int] = []
    if (
        threading.current_thread() is not threading.main_thread()
        or signal.getsignal(signal.SIGINT) is not signal.default_int_handler
    ):
        yield seen
        return

    def handler(signum: int, frame: object) -> None:
        seen.append(signum)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handler)
    try:
        yield seen
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)


def _run_in_process(argv: Sequence[str]) -> tuple[int, str, str]:
    from .._cli.app import main  # noqa: PLC0415

    out, err = io.StringIO(), io.StringIO()
    with _isolated_cli_state(), contextlib.redirect_stdout(
        out
    ), contextlib.redirect_stderr(err), _noting_interrupts() as interrupts:
        try:
            code = main(list(argv))
        except SystemExit as exc:  # argparse usage errors, --help
            code = _exit_status(exc.code, err)
    if interrupts:
        raise KeyboardInterrupt
    return int(code), out.getvalue(), err.getvalue()


# --- subprocess --------------------------------------------------------------


def _child_argv(argv: Sequence[str]) -> list[str]:
    """
    Return the child's command line: the kernel's interpreter, ``-m scikitplot``.

    Parameters
    ----------
    argv : sequence of str
        Arguments after ``scikitplot``.

    Returns
    -------
    list of str
        The command line (the seam tests patch to run a stand-in child).
    """
    return [sys.executable, "-m", "scikitplot", *argv]


def _child_env() -> dict[str, str]:
    # Rule 77: start from os.environ (Windows needs SystemRoot).
    return {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}


def _pump(stream: IO[str], kind: str, sink: queue.Queue, done: threading.Event) -> None:
    # The reader closes its own pipe: closing it from another thread blocks
    # while ``readline`` holds the stream's lock (a process that outlives the
    # command keeps the pipe open). Once the run is over (``done``) lines are
    # read and dropped, so such a process neither blocks nor fills memory.
    try:
        for line in iter(stream.readline, ""):
            if not done.is_set():
                sink.put((kind, line))
    except (OSError, ValueError):
        pass
    finally:
        with contextlib.suppress(OSError, ValueError):
            stream.close()
        sink.put((kind, None))


def _feed(stream: IO[str], text: str) -> None:
    try:
        stream.write(text)
    except (BrokenPipeError, OSError, ValueError):
        pass  # the command stopped reading; its exit status tells why
    finally:
        with contextlib.suppress(OSError, ValueError):
            stream.close()


def _stop(proc: subprocess.Popen) -> None:
    """
    Stop ``proc`` and its children: terminate, then kill after a grace time.

    Parameters
    ----------
    proc : subprocess.Popen
        The child to stop. On POSIX its whole process group is signalled even
        when the child itself has exited (processes it left running).

    Notes
    -----
    POSIX: SIGTERM to the group, up to :data:`STOP_GRACE` seconds for the
    child, then SIGKILL to the group (whatever is still running in it).
    A process that started its own session is beyond reach. Windows:
    ``taskkill /T /F`` on a running child; the tree of an exited child
    cannot be found.
    """
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGTERM)  # its own session: children too
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(STOP_GRACE)
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)  # whatever is left in the group
        proc.wait()
        return
    if proc.poll() is not None:  # pragma: no cover - Windows
        return
    taskkill = os.path.join(  # pragma: no cover - Windows
        os.environ.get("SYSTEMROOT", r"C:\Windows"), "System32", "taskkill.exe"
    )
    with contextlib.suppress(OSError):  # pragma: no cover - Windows
        subprocess.run(  # noqa: S603 - fixed argv, no shell
            [taskkill, "/T", "/F", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
    with contextlib.suppress(OSError):  # pragma: no cover - Windows
        proc.kill()
    proc.wait()  # pragma: no cover - Windows


def _collect(
    proc: subprocess.Popen,
    sink: queue.Queue,
    deadline: float | None,
    take: Callable[[str, str], None],
) -> bool:
    """
    Pass the child's lines to ``take`` until both pipes close or time runs out.

    Parameters
    ----------
    proc : subprocess.Popen
        The child.
    sink : queue.Queue
        ``(kind, line)`` items from the reader threads; ``line`` None at EOF.
    deadline : float or None
        :py:func:`time.monotonic` value to stop at.
    take : callable
        ``take(kind, line)``.

    Returns
    -------
    bool
        Whether the deadline stopped the command itself (a command that had
        exited is never reported as timed out).

    Notes
    -----
    A process the command started and left running keeps the pipes open
    after the command exits. The wait for it ends :data:`STOP_GRACE` seconds
    after the exit, or at the deadline if that is sooner; it is then stopped
    with the command's process group (:func:`_stop`). Output it writes later
    is not collected, and the readers end when it closes the pipes.
    """
    open_streams = 2
    exited_at: float | None = None
    while open_streams:
        wait = 0.1
        now = time.monotonic()
        if proc.poll() is not None:
            exited_at = now if exited_at is None else exited_at
            if now - exited_at > STOP_GRACE or (
                deadline is not None and now >= deadline
            ):
                _stop(proc)
                return False
        if deadline is not None:
            left = deadline - now
            if left <= 0:
                _stop(proc)
                return True
            wait = min(wait, left)
        try:
            kind, line = sink.get(timeout=wait)
        except queue.Empty:
            continue
        if line is None:
            open_streams -= 1
        else:
            take(kind, line)
    proc.wait()
    return False


def _run_subprocess(
    argv: Sequence[str],
    *,
    stdin: str | None,
    timeout: float | None,
    show: dict[str, IO[str] | None],
) -> tuple[int, str, str, bool, bool]:
    command = _child_argv(argv)
    try:
        proc = subprocess.Popen(  # noqa: S603 - argv list, no shell
            command,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",  # Rule 45
            env=_child_env(),
            start_new_session=os.name == "posix",
        )
    except OSError as exc:
        return (
            EXIT_NOT_STARTED,
            "",
            f"cannot start {command[0]}: {exc}\n",
            False,
            False,
        )
    sink: queue.Queue = queue.Queue()
    done = threading.Event()
    threads = [
        threading.Thread(target=_pump, args=(pipe, kind, sink, done), daemon=True)
        for pipe, kind in ((proc.stdout, "out"), (proc.stderr, "err"))
    ]
    if stdin is not None:
        threads.append(
            threading.Thread(target=_feed, args=(proc.stdin, stdin), daemon=True)
        )
    for thread in threads:
        thread.start()
    captures = {"out": _Capture(), "err": _Capture()}

    def take(kind: str, line: str) -> None:
        captures[kind].add(line)
        target = show.get(kind)
        if target is not None:
            target.write(line)
            target.flush()

    deadline = None if timeout is None else time.monotonic() + timeout
    try:
        timed_out = _collect(proc, sink, deadline, take)
    except KeyboardInterrupt:
        _stop(proc)
        raise
    finally:
        if proc.poll() is None:
            _stop(proc)  # an unexpected error in the loop: no orphan child
        for thread in threads:  # the readers close their own pipes
            thread.join(timeout=1.0)
        while True:  # lines read before the pipes closed
            try:
                kind, line = sink.get_nowait()
            except queue.Empty:
                break
            if line is not None:
                take(kind, line)
        done.set()
    code = EXIT_TIMEOUT if timed_out else proc.wait()
    truncated = captures["out"].truncated or captures["err"].truncated
    return code, captures["out"].value(), captures["err"].value(), timed_out, truncated


# --- the entry point ---------------------------------------------------------


def _argv(args: str | Sequence[str]) -> list[str]:
    if isinstance(args, str):
        return split_line(args)
    argv = list(args)
    for i, item in enumerate(argv):
        if not isinstance(item, str):
            raise MagicUsageError(
                f"argument {i} is {type(item).__name__}, not str",
                hint="pass a string or a list of strings",
            )
    return argv


def run_command(
    args: str | Sequence[str],
    *,
    stdin: str | None = None,
    settings: MagicSettings | None = None,
    out: IO[str] | None = None,
    err: IO[str] | None = None,
) -> CommandResult:
    """
    Run one ``scikitplot`` command and return its result.

    Parameters
    ----------
    args : str or sequence of str
        The command line after ``scikitplot`` (``"env plan -n demo"``) or its
        arguments.
    stdin : str or None, default=None
        Text given to the command on standard input (the cell of
        ``%%scikitplot``); the command then runs in a subprocess.
    settings : MagicSettings or None, default=None
        Timeout, isolation, streaming, raising; defaults when None.
    out, err : text stream or None, default=None
        Where the output is shown; :py:data:`sys.stdout` /
        :py:data:`sys.stderr` at call time when None.

    Returns
    -------
    CommandResult
        Exit status, captured output, timing and, for ``--format json``, the
        parsed data.

    Raises
    ------
    MagicUsageError
        For arguments that name no command (nothing is run).
    CommandFailedError
        For a non-zero exit status when ``settings.raise_on_error``.
    KeyboardInterrupt
        After stopping the command, when interrupted.

    Examples
    --------
    >>> import io
    >>> from scikitplot._ipymagic import run_command
    >>> result = run_command("info --format json", out=io.StringIO())
    >>> result.ok, sorted(result.data)
    (True, ['platform', 'python', 'scikitplot'])
    """
    settings = MagicSettings() if settings is None else settings
    if stdin is not None and not isinstance(stdin, str):
        raise MagicUsageError("stdin must be a str or None")
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    argv = _argv(args)
    where = route(argv)
    as_json = wants_json(argv)
    in_process = not (where.delegated or stdin is not None or settings.isolate)
    start = time.monotonic()
    timed_out = False
    if in_process:
        code, shown_out, shown_err = _run_in_process(argv)
        stdout, cut_out = _bounded(shown_out)
        stderr, cut_err = _bounded(shown_err)
        truncated = cut_out or cut_err
    else:
        show = {
            "out": out if settings.stream and not as_json else None,
            "err": err if settings.stream else None,
        }
        code, stdout, stderr, timed_out, truncated = _run_subprocess(
            argv, stdin=stdin, timeout=settings.timeout, show=show
        )
        shown_out, shown_err = stdout, stderr
    duration = time.monotonic() - start
    data = None
    if as_json and stdout.strip() and not truncated:
        try:
            data = json.loads(stdout)
        except ValueError:
            data = None
    held_out = in_process or not settings.stream or as_json
    if data is None and held_out:  # stdout was not shown while it ran
        out.write(shown_out)
    if in_process or not settings.stream:  # nor was stderr
        err.write(shown_err)
    if truncated and not in_process and held_out:
        err.write(
            f"[output cut to the first {MAX_CAPTURE} characters per stream;"
            " stream=true shows it in full]\n"
        )
    if timed_out:
        err.write(f"stopped after the {settings.timeout:g} s timeout\n")
    result = CommandResult(
        argv=tuple(argv),
        command=where.command,
        mode="in-process" if in_process else "subprocess",
        exit_code=code,
        stdout=stdout,
        stderr=stderr,
        duration=duration,
        data=data,
        timed_out=timed_out,
        truncated=truncated,
    )
    logger.debug(
        "scikitplot %s: exit %d in %.3f s (%s; %d chars out, %d err)",
        where.command or "-",
        code,
        duration,
        result.mode,
        len(stdout),
        len(stderr),
    )
    if settings.raise_on_error and not result.ok:
        raise CommandFailedError(
            result.summary(),
            result=result,
            hint="the command's own message is above; the result is exc.result",
        )
    return result
