"""
Put a guard in front of any model you run as a command.

Local models (``ollama run llama3``), vendor command-line clients, and
agent CLIs all read a prompt on standard input and write an answer on
standard output. :func:`run_command` guards that pipe: the command receives the
encoded prompt, and its answer is decoded as it streams back.

Notes
-----
**User notes.** From the shell::

    python -m scikitplot.cleanprompt ask --via "ollama run llama3" --in notes.txt
    python -m scikitplot.cleanprompt ask --via "llm -m gpt-4o" "Draft a reply to ann@example.com"

The command sees placeholders; you see the real values. The values are held
in memory for the length of the call and are never written to disk.

**Developer notes — what this does not do, deliberately.**

*No shell.* The command is split with :func:`shlex.split` and run with
``shell=False``, so nothing in a prompt or a flag is interpreted by a shell,
and a model's output is never executed.

*No network.* The command is the user's; whatever it connects to is its own
business. This module only moves bytes through pipes.

*No deadlock.* The prompt is written and the command's standard error is
drained on their own threads while standard output is read, so a command that
writes before it has read everything, or writes a lot of diagnostics, cannot
stall the pipe.

See Also
--------
scikitplot.cleanprompt._guard.Guard : The gate this runs through.
"""

from __future__ import annotations

import codecs
import os
import shlex
import subprocess
import threading
from typing import IO

from ._exceptions import CleanPromptError
from ._guard import Guard
from ._logging import audit

__all__ = [
    "run_command",
    "split_command",
]

_CHUNK = 4096


def split_command(command: str) -> list[str]:
    """
    Split a command line into arguments, without a shell.

    Parameters
    ----------
    command : str
        For example ``'ollama run llama3'``.

    Returns
    -------
    list of str
        The argument vector.

    Raises
    ------
    CleanPromptError
        If the command is empty or cannot be split (an unclosed quote).

    Examples
    --------
    >>> split_command('llm -m "gpt 4o"')
    ['llm', '-m', 'gpt 4o']
    """
    try:
        argv = shlex.split(command, posix=os.name != "nt")
    except ValueError as exc:
        msg = f"could not read the command {command!r}: {exc}"
        raise CleanPromptError(msg) from exc
    if not argv:
        msg = "the model command is empty"
        raise CleanPromptError(msg)
    return argv


def run_command(  # ruff: ignore[too-many-positional-arguments]
    guard: Guard,
    argv: list[str],
    prompt: str,
    stdout: IO[str],
    stderr: IO[str],
    timeout: float | None = None,
) -> int:
    """
    Send a guarded prompt to a command and stream its decoded answer.

    Parameters
    ----------
    guard : Guard
        Encodes the prompt, checks it, and decodes the answer.
    argv : list of str
        The command, already split.
    prompt : str
        What you would have sent.
    stdout, stderr : file-like
        Where the decoded answer and the command's diagnostics go.
    timeout : float, optional
        Seconds before the command is stopped.

    Returns
    -------
    int
        The command's exit status.

    Raises
    ------
    LeakError
        Before the command is started, if the check fails.
    CleanPromptError
        If the command cannot be started, or runs past ``timeout``.

    Notes
    -----
    **Developer notes.** The command's standard error is passed through
    decoded as well. A model client that echoes the prompt in an error
    message would otherwise show the user placeholders, and a user who then
    pastes that error into a ticket should not be pasting values either — so
    it is decoded for display here, locally, and never logged.
    """
    safe = guard.outgoing(prompt)
    try:
        process = subprocess.Popen(  # noqa: S603 - argv, never a shell
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        msg = f"could not start {argv[0]!r}: {exc.strerror or exc}"
        raise CleanPromptError(msg) from exc
    audit("sent", command=os.path.basename(argv[0]), chars=len(safe))

    unread: list[str] = []

    def feed() -> None:
        # A command may exit, or close its input, before reading the whole
        # prompt; that is its decision and its exit status reports it. It is
        # recorded, not hidden, and noted to the user below.
        try:
            process.stdin.write(safe.encode("utf-8"))
            process.stdin.close()
        except OSError as exc:
            unread.append(type(exc).__name__)

    errors: list[bytes] = []

    def drain() -> None:
        errors.append(process.stderr.read())

    timed_out = threading.Event()

    def stop() -> None:
        timed_out.set()
        process.kill()

    writer = threading.Thread(target=feed, daemon=True)
    reader = threading.Thread(target=drain, daemon=True)
    timer = threading.Timer(timeout, stop) if timeout else None
    writer.start()
    reader.start()
    if timer is not None:
        timer.start()
    decoder = guard.stream()
    utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")
    try:
        while True:
            block = (
                process.stdout.read1(_CHUNK)
                if hasattr(process.stdout, "read1")
                else process.stdout.read(_CHUNK)
            )
            if not block:
                break
            stdout.write(decoder.feed(utf8.decode(block)))
            stdout.flush()
        stdout.write(decoder.feed(utf8.decode(b"", final=True)))
        stdout.write(decoder.flush())
        stdout.flush()
        status = process.wait()
    finally:
        if timer is not None:
            timer.cancel()
        if process.poll() is None:
            # We are leaving early — our own output pipe closed, or an
            # interrupt. A command left running would block on a full pipe
            # nobody reads, and the joins below would wait for it forever.
            process.kill()
            process.wait()
        process.stdout.close()
        writer.join()
        reader.join()
    if errors and errors[0]:
        stderr.write(guard.incoming(errors[0].decode("utf-8", errors="replace")))
    if unread:
        stderr.write(
            f"note: {argv[0]!r} closed its input before reading the whole prompt ({unread[0]})\n"
        )
    if timed_out.is_set():
        msg = f"{argv[0]!r} ran past {timeout} seconds and was stopped"
        raise CleanPromptError(msg)
    audit("received", command=os.path.basename(argv[0]), status=status)
    return status
