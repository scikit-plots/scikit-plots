# scikitplot/_ipymagic/_snakeviz.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Optional interactive profile viewer: a snakeviz server with a managed lifetime.

Notes
-----
**User notes.** ``%scikitplot_profile --snakeviz ...`` starts ``snakeviz``
(``pip install snakeviz``) on ``127.0.0.1`` and shows it in the notebook. It
works when the browser runs on the same machine as the kernel (a local
Jupyter); on a remote kernel use the table, or open the ``.prof`` file with
snakeviz where you are. Servers stop with ``%scikitplot_profile
--stop-snakeviz``, when the extension is unloaded, and when the kernel exits.

**Developer notes.** Fixed from the copy of snakeviz's own magic this package
held: the server listens on ``127.0.0.1``, never ``0.0.0.0`` (every network
interface); readiness has a deadline instead of an unbounded ``readline``
loop (which hung the kernel when the server failed); the URL is read from
snakeviz's own output (it moves to another port when one is taken) and its
host is checked by parsing; the server is not killed after a fixed
``sleep(3)`` that could be shorter than the page load.

.. seealso::
  * https://github.com/jiffyclub/snakeviz
  * https://jiffyclub.github.io/snakeviz/
  * https://pypi.org/project/snakeviz/
"""

from __future__ import annotations

import atexit
import contextlib
import importlib.util
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from ._errors import CapabilityMissingError, IpyMagicError

__all__ = ["SnakevizServer", "require_snakeviz", "start_snakeviz", "stop_all"]

#: Seconds snakeviz may take to print its address.
START_TIMEOUT = 15.0

_READY = "snakeviz web server started on"
_LOCK = threading.Lock()
_SERVERS: list[SnakevizServer] = []
_ATEXIT = False


@dataclass
class SnakevizServer:  # numpydoc ignore=PR02
    """
    A running snakeviz server.

    Parameters
    ----------
    url : str
        Address of the profile page.
    stats_file : str
        The profile it shows.
    process : subprocess.Popen
        The server process.
    """

    url: str
    stats_file: str
    process: subprocess.Popen = field(repr=False)

    def stop(self) -> None:
        """Stop the server (terminate, then kill after 5 s); idempotent."""
        proc = self.process
        if proc.poll() is not None:
            if proc.stdout is not None:
                with contextlib.suppress(OSError, ValueError):
                    proc.stdout.close()
            return
        if os.name == "posix":
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGTERM)
        else:  # pragma: no cover - Windows
            proc.terminate()
        try:
            proc.wait(5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        finally:
            if proc.stdout is not None:
                with contextlib.suppress(OSError, ValueError):
                    proc.stdout.close()


def _available() -> bool:
    """
    Return whether snakeviz can be imported.

    Returns
    -------
    bool
        False also when ``sys.modules['snakeviz'] is None`` (blocked).
    """
    if "snakeviz" in sys.modules:
        return sys.modules["snakeviz"] is not None
    try:
        return importlib.util.find_spec("snakeviz") is not None
    except (ImportError, ValueError):
        return False


def require_snakeviz() -> None:
    """
    Raise unless snakeviz can be imported.

    Raises
    ------
    CapabilityMissingError
        When snakeviz is not installed (or blocked with
        ``sys.modules['snakeviz'] = None``).
    """
    if not _available():
        raise CapabilityMissingError(
            "snakeviz is not installed",
            hint="pip install scikit-plots[ipymagic] (or snakeviz); the table needs nothing",
        )


def _drain(stream, sink: queue.Queue) -> None:
    try:
        for line in iter(stream.readline, ""):
            with contextlib.suppress(queue.Full):  # once serving, nobody reads
                sink.put_nowait(line)
    except (OSError, ValueError):
        pass
    finally:
        with contextlib.suppress(queue.Full):
            sink.put_nowait(None)


def start_snakeviz(
    stats_file: str, *, timeout: float = START_TIMEOUT
) -> SnakevizServer:
    """
    Start snakeviz for ``stats_file`` on 127.0.0.1 and wait until it serves.

    Parameters
    ----------
    stats_file : str
        A :py:mod:`pstats` file.
    timeout : float, default=15.0
        Seconds to wait for the server's address.

    Returns
    -------
    SnakevizServer
        The running server (registered: stopped at exit).

    Raises
    ------
    CapabilityMissingError
        When snakeviz is not installed.
    IpyMagicError
        When the server exits or prints no address in time (its output is in
        the message); the process is stopped.
    """
    global _ATEXIT  # noqa: PLW0603
    require_snakeviz()
    proc = subprocess.Popen(  # noqa: S603 - argv list, no shell
        [
            sys.executable,
            "-m",
            "snakeviz",
            "-s",
            "-H",
            "127.0.0.1",
            os.path.abspath(stats_file),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        start_new_session=os.name == "posix",
    )
    server = SnakevizServer(url="", stats_file=stats_file, process=proc)
    sink: queue.Queue = queue.Queue(maxsize=1000)
    threading.Thread(target=_drain, args=(proc.stdout, sink), daemon=True).start()
    seen: list[str] = []
    deadline = time.monotonic() + timeout
    ready = False
    while time.monotonic() < deadline:
        try:
            line = sink.get(timeout=0.1)
        except queue.Empty:
            continue
        if line is None:
            break
        seen.append(line)
        if line.startswith(_READY):
            ready = True
        elif ready and line.strip().startswith("http"):
            url = line.strip()
            if urlsplit(url).hostname != "127.0.0.1":
                server.stop()
                raise IpyMagicError(
                    f"snakeviz serves on {url}, not 127.0.0.1", code="snakeviz_start"
                )
            server.url = url
            break
    if not server.url:
        exited = proc.poll()
        server.stop()
        output = "".join(seen).strip()[:4000] or "(no output)"
        reason = (
            f"exited with status {exited}"
            if exited is not None
            else f"printed no address within {timeout:g} s"
        )
        raise IpyMagicError(
            f"snakeviz {reason}: {output}",
            code="snakeviz_start",
            hint="run 'python -m snakeviz -s <file>' to see why",
        )
    with _LOCK:
        _SERVERS.append(server)
        if not _ATEXIT:
            atexit.register(stop_all)
            _ATEXIT = True
    return server


def stop_all() -> int:
    """
    Stop every server this process started.

    Returns
    -------
    int
        How many were stopped.
    """
    with _LOCK:
        servers, _SERVERS[:] = list(_SERVERS), []
    for server in servers:
        server.stop()
    return len(servers)
