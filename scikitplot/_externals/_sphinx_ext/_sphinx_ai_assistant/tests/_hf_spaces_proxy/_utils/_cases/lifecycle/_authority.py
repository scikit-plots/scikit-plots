"""
One check that a Redis URL was not echoed into a process's output.

Notes
-----
**User notes.** Call :func:`assert_redis_url_not_echoed` with everything a
child process printed and the URL it was given.

**Developer notes.** The earlier checks forbade the port as a bare substring
(``"6379" not in rendered``). The output of a child process also carries
numbers that have nothing to do with the URL: the project's log format prints
a microsecond timestamp and a fifteen-digit thread identifier on every line,
so four given digits turn up by chance about once in several hundred runs, and
a run failed that way. A number is therefore matched as a whole number, not
inside a longer run of digits, which is how a port appears in every form a
leak can take (``host:port``, ``port=...``, ``'port': ...``). The fixtures use
five-digit ports, so a whole-number match cannot be a year, a time field or a
source line either.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

__all__ = ["assert_redis_url_not_echoed", "echoed_parts"]


def echoed_parts(rendered: str, url: str) -> list[str]:
    """
    Return the parts of ``url`` that occur in ``rendered``.

    Parameters
    ----------
    rendered : str
        Everything a process printed, standard output and standard error.
    url : str
        The Redis URL the process was configured with. It must carry a user
        name, a password, a host and a port, so that each is checked.

    Returns
    -------
    list of str
        Names of the parts found: any of ``'scheme'``, ``'username'``,
        ``'password'``, ``'host'``, ``'port'``, in that order.

    Raises
    ------
    ValueError
        If ``url`` lacks one of the parts; a check that silently skipped a
        missing part would pass for the wrong reason.
    """
    parts = urlsplit(url)
    named = {
        "scheme": parts.scheme and parts.scheme + "://",
        "username": parts.username,
        "password": parts.password,
        "host": parts.hostname,
    }
    missing = [name for name, value in named.items() if not value]
    if parts.port is None:
        missing.append("port")
    if missing:
        raise ValueError(f"the URL under test has no {', '.join(missing)}")
    found = [name for name, value in named.items() if value in rendered]
    if re.search(rf"(?<![0-9]){parts.port}(?![0-9])", rendered):
        found.append("port")
    return found


def assert_redis_url_not_echoed(rendered: str, url: str) -> None:
    """
    Assert that no part of ``url`` occurs in ``rendered``.

    Parameters
    ----------
    rendered : str
        Everything a process printed.
    url : str
        The Redis URL the process was configured with.

    Raises
    ------
    AssertionError
        Naming the parts that were echoed. The output itself is not put in
        the message: it would repeat the leak in the test report.
    """
    found = echoed_parts(rendered, url)
    assert found == [], f"the output echoes the Redis URL's {', '.join(found)}"
