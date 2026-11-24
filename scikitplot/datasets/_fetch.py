"""
Bounded HTTP downloads for :mod:`scikitplot.datasets`.

Every network read of the submodule goes through :func:`fetch_bytes` or
:func:`fetch_to_file`: a time limit on the connection and on each read, a
limit on the bytes read (checked against ``Content-Length`` first, then while
streaming), the response closed on every path, and errors that name the URL.

Notes
-----
**User notes.** The limits default to :data:`DEFAULT_TIMEOUT` seconds and
:data:`DEFAULT_MAX_BYTES` bytes; pass ``timeout=`` and ``max_bytes=`` to the
loading function to change them for one call.

**Developer notes.** Standard library only. ``scikitplot.corpus`` has a
fuller downloader (retries, SSRF checks), but no submodule imports another at
module level, and this one only fetches the package's own example data and
URLs its caller names. :func:`fetch_to_file` writes to a temporary file in
the target folder and renames it, so an interrupted download never leaves a
partial file in the cache.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

__all__ = ["DEFAULT_MAX_BYTES", "DEFAULT_TIMEOUT", "fetch_bytes", "fetch_to_file"]

#: Seconds to wait for the connection and for each read.
DEFAULT_TIMEOUT: float = 30.0
#: Largest response read, in bytes (512 MiB).
DEFAULT_MAX_BYTES: int = 512 * 1024 * 1024

_CHUNK = 1024 * 1024
_SCHEMES = ("http", "https")


def _check(url: str, timeout: float | None, max_bytes: int | None) -> tuple[float, int]:
    if urlparse(url).scheme not in _SCHEMES:
        raise ValueError(f"only http and https URLs are downloaded; got {url!r}.")
    timeout = DEFAULT_TIMEOUT if timeout is None else timeout
    max_bytes = DEFAULT_MAX_BYTES if max_bytes is None else max_bytes
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
    ):
        raise ValueError(f"timeout must be a number of seconds > 0; got {timeout!r}.")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError(f"max_bytes must be an int > 0; got {max_bytes!r}.")
    return float(timeout), max_bytes


def _stream(url: str, timeout: float, max_bytes: int, sink) -> int:
    """Copy the response of *url* into ``sink.write``; return the bytes copied."""
    try:
        with urlopen(url, timeout=timeout) as response:  # noqa: S310 - scheme checked
            final = response.geturl()
            if urlparse(final).scheme not in _SCHEMES:
                raise ValueError(
                    f"{url!r} redirected to {final!r}, which is not http(s)."
                )
            declared = response.headers.get("Content-Length")
            if (
                declared is not None
                and declared.isdigit()
                and int(declared) > max_bytes
            ):
                raise ValueError(
                    f"{url!r} declares {int(declared):,} bytes, over max_bytes={max_bytes:,}."
                )
            total = 0
            while True:
                block = response.read(_CHUNK)
                if not block:
                    return total
                total += len(block)
                if total > max_bytes:
                    raise ValueError(
                        f"{url!r} sent more than max_bytes={max_bytes:,} bytes; stopped."
                    )
                sink.write(block)
    except (URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise OSError(f"cannot download {url!r}: {reason}") from exc


def fetch_bytes(
    url: str, *, timeout: float | None = None, max_bytes: int | None = None
) -> bytes:
    """
    Download *url* into memory, within a time and a size limit.

    Parameters
    ----------
    url : str
        An ``http`` or ``https`` URL.
    timeout : float or None, default None
        Seconds for the connection and for each read; ``None`` uses
        :data:`DEFAULT_TIMEOUT`.
    max_bytes : int or None, default None
        Largest response accepted; ``None`` uses :data:`DEFAULT_MAX_BYTES`.

    Returns
    -------
    bytes
        The response body.

    Raises
    ------
    ValueError
        For another scheme, a redirect to one, a bad limit, or a response
        over ``max_bytes`` (declared or sent).
    OSError
        When the server cannot be reached or does not answer in time; the
        message names the URL.
    """
    timeout, max_bytes = _check(url, timeout, max_bytes)
    parts: list[bytes] = []

    class _Sink:
        write = parts.append

    _stream(url, timeout, max_bytes, _Sink)
    return b"".join(parts)


def fetch_to_file(
    url: str,
    path: str | os.PathLike,
    *,
    timeout: float | None = None,
    max_bytes: int | None = None,
) -> Path:
    """
    Download *url* to *path* atomically, within a time and a size limit.

    Parameters
    ----------
    url : str
        An ``http`` or ``https`` URL.
    path : str or os.PathLike
        Destination file; its folder must exist.
    timeout, max_bytes
        As in :func:`fetch_bytes`.

    Returns
    -------
    pathlib.Path
        *path*, complete. On any error nothing is left at *path* (an existing
        file there is kept).

    Raises
    ------
    ValueError, OSError
        As in :func:`fetch_bytes`.
    """
    timeout, max_bytes = _check(url, timeout, max_bytes)
    target = Path(path)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".part", dir=target.parent
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            _stream(url, timeout, max_bytes, fh)
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    return target
