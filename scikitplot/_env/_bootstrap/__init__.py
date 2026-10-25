# scikitplot/_env/_bootstrap/__init__.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Bootstrap: install the builder's tools (micromamba, uv) and shell start-up lines.

Notes
-----
**User notes.** Nothing here runs without an approval (``bootstrap`` for a
tool, ``shell_init`` for a start-up file). Tools go to ``<home>/tools/bin``
(``SKPLT_ENV_HOME``, default ``~/.scikitplot/env``), never to a system
directory, and are found there by every later plan.

* **micromamba** is the conda-forge package ``micromamba 2.9.0`` for the
  host's platform, checked against a SHA-256 written in this module before
  anything is extracted. The archive is kept in ``<home>/tools/cache``: copy
  it there (from any machine) and the bootstrap also works offline.
* **uv** is installed with the running Python's pip (``uv==0.11.32``, the
  version the tests ran with) with ``--require-hashes``: it follows pip's
  index configuration, but only a wheel whose SHA-256 is pinned here installs.

**Developer notes.** To move the pin: download the five archives, record
their SHA-256 in :py:data:`MICROMAMBA` and run the bootstrap tests; a wrong
hash fails before extraction, so a bad pin cannot install anything.
"""

from __future__ import annotations

import hashlib
import http
import logging
import os
import stat
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from types import MappingProxyType
from typing import Callable

from .._core._errors import ExecutionError

__all__ = [
    "MICROMAMBA",
    "MICROMAMBA_VERSION",
    "UV_REQUIREMENT",
    "UV_WHEEL_SHA256",
    "install_micromamba",
    "micromamba_archive",
]

logger = logging.getLogger(__name__)

#: micromamba version installed by the bootstrap.
MICROMAMBA_VERSION = "2.9.0"

#: Requirement used to bootstrap uv (the version the tests ran with).
UV_REQUIREMENT = "uv==0.11.32"

#: SHA-256 of every ``uv 0.11.32`` wheel on PyPI (all platforms); pip installs
#: with ``--require-hashes --only-binary=:all:``, so no other file is accepted.
UV_WHEEL_SHA256 = (
    "26d61d1b640d2dcfd7c3b64e75ada14c8eb477accba57f414c1a3759d6ab2253",
    "c82954b1bd507e767b1d996c2865ea3eb7d464a479ec0e4d60f0b8e2e07dee45",
    "2a56d1abf16337350e878b2c869b256b8448ecc4a722b98f8d41e1b3680a587b",
    "125c142363d0842c8506a057da56bae182e2aa3957344f57dd9ef20ea10f06b0",
    "96625d832fe5b94ed0a029764cef0879ff47e3a525aec5c99d360a959e2f4aea",
    "07ffc8322ce88b29dc0a905c102ea7d32c9021ee0353b641674e350c65fa930a",
    "8e36fe8ff79ab7d13e9ad45d24bb9a52c1024caa6897cf62933bb3f760643d13",
    "19fb9f2dba53851c0ee6e7e5582bd355fcaa9b1387b5d688f5bb9efc543dc605",
    "f6f030a91a94655af11e41dafe1b41e1a90e1f2641bf13ff7ed7d4f5fb4f031d",
    "3da76cd4e2697de30928b8a8524bd39183ac1e08cb7e72833807c022b7cba6c4",
    "be0799f1ad70c755d10de5aaf46af94199d4f16a992f90278f2662350cd3f4fe",
    "f35b3b8b65ba4579f8b23645a19394f7e5f90b20b07376ecff18b3bb656a81ae",
    "cd085addb35f074561d2c5659a088f5f10361502bd3ad4c8f5105f5a4b9d2817",
    "43360834111ae917808a70b36f4fb0e53de8e0be422e4cd4445876ffae8decc2",
    "77f4356548ee8dc47efae154efd4e930c65570e7d4971c57bdef592f6eefb39c",
    "bb1ae8189e315499a77f3cc27cac5985ab1d8cb79baa82e368b6c1e574a7d9cf",
    "855632ee1d2a8491986efa54c562b806cd32477761889a1e13624b519d98a45e",
    "bf988bf510772785eac1edfc7cdedca074e8936b45e755b58f7f3e1e9ca86424",
)

_BASE = "https://conda.anaconda.org/conda-forge"

#: ``subdir -> (sha256, member)`` of the conda-forge ``micromamba`` archives.
MICROMAMBA = MappingProxyType(
    {
        "linux-64": (
            "8761c382127e6363bd9e0a2451aa3ef90d071a79133f736e2f759a3bf13040dd",
            "bin/micromamba",
        ),
        "linux-aarch64": (
            "e705ffeed90ce0659eb546e4b1e1028c9eaf0bc9cc854867b19ac5ce0ba5852f",
            "bin/micromamba",
        ),
        "osx-64": (
            "0426ecdc41636d369f57b8fe6acbf4385a69eca45b56d9ee7d3a840a9965d44f",
            "bin/micromamba",
        ),
        "osx-arm64": (
            "500f5074feb8d02c4296ef9921c3650ed2874171805a9fbb8fbb53896433646b",
            "bin/micromamba",
        ),
        "win-64": (
            "97a336f4ab794bd96a6a4da5e6ed63e75a1d31830414a182419b23d3b36f3fe0",
            "Library/bin/micromamba.exe",
        ),
    }
)

_MAX_BYTES = 64 * 1024 * 1024


def micromamba_archive(subdir: str) -> dict[str, str]:
    """
    Return the pinned micromamba archive of a platform.

    Parameters
    ----------
    subdir : str
        Conda platform (``"linux-64"`` ...).

    Returns
    -------
    dict
        ``url``, ``sha256``, ``member`` (file inside the archive) and
        ``filename``.

    Raises
    ------
    KeyError
        When no archive is pinned for ``subdir``.

    Examples
    --------
    >>> from scikitplot._env._bootstrap import micromamba_archive
    >>> micromamba_archive("linux-64")["filename"]
    'micromamba-2.9.0-0.tar.bz2'
    """
    sha256, member = MICROMAMBA[subdir]
    filename = f"micromamba-{MICROMAMBA_VERSION}-0.tar.bz2"
    return {
        "url": f"{_BASE}/{subdir}/{filename}",
        "sha256": sha256,
        "member": member,
        "filename": filename,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(
    url: str, dest: Path, *, timeout: float, opener: Callable[..., object]
) -> None:
    if not url.startswith("https://"):
        raise ExecutionError(
            f"refusing to download {url!r}: not https", step="bootstrap"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".download-")
    try:
        with os.fdopen(fd, "wb") as out, opener(url, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if status != http.HTTPStatus.OK:
                raise ExecutionError(
                    f"GET {url} returned HTTP {status}", step="bootstrap"
                )
            total = 0
            for block in iter(lambda: response.read(1 << 20), b""):
                total += len(block)
                if total > _MAX_BYTES:
                    raise ExecutionError(
                        f"{url} is larger than {_MAX_BYTES} bytes", step="bootstrap"
                    )
                out.write(block)
        os.replace(tmp, dest)
    except OSError as exc:
        raise ExecutionError(
            f"download of {url} failed: {exc}",
            step="bootstrap",
            hint=f"check the network or copy the archive into {dest.parent}",
        ) from exc
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def install_micromamba(
    tools_dir: str | os.PathLike,
    subdir: str,
    *,
    offline: bool = False,
    timeout: float = 60.0,
    opener: Callable[..., object] | None = None,
) -> Path:
    """
    Install the pinned micromamba into ``<tools_dir>/bin``.

    Parameters
    ----------
    tools_dir : path-like
        Tools directory (``<home>/tools``).
    subdir : str
        Conda platform of the host.
    offline : bool, default=False
        Use only ``<tools_dir>/cache``; never download.
    timeout : float, default=60.0
        Network timeout in seconds.
    opener : callable or None, default=None
        ``opener(url, timeout=...)`` returning a response context manager;
        :py:func:`urllib.request.urlopen` when None (tests pass a fake).

    Returns
    -------
    pathlib.Path
        The installed executable.

    Raises
    ------
    ExecutionError
        No pinned archive for ``subdir``, offline without a cached archive,
        a failed download, a SHA-256 mismatch, or an archive without the
        expected file.

    Notes
    -----
    Idempotent: a cached archive with the right hash is reused, and the
    executable is replaced atomically, so a concurrent reader never sees a
    half-written file.
    """
    if subdir not in MICROMAMBA:
        raise ExecutionError(
            f"no micromamba archive is pinned for {subdir!r}",
            step="bootstrap",
            hint=f"pinned platforms: {sorted(MICROMAMBA)}",
        )
    info = micromamba_archive(subdir)
    tools = Path(tools_dir)
    archive = tools / "cache" / f"{subdir}-{info['filename']}"
    if archive.is_file() and _sha256(archive) != info["sha256"]:
        logger.warning("cached %s has the wrong SHA-256; downloading it again", archive)
        archive.unlink()
    if not archive.is_file():
        if offline:
            raise ExecutionError(
                f"offline and {archive} is not cached",
                step="bootstrap",
                hint=f"copy {info['url']} to {archive}",
            )
        logger.info("downloading %s", info["url"])
        _download(
            info["url"],
            archive,
            timeout=timeout,
            opener=(
                opener  # callable
                or urllib.request.urlopen  # noqa: S310 (https only, see _download)
            ),
        )
    actual = _sha256(archive)
    if actual != info["sha256"]:
        archive.unlink()
        raise ExecutionError(
            f"SHA-256 of {info['url']} is {actual}, expected {info['sha256']}",
            step="bootstrap",
            hint="the download was altered or truncated; retry",
        )
    name = Path(info["member"]).name
    dest = tools / "bin" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".micromamba-")
    try:
        with os.fdopen(fd, "wb") as out, tarfile.open(archive, mode="r:bz2") as tar:
            member = tar.getmember(info["member"])
            if not member.isfile():
                raise ExecutionError(
                    f"{info['member']} in {archive} is not a file", step="bootstrap"
                )
            source = tar.extractfile(member)
            with source:
                for block in iter(lambda: source.read(1 << 20), b""):
                    out.write(block)
        # an executable: owner rwx, others r-x (as any installed binary)
        mode = stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
        os.chmod(tmp, mode)  # noqa: S103
        os.replace(tmp, dest)
    except (tarfile.TarError, KeyError) as exc:
        raise ExecutionError(
            f"{archive} has no usable {info['member']}: {exc}", step="bootstrap"
        ) from exc
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    logger.info("installed micromamba %s at %s", MICROMAMBA_VERSION, dest)
    return dest
