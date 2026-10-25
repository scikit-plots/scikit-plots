# scikitplot/_env/_bootstrap/_shell.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Shell start-up lines: put the builder's tools on ``PATH`` and hook micromamba.

Notes
-----
**User notes.** ``scikitplot env shell-init`` prints the lines; with
``--write`` (approval ``shell_init``) it puts them in the start-up file
between two marker lines and replaces that block on the next run, so running
it twice changes nothing and removing the block undoes it. The micromamba
hook is what makes ``micromamba activate`` work in an interactive shell.

=============  ==========================================
Shell          Start-up file
=============  ==========================================
``bash``       ``~/.bashrc``
``zsh``        ``$ZDOTDIR/.zshrc`` (``~/.zshrc``)
``fish``       ``~/.config/fish/config.fish``
``powershell`` no default: pass the profile path (``$PROFILE``)
=============  ==========================================

**Developer notes.** Values are quoted for each shell; paths with a quote
character are refused instead of being escaped, since a start-up file that
breaks is worse than an error here.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Mapping

from .._core._errors import ConfigError

__all__ = ["BEGIN", "END", "SHELLS", "rc_file", "shell_block", "write_shell_block"]

BEGIN = "# >>> scikitplot env >>>"
END = "# <<< scikitplot env <<<"
SHELLS = ("bash", "zsh", "fish", "powershell")
_Q = "'"


def _quoted(value: str, quote: str) -> str:
    # Single quotes: nothing inside is expanded by any of the four shells.
    if quote in value or "\n" in value:
        raise ConfigError(
            f"path {value!r} contains {quote!r} or a newline",
            hint="use a tools directory without quotes or newlines",
        )
    return f"{quote}{value}{quote}"


def shell_block(
    shell: str,
    tools_dir: str,
    *,
    micromamba: str | None = None,
    root_prefix: str | None = None,
) -> str:
    """
    Return the start-up block for ``shell``.

    Parameters
    ----------
    shell : {"bash", "zsh", "fish", "powershell"}
        Shell.
    tools_dir : str
        Tools directory; its ``bin`` is put first on ``PATH``.
    micromamba : str or None, default=None
        micromamba executable to hook; no hook when None.
    root_prefix : str or None, default=None
        ``MAMBA_ROOT_PREFIX`` for the hook.

    Returns
    -------
    str
        The block, marker lines included, ending with a newline.

    Raises
    ------
    ConfigError
        For an unknown shell or a path that cannot be quoted.

    Examples
    --------
    >>> from scikitplot._env._bootstrap._shell import shell_block
    >>> print(shell_block("bash", "/opt/t"))
    # >>> scikitplot env >>>
    export PATH='/opt/t/bin':"$PATH"
    # <<< scikitplot env <<<
    <BLANKLINE>
    """
    if shell not in SHELLS:
        raise ConfigError(
            f"unknown shell {shell!r}", hint=f"choose one of {list(SHELLS)}"
        )
    bin_dir = str(Path(tools_dir) / "bin")
    lines = [BEGIN]
    if shell in ("bash", "zsh"):
        lines.append(f'export PATH={_quoted(bin_dir, _Q)}:"$PATH"')
        if root_prefix:
            lines.append(f"export MAMBA_ROOT_PREFIX={_quoted(root_prefix, _Q)}")
        if micromamba:
            lines.append(
                f'eval "$({_quoted(micromamba, _Q)} shell hook --shell {shell})"'
            )
    elif shell == "fish":
        lines.append(f"fish_add_path --prepend --path {_quoted(bin_dir, _Q)}")
        if root_prefix:
            lines.append(f"set -gx MAMBA_ROOT_PREFIX {_quoted(root_prefix, _Q)}")
        if micromamba:
            lines.append(f"{_quoted(micromamba, _Q)} shell hook --shell fish | source")
    else:
        lines.append(
            f"$env:PATH = {_quoted(bin_dir, _Q)} + [IO.Path]::PathSeparator + $env:PATH"
        )
        if root_prefix:
            lines.append(f"$env:MAMBA_ROOT_PREFIX = {_quoted(root_prefix, _Q)}")
        if micromamba:
            lines.append(
                f"(& {_quoted(micromamba, _Q)} shell hook --shell powershell)"
                " | Out-String | Invoke-Expression"
            )
    lines.append(END)
    return "\n".join(lines) + "\n"


def rc_file(
    shell: str, *, home: str | None = None, environ: Mapping[str, str] | None = None
) -> Path:
    """
    Return the default start-up file of ``shell``.

    Parameters
    ----------
    shell : {"bash", "zsh", "fish"}
        Shell (PowerShell has no default here).
    home : str or None, default=None
        Home directory; ``~`` when None.
    environ : mapping or None, default=None
        Environment (``ZDOTDIR``); :py:data:`os.environ` when None.

    Returns
    -------
    pathlib.Path
        The file.

    Raises
    ------
    ConfigError
        For ``powershell`` or an unknown shell.
    """
    environ = os.environ if environ is None else environ
    base = Path(home) if home else Path.home()
    if shell == "bash":
        return base / ".bashrc"
    if shell == "zsh":
        return Path(environ.get("ZDOTDIR") or base) / ".zshrc"
    if shell == "fish":
        return base / ".config" / "fish" / "config.fish"
    raise ConfigError(
        f"no default start-up file for {shell!r}",
        hint="pass the file explicitly (PowerShell: the value of $PROFILE)",
    )


def write_shell_block(path: str | os.PathLike, block: str) -> bool:
    """
    Put ``block`` into ``path``, replacing an earlier block.

    Parameters
    ----------
    path : path-like
        Start-up file (created when missing).
    block : str
        Output of :py:func:`shell_block`.

    Returns
    -------
    bool
        Whether the file changed.

    Raises
    ------
    ConfigError
        When the file has a begin marker without an end marker (it is left
        as it is).

    Notes
    -----
    The file is written to a temporary file next to it and renamed, so it
    is never half-written; its permission bits are kept. A symlink is
    followed: the file it points to is the one changed.
    """
    # Write through a symlink (a dotfiles-managed ~/.bashrc), never replace it.
    target = Path(os.path.realpath(path))
    try:
        text = target.read_text(encoding="utf-8") if target.exists() else ""
    except UnicodeDecodeError as exc:
        raise ConfigError(
            f"{target} is not UTF-8 text; it is left as it is",
            hint="add the block by hand (scikitplot env shell-init prints it)",
        ) from exc
    start = text.find(BEGIN)
    if start >= 0:
        end = text.find(END, start)
        if end < 0:
            raise ConfigError(
                f"{target} has {BEGIN!r} without {END!r}",
                hint="fix or delete the block by hand, then rerun",
            )
        stop = end + len(END)
        if text[stop : stop + 1] == "\n":
            stop += 1
        new = text[:start] + block + text[stop:]
    else:
        new = text + ("" if not text or text.endswith("\n") else "\n") + block
    if new == text:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = target.stat().st_mode & 0o7777 if target.exists() else 0o644
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(new)
        os.chmod(tmp, mode)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return True
