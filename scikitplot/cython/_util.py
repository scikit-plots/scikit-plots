"""Small utilities."""

from __future__ import annotations


def sanitize(name: str) -> str:
    """
    Convert an arbitrary string into a valid Python module name.

    Parameters
    ----------
    name : str
        Input string (path-like strings allowed).

    Returns
    -------
    str
        Sanitized module-like identifier.

    Notes
    -----
    - Non-alphanumeric characters are replaced with underscores.
    - If the name begins with a numeral, a leading underscore is added.
    """
    if not name:
        return "_"
    out: list[str] = []
    if name[0].isdigit():
        out.append("_")
    for ch in name:
        out.append(ch if ch.isalnum() else "_")
    return "".join(out)
