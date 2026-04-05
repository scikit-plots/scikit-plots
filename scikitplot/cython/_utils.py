"""Small utilities."""

from __future__ import annotations

__all__ = [
    "sanitize",
]


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
        Sanitized module-like identifier.  The returned string is guaranteed to
        be a non-empty, valid Python identifier consisting only of ASCII
        alphanumerics and underscores.

    Raises
    ------
    TypeError
        If ``name`` is not a ``str``.

    Notes
    -----
    - Non-alphanumeric characters (including ``/``, ``-``, ``.``, spaces) are
      replaced with underscores.
    - If the first character of the result would be a digit, a leading
      underscore is prepended so that the output is always a valid identifier.
    - An empty input string returns ``"_"`` (the minimal valid identifier).

    Examples
    --------
    >>> sanitize("hello-world")
    'hello_world'
    >>> sanitize("123abc")
    '_123abc'
    >>> sanitize("")
    '_'
    >>> sanitize("a/b/c")
    'a_b_c'
    """
    if not isinstance(name, str):
        raise TypeError(f"sanitize() requires a str, got {type(name).__name__!r}")
    # Empty input: return the minimal valid identifier.
    if not name:
        return "_"
    out: list[str] = []
    # Prepend underscore when the first character is a digit so that the
    # result is always a valid Python identifier start character.
    if name[0].isdigit():
        out.append("_")
    for ch in name:
        out.append(ch if ch.isalnum() else "_")
    return "".join(out)
