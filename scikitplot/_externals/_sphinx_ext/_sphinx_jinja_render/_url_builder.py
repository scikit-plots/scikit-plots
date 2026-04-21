# scikitplot/_externals/_sphinx_ext/_sphinx_jinja_render/_url_builder.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
JupyterLite REPL URL builder.

Responsibility
--------------
Assemble a fully-qualified, percent-encoded URL that opens the hosted
JupyterLite REPL with a pre-loaded kernel and bootstrap code.

Notes
-----
Developer
    All string concatenation goes through :func:`urllib.parse.urlencode`
    so that special characters in *code* are always safely encoded.
    The function is pure: same inputs ↔ same output; no side effects.

User
    Pass the return value to a Sphinx ``html-page-context`` handler or
    embed it directly in a Jinja2 template as ``{{ repl_url }}``.
"""

from __future__ import annotations

from urllib.parse import quote, urlencode  # noqa: F401

from ._constants import (
    CODE_PARAM,
    DEFAULT_KERNEL_NAME,
    JUPYTERLITE_BASE_URL,
    KERNEL_PARAM,
)
from ._validators import (
    validate_kernel_name,
    validate_non_empty_string,
    validate_url_length,
)


def build_repl_url(
    code: str,
    kernel: str = DEFAULT_KERNEL_NAME,
    base_url: str = JUPYTERLITE_BASE_URL,
) -> str:
    """Build a JupyterLite REPL URL with embedded bootstrap code.

    Parameters
    ----------
    code : str
        Python source code to pre-load in the REPL.  Must be a
        non-empty string.
    kernel : str, optional
        Jupyter kernel identifier.  Must contain only ASCII letters,
        digits, hyphens, underscores and dots.  Defaults to
        :data:`~._constants.DEFAULT_KERNEL_NAME` (``"python"``).
    base_url : str, optional
        Base URL of the hosted JupyterLite instance.  Defaults to
        :data:`~._constants.JUPYTERLITE_BASE_URL`.

    Returns
    -------
    str
        Fully assembled, percent-encoded REPL URL.

    Raises
    ------
    TypeError
        If any argument is not a :class:`str`.
    ValueError
        If *code* or *base_url* is empty / whitespace-only, if *kernel*
        contains illegal characters, or if the assembled URL exceeds
        :data:`~._constants.MAX_URL_LENGTH`.

    See Also
    --------
    _validators.validate_url_length : URL length enforcement.
    _validators.validate_kernel_name : Kernel name enforcement.

    Notes
    -----
    Developer
        The query string is always appended with ``?`` regardless of
        whether *base_url* already contains a query component.  The
        current base URL constants do not include query strings; if they
        ever do, this function must be updated to merge parameters using
        :func:`urllib.parse.urlparse` / :func:`urllib.parse.parse_qs`.

    Examples
    --------
    >>> url = build_repl_url("import scikitplot")
    >>> url.startswith("https://")
    True
    >>> "kernel=python" in url
    True
    >>> "import+scikitplot" in url or "import%20scikitplot" in url
    True
    """
    validate_non_empty_string(code, "code")
    validate_kernel_name(kernel)
    validate_non_empty_string(base_url, "base_url")

    # https://jupyterlite.github.io/demo/repl/?toolbar=1&kernel=python&execute=1&code=import%20numpy%20as%20np
    query: str = urlencode({KERNEL_PARAM: kernel, CODE_PARAM: code})
    url: str = f"{base_url.rstrip('/')}?toolbar=1&execute=1&{query}"
    validate_url_length(url)
    return url
