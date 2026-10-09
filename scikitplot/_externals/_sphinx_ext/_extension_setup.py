"""
One import root per Sphinx application for the private extension stack.

The stack is importable under two roots that hold the same code:

- ``scikitplot._externals._sphinx_ext`` -- the copy installed with scikit-plots;
- ``_sphinx_ext`` -- a copy placed on ``sys.path`` by a documentation checkout
  (for example ``docs/source/_sphinx_ext``) for quick local testing.

Loading children from both roots in one application gives two module objects
for one extension: two directive registrations, two config values, two asset
sets. :func:`check_namespace` refuses that before an extension registers
anything.

Notes
-----
**User notes.** Pick one root and use it for every stack entry in
``extensions`` (and for every dotted path you hand to another tool, such as
sphinx-gallery's ``reset_modules``). Other packages that happen to be called
``_sphinx_ext`` -- the documentation's own ``_sphinx_ext.mpl_ext``,
``_sphinx_ext.sklearn_ext`` and ``_sphinx_ext.skplt_ext`` helpers -- are not
part of the stack and are ignored.

**Developer notes.** A dotted name counts as a stack entry only when the
component right after a ``_sphinx_ext`` component is a name the stack declares
(:data:`STACK_MEMBERS`) or used to declare (:data:`RETIRED_MEMBERS`). Matching
the bare ``_sphinx_ext.`` prefix instead treats the documentation's unrelated
helper package as a second root and refuses every build that also uses the
installed stack. The member list is the package's declared registry, not a
directory listing, so the answer does not depend on which optional packages a
given copy happens to contain.
"""

from __future__ import annotations

from sphinx.errors import ExtensionError

from . import _CORE_PRIVATE_SUBMODULES, _OPTIONAL_PRIVATE_SUBMODULES

#: Name of the package component that marks a stack root.
NAMESPACE_COMPONENT = "_sphinx_ext"

#: Child packages the stack declares, present on disk or not.
STACK_MEMBERS: frozenset[str] = _CORE_PRIVATE_SUBMODULES | _OPTIONAL_PRIVATE_SUBMODULES

#: Former child packages, mapped to the message that names their replacement.
RETIRED_MEMBERS: dict[str, str] = {
    "youtube_catalog": (
        "youtube_catalog was renamed to _sphinx_youtube_gallery. "
        "Update this extension entry, remove the obsolete youtube_catalog "
        "and collection directories after installing the replacement, "
        "and rebuild with -E."
    ),
    "collection": (
        "collection was renamed to _sphinx_collection. Update this "
        "private extension/import entry, remove the obsolete collection "
        "directory after installing the replacement, and rebuild with -E."
    ),
    "_pydata_sphinx_theme": (
        "The private _pydata_sphinx_theme extension bucket was split by "
        "responsibility: use _sphinx_gallery_grid for gallery-grid and "
        "_pydata_component_list for component-list, remove the obsolete "
        "_pydata_sphinx_theme directory, and rebuild with -E."
    ),
}

#: Attribute on the Sphinx application that records the accepted root.
ROOT_ATTRIBUTE = "_scikitplot_sphinx_extension_root"


def split_stack_name(name):
    """
    Split a dotted module name into its stack root and stack member.

    Parameters
    ----------
    name : str
        Dotted module name, for example an entry of ``extensions``.

    Returns
    -------
    tuple of (str, str) or None
        ``(root, member)`` when *name* addresses a declared or retired stack
        member, for example ``("_sphinx_ext", "_sphinx_collection")`` for
        ``"_sphinx_ext._sphinx_collection.setup"``; ``None`` otherwise.

    See Also
    --------
    check_namespace : Uses this split to find the roots in use.

    Notes
    -----
    **Developer notes.** The first ``_sphinx_ext`` component followed by a
    known member wins, so ``a._sphinx_ext.mpl_ext`` is not a stack name while
    ``a._sphinx_ext._sphinx_collection`` is, with root ``a._sphinx_ext``.

    Examples
    --------
    >>> split_stack_name("scikitplot._externals._sphinx_ext._sphinx_gallery_grid")
    ('scikitplot._externals._sphinx_ext', '_sphinx_gallery_grid')
    >>> split_stack_name("_sphinx_ext.mpl_ext.redirect_from") is None
    True
    """
    if not isinstance(name, str):
        return None
    parts = name.split(".")
    for index in range(len(parts) - 1):
        member = parts[index + 1]
        if parts[index] == NAMESPACE_COMPONENT and (
            member in STACK_MEMBERS or member in RETIRED_MEMBERS
        ):
            return ".".join(parts[: index + 1]), member
    return None


def check_namespace(app, root):
    """
    Refuse a second stack root, or a retired member, before registration.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The application being set up. Only ``app.config.extensions``,
        ``app.extensions`` and the :data:`ROOT_ATTRIBUTE` attribute are used.
    root : str
        Root of the calling extension, for example
        ``"scikitplot._externals._sphinx_ext"`` or ``"_sphinx_ext"``.

    Returns
    -------
    None
        On success *root* is recorded on *app* as :data:`ROOT_ATTRIBUTE`.

    Raises
    ------
    sphinx.errors.ExtensionError
        If a configured or loaded entry names a retired member (the message
        names its replacement), or if the entries, the recorded root and
        *root* do not all share one root (the message names every root found).
        A refused call leaves the recorded root unchanged.

    See Also
    --------
    split_stack_name : Decides which names belong to the stack.

    Notes
    -----
    **User notes.** Calling it again with the same root is harmless.

    Examples
    --------
    >>> import types
    >>> app = types.SimpleNamespace(
    ...     config=types.SimpleNamespace(extensions=["_sphinx_ext.mpl_ext.github"]),
    ...     extensions={},
    ... )
    >>> check_namespace(app, "scikitplot._externals._sphinx_ext")
    >>> app._scikitplot_sphinx_extension_root
    'scikitplot._externals._sphinx_ext'
    """
    names = list(app.config.extensions) + list(app.extensions)
    roots = {root}
    for name in names:
        split = split_stack_name(name)
        if split is None:
            continue
        found_root, member = split
        if member in RETIRED_MEMBERS:
            raise ExtensionError(RETIRED_MEMBERS[member])
        roots.add(found_root)
    previous = getattr(app, ROOT_ATTRIBUTE, None)
    if previous:
        roots.add(previous)
    if len(roots) != 1:
        raise ExtensionError(
            "Mixed scikit-plots extension namespaces: "
            + ", ".join(sorted(roots))
            + ". Use one namespace consistently in extensions and setup_extension()."
        )
    setattr(app, ROOT_ATTRIBUTE, root)
