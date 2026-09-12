"""Application-local validation for the two supported installation layouts."""

from sphinx.errors import ExtensionError


def check_namespace(app, root):
    """Reject mixed import roots before registering this extension's objects."""
    names = list(app.config.extensions) + list(app.extensions)
    for name in names:
        if name.endswith("_sphinx_ext.youtube_catalog"):
            raise ExtensionError(
                "youtube_catalog was renamed to _sphinx_youtube_gallery. "
                "Update this extension entry, remove the obsolete youtube_catalog "
                "and collection directories after installing the replacement, "
                "and rebuild with -E."
            )
        if name.endswith("_sphinx_ext.collection"):
            raise ExtensionError(
                "collection was renamed to _sphinx_collection. Update this "
                "private extension/import entry, remove the obsolete collection "
                "directory after installing the replacement, and rebuild with -E."
            )
        if "._pydata_sphinx_theme" in name or name.endswith("_pydata_sphinx_theme"):
            raise ExtensionError(
                "The private _pydata_sphinx_theme extension bucket was split by "
                "responsibility: use _sphinx_gallery_grid for gallery-grid and "
                "_pydata_component_list for component-list, remove the obsolete "
                "_pydata_sphinx_theme directory, and rebuild with -E."
            )
    roots = {root}
    for name in names:
        if "_sphinx_ext." in name:
            roots.add(name.split("_sphinx_ext.", 1)[0] + "_sphinx_ext")
    previous = getattr(app, "_scikitplot_sphinx_extension_root", None)
    if previous:
        roots.add(previous)
    if len(roots) != 1:
        raise ExtensionError(
            "Mixed scikit-plots extension namespaces: "
            + ", ".join(sorted(roots))
            + ". Use one namespace consistently in extensions and setup_extension()."
        )
    app._scikitplot_sphinx_extension_root = root
