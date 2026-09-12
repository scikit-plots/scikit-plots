"""
Expose Sphinx Design presentation options through namespaced gallery options.

Validators come from the installed Sphinx Design version, avoiding a duplicated
option table. Values are also constrained to one generated-source line. URL-like
card options admit HTTP(S), mailto, and relative paths while rejecting active
schemes. The module registers no Sphinx extension and performs no I/O.
"""

from urllib.parse import urlsplit

from sphinx_design.grids import GridDirective, GridItemCardDirective


def _preserve_source(converter, name):
    """Wrap an upstream converter with source-line and URL-scheme checks."""

    def validate(argument):
        """Validate and retain source text for later nested directive output."""
        # Nested directive options must remain one source line.
        value = "" if argument is None else argument.strip()
        if "\n" in value or "\r" in value or "\x00" in value:
            raise ValueError("presentation options must be a single line")
        if name == "link" or (name.startswith("img-") and name != "img-alt"):
            scheme = urlsplit(value).scheme.lower()
            if scheme and scheme not in ("http", "https", "mailto"):
                raise ValueError("use an HTTP(S) URL or a relative document/asset path")
        converted = converter(value)
        # docutils calls flag converters with ``argument is None``. Preserve
        # that semantic value so nested directive generation can emit
        # canonical ``:flag:`` syntax instead of depending on a trailing blank.
        if argument is None and converted is None:
            return None
        return value

    return validate


GRID_SPEC = {
    "grid-" + key: _preserve_source(value, key)
    for key, value in GridDirective.option_spec.items()
}
CARD_SPEC = {
    "card-" + key: _preserve_source(value, key)
    for key, value in GridItemCardDirective.option_spec.items()
}


def forwarded(options, prefix):
    """Strip one namespace prefix from recognized parsed options."""
    spec = GRID_SPEC if prefix == "grid-" else CARD_SPEC
    return {key[len(prefix) :]: value for key, value in options.items() if key in spec}
