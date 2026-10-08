# scikitplot/_externals/_sphinx_ext/_sphinx_jinja_render/_example_conf.py
#
# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

extensions = [
    # ... your other extensions ...
    "scikitplot._externals._sphinx_ext._sphinx_jinja_render",
]

# ---------------------------------------------------------------------------
# jinja render — context kwargs
# ---------------------------------------------------------------------------

# ``_is_devrelease`` is a name of *your* conf.py: whether the documented version
# is a development release (for example
# ``packaging.version.parse(release).is_devrelease``). When this excerpt is
# pasted below that definition the line keeps your value; on its own it falls
# back to ``False`` instead of raising NameError.
_is_devrelease = globals().get("_is_devrelease", False)

# If development build, link to local page in the top navbar; otherwise link to the
# development version; see https://github.com/scikit-learn/scikit-learn/pull/22550
development_link = (
    "devel/index"
    if _is_devrelease
    else "https://scikit-plots.github.io/dev/devel/index.html"
)
index_template_kwargs = {
    "development_link": development_link,
}
