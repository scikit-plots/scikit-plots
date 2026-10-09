.. currentmodule:: scikitplot._externals._sphinx_ext._sphinx_jinja_render

.. _externals-sphinx-ext-sphinx-jinja-render-index:

======================================================================
Sphinx Jinja RST renderer
======================================================================

``_sphinx_jinja_render`` combines two documentation-build helpers:

* render ``*.rst.template`` files with Jinja2 before Sphinx reads the sources;
* add a generated JupyterLite ``repl_url`` to each HTML page template context.

Enable it
----------------------------------------------------------------------

::

   extensions += [
       "scikitplot._externals._sphinx_ext._sphinx_jinja_render",
   ]

   index_template_kwargs = {
       "development_link": "devel/index",
   }

The extension connects to ``builder-inited`` and ``html-page-context``.  The
only Sphinx config value it currently registers is ``index_template_kwargs``.

RST templates
----------------------------------------------------------------------

A source named ``index.rst.template`` is rendered to ``index.rst`` beside the
template.  Treat the template as the authoring source and the generated RST as
derived output::

   index.rst.template   # edit this
          |
          v
   index.rst            # generated; may be overwritten

Jinja rendering uses ``StrictUndefined`` for a single template, so an undefined
variable is an error at the rendering layer rather than silently becoming an
empty string.

The public helper ``render_rst_templates`` processes templates in sorted order
for reproducible output.  Programmatic callers can select recursive discovery
and strict failure behavior.

REPL URL
----------------------------------------------------------------------

For HTML pages the extension places ``repl_url`` into the page Jinja context.
Theme/templates can then render ``{{ repl_url }}`` without rebuilding the URL
logic themselves.

Reliability note
----------------------------------------------------------------------

Do not hand-edit a generated ``.rst`` file and assume the edit will survive the
next build.  The current Sphinx event hook invokes the renderer in its default
non-strict mode; maintainers should therefore keep clean-build verification in
the documentation release gate so a previous generated file cannot mask a
failed template render.  A repository follow-up tracks making this path fail
closed against stale output.
