- :meth:`~scikitplot.lean.LeanIndex.summary` counts syntax extensions by
  kind (``syntax_extensions.by_kind``) and lists the flagged ones apart
  (``syntax_extensions.flagged``): Lean applies an extension without its
  name, so a macro that expands to ``sorry`` can enter a proof that never
  names it. The Markdown summary shows both.
