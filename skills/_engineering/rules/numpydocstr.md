# NumPy-style docstrings

Categories: NumPyDoc, Cross-references

Rules keep the global number they were given; a new rule takes the next
free number (see [`../SKILL.md`](../SKILL.md)). Also apply Rule 48 (a
docstring with a backslash is raw, `r"""`, and each backslash is written
once) and Rule 54 (every example is executed before it is published).

## How the docs read a docstring (read first)

- `docs/source/conf.py` loads `numpydoc` after `autodoc`, and
  `docs/source/_templates/numpydoc_docstring.rst` renders exactly these
  parts, in this order: index, summary, extended summary, **Parameters,
  Returns, Yields, Receives, Other Parameters, Attributes, Raises, Warns,
  Warnings, See Also, Notes, References, Examples, Methods**. Summary and
  extended summary have no heading; every other part is a heading spelled
  as above with an underline of the same length.
- A heading numpydoc does not know is not rendered: numpydoc 1.11 warns
  `Unknown section ...` and its text is missing from the page (measured:
  a docstring with `Developer Notes` parses to summary, Notes and Examples
  only).
- `default_role = "literal"`: single backticks are code, never a link. A
  link needs a role (`:py:func:`, `:py:class:`, ...).
- In a raw docstring a backslash is written once: `sanitize_log_message`
  (`scikitplot/logging/_logging.py`) writes `` \\b `` inside `r"""` and its
  page shows two backslashes (Rule 48).
- `python -m numpydoc lint <files>` checks docstrings without building the
  docs; it writes to stderr. Survey of 2026-10-23 (702 source files outside
  `_externals`, tests and templates): GL06 unknown section 64, GL07 wrong
  order 233, PR01 parameter not documented 1 833, PR02 documented parameter
  that does not exist 397, PR03 wrong order 8, PR04 no type 382, PR10
  missing space before the colon 178, RT01 no Returns 1 931, YD01 no
  Yields 21.

### Rule 147: NumPyDoc - only the template's sections, under their exact titles
- **When:** writing or changing any docstring (module, class, function, method), whoever writes it.
- **Then:** use only the headings listed above, spelled exactly (`See Also`, not `See also`; `Examples`, not `Example`). Everything else a reader needs (usage, behaviour, architecture, security, compatibility, the rule a function enforces) goes into `Notes`, introduced by a bold lead-in as the package does: `**User notes.**` for callers, `**Developer notes.**` for maintainers. Sphinx directives (`.. note::`, `.. versionadded::`) go inside a section, never as a section.
- **Verified by:** `python -m numpydoc lint <files> 2>&1 | grep -E ': GL06 '` prints nothing for the changed files.
- **Added:** 2026-10-23. Root cause: 64 unknown sections in source docstrings, all silently missing from the built pages: `See also` (10), `Example`, `Architecture`, `Usage`, `The rule`, `Developer note`, `Behavior`, `Security`, `Scope and limitations`, ...

### Rule 148: NumPyDoc - sections in the template's order
- **When:** a docstring has more than one section.
- **Then:** keep the order of the template: Parameters, Returns, Yields, Receives, Other Parameters, Attributes, Raises, Warns, Warnings, See Also, Notes, References, Examples, Methods. Omit a section that would be empty; do not add sections mechanically.
- **Verified by:** `python -m numpydoc lint <files> 2>&1 | grep -E ': GL07 '` prints nothing for the changed files.
- **Added:** 2026-10-23. Root cause: 233 docstrings with sections out of order (survey of 2026-10-23); numpydoc renders them in its own order, so a `Notes` written to follow `Examples` is shown before them and a reference such as "the example below" points the wrong way.

### Rule 149: NumPyDoc - the documented parameters, returns and yields are the code's, with types
- **When:** writing or changing a callable, or its signature.
- **Then:** `Parameters` lists every parameter of the signature, in order, each as `name : type` (a space on both sides of the colon) with its default (`, default=None` or `, optional`), and no parameter the signature does not have; keyword-only parameters are listed too, `*args`/`**kwargs` as `*args` / `**kwargs`. A callable that returns a value has `Returns` (type on its own line, then what it is); a generator has `Yields`; a class documents its public attributes under `Attributes`. Types are written as NumPy does: `int`, `str or None`, `array-like of shape (n_samples, n_classes)`, `{'auto', 'manual'}`, a class as its importable name.
- **Verified by:** `python -m numpydoc lint <files> 2>&1 | grep -E ': (PR01|PR02|PR03|PR04|PR10|RT01|YD01) '` prints nothing for the docstrings added or changed (existing ones are fixed when their callable is touched).
- **Added:** 2026-10-23. Root cause: the survey of 2026-10-23: 1 833 undocumented parameters, 397 documented parameters that no longer exist (a rename left the docstring behind), 382 without a type, 178 written `name: type`, 1 931 callables that return a value without `Returns`.

### Rule 150: Cross-references - a role on every link, `~.name` for the package's own objects
- **When:** a docstring refers to another object.
- **Then:** use the role of the target (`:py:mod:`, `:py:class:`, `:py:func:`, `:py:meth:`, `:py:attr:`, `:py:obj:` when unsure of the kind). For an object of the package, write the relative form with a tilde, `` :py:obj:`~.logger` ``, `` :py:class:`~.Index` ``: the leading dot looks the name up from the current module and then as a suffix of every documented name, and the tilde shows only its last part, so the reference survives a move of the subtree. When the suffix is not unique (several documented objects end in `.load`), qualify it with its module (`` :py:func:`~.module.name` ``) until it is. For the standard library and other projects, write the full name: `` :py:class:`logging.Logger` ``, `` :py:func:`numpy.asarray` ``. Single backticks are code (`default_role = "literal"`), never a link.
- **Verified by:** a docs build with nitpicky warnings (`sphinx-build -n`) of the changed pages reports no `reference target not found` and no `more than one target found for cross-reference` for the changed docstrings.
- **Added:** 2026-10-23. Root cause: a hard-coded `scikitplot.x.y` reference breaks when the subtree moves, exactly as a hard-coded import does (Rule 142); a dot reference with an ambiguous suffix resolves to an arbitrary match with only a warning, so it must be checked.

### Rule 151: NumPyDoc - a docstring is checked against the code it describes, also when an assistant wrote it
- **When:** a docstring is written or rewritten, in particular by an AI assistant or from a template.
- **Then:** read the body and confirm each claim: the summary says what the code does (one line, imperative or descriptive, no repetition of the name); every exception under `Raises` is raised by the body or a callee it documents; defaults and types match the signature; `Notes` states what a caller or maintainer must know and the summary does not say (limits, side effects, thread safety, what is not done); examples are executed (Rule 54).
- **Verified by:** Rules 147-149's lint is clean for the docstring, `python -m doctest -v` or the package's doctest run executes its examples, and the review of the increment lists the docstrings checked against their code.
- **Added:** 2026-10-23. Root cause: generated or copied docstrings describe what code of that name usually does, not what this code does; the 397 documented parameters that do not exist and the `Notes`-only facts lost under unknown headings (Rules 147, 149) are that drift, and nothing but reading the body catches a wrong `Raises` or summary.

## Checklist

- [ ] Only the template's headings, exact spelling; extra prose in `Notes` with **User notes.** / **Developer notes.** (Rule 147).
- [ ] Sections in template order, none empty (Rule 148).
- [ ] Parameters match the signature with `name : type, default=...`; Returns / Yields / Attributes present where they apply (Rule 149).
- [ ] Links have roles; the package's own objects as `~.name`, external ones fully qualified (Rule 150).
- [ ] Backslashes: raw docstring, written once (Rule 48).
- [ ] Claims checked against the body; examples executed (Rules 151, 54).
- [ ] `python -m numpydoc lint <files> 2>&1 | grep -E ': (GL06|GL07|PR01|PR02|PR03|PR04|PR10|RT01|YD01) '` prints nothing for what changed.
