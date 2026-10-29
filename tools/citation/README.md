# Citation files

`citation_ref.py` renders the root `CITATION.bib` and `CITATION.cff` from the
templates in this folder and from facts read from the checkout:

| Fact | Source |
|---|---|
| public version (`0.5.dev0`, `0.4.0.post11`) and its kind | `scikitplot/__init__.py` `__version__` (PEP 440) |
| build version (`0.5.dev0+git.20261028.b656b70`) | the same format as `scikitplot/_build_utils/gitversion.py` (reused, not copied) |
| commit and date | `git log -1` (read-only), or `--commit` / `--date`, or none with `--no-git` |
| a release's own DOI | `VERSION_DOIS` in `citation_ref.py` (from `docs/source/project/citing.rst`) |
| title, authors, concept DOI, license | `PROJECT` in `citation_ref.py` |

```sh
python tools/citation/citation_ref.py render          # print both files; writes nothing
python tools/citation/citation_ref.py sync            # preview the update as a diff
python tools/citation/citation_ref.py sync --apply    # write both files (atomic)
python tools/citation/citation_ref.py check           # exit 1 when stale or hand-edited
python -m unittest tools/citation/tests/test_citation_ref.py
```

`make -C tools cite` runs `sync --apply`; `make -C tools cite-check` runs `check`.
Rendering needs `jinja2`.

**When to regenerate.** After changing `__version__` and when preparing a
release (then add the release's Zenodo DOI to `VERSION_DOIS` once Zenodo has
archived it). `check` re-renders from the provenance line recorded in
`CITATION.bib` (`% provenance: version=... commit=... date=...`), so it fails
when the version changed or a file was edited by hand, never merely because
newer commits exist. A committed file cannot name the commit that contains
it: the files say which commit they were *generated from*.

**Compatibility.** The first `@software` entry keeps the key
`The-scikit-plots-developers_2024`, so existing `\cite` keys work, and
`scikitplot.config._citation` (generated at build time from `CITATION.bib`)
keeps reading that first entry as `scikitplot.__bibtex__`.

`CITATION1.*.template` and `cache_zenodo_svg.py` are older drafts and are
not used by `citation_ref.py`.
