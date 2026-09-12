# Corpus G10 — project integration checkpoint

Date: 2026-08-19

Status: **VERIFIED DROP-IN INTEGRATION / FULL SPHINX EXECUTION ENVIRONMENT-BLOCKED**

## Source authority

```text
implementation source: scikit-plots(20260818-204251).zip
SHA-256: 4f62b638647a1645ad6174b646c2aabb4b30bee4a28ac9afb12d25ee71ff4b27
renderer integration reference: current public scikit-plots main/docs Makefile and dev gallery
```

## Scope

G10 closes the repository-path integration layer for the canonical Corpus gallery.
It does not change Corpus runtime semantics.

The canonical gallery is intended to replace the active contents under:

```text
galleries/examples/corpus/
```

with seven canonical scripts plus the subsection `README.txt` and real supplied
assets. Historical Hamlet v1/v2 variants are not active pages.

## Project-specific renderer alignment

The public scikit-plots gallery renders Corpus as a subsection of the top-level
Examples page. The gallery README therefore uses the repository-compatible
subsection heading:

```rst
Corpus
------
```

and retains:

```rst
.. currentmodule:: scikitplot.corpus
```

for short public cross-reference roles.

The current public docs Makefile exposes `EXAMPLES_PATTERN` and routes it to
`sphinx_gallery_conf.filename_pattern`, so the intended focused renderer gate is:

```bash
cd docs
make html \
  EXAMPLES_PATTERN="corpus" \
  SPHINXOPTS="-T -W --keep-going -n"
```

The normal full docs target remains the final authority.

## Verification completed in this harness

```text
README subsection hierarchy             PASS
README currentmodule                     PASS
public Corpus class-role resolution      PASS
7 canonical scripts compile              PASS
7 canonical scripts execute              PASS
normal network opt-in                     disabled
normal Whisper opt-in                     disabled
maintenance tracker                       PASS
```

## Environment limitation

The uploaded implementation snapshot does not include the repository `docs/`
source tree, and this harness cannot install Sphinx/Sphinx-Gallery because
package-index DNS resolution is unavailable. Therefore the real project
`sphinx-build` remains **BLOCKED**, not PASS.

Public repository/docs inspection verifies that:

- `docs/Makefile` uses `docs/source` and supports `EXAMPLES_PATTERN`;
- the generated docs publish gallery output under `auto_examples`;
- the current published gallery renders `Corpus` as a subsection of `Examples`.

## Gallery reliability rule

```text
missing optional package/resource/native capability/network opt-in
    -> visible specific SKIP when truthful continuation is possible

invalid public API / security-policy failure / installed-backend regression
    -> FAIL visibly
```

## Next gate

Run the focused command above, then a normal project `make html`, in the real
documentation environment. Close the campaign only when generated Corpus pages,
index ordering, tags, thumbnails/downloads, cross-references, and warnings are
green.
