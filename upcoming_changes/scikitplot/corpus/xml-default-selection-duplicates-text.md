---
title: "XMLReader repeats nested text once per ancestor element by default (CX-46)"
status: open
kind: "bug"
area: "scikitplot/corpus"
discovered_during: "Real-data snapshot and extraction benchmark of a private mixed research archive (2026-11-23)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "fix"
towncrier_fragment: ""
---

# XMLReader repeats nested text once per ancestor element by default (CX-46)

## Summary

With its defaults (`block_xpath=".//*"`, `text_xpath=None`), `XMLReader`
selects every descendant element and takes each one's `itertext()`, which
includes the text of all its descendants. Every text node is therefore
emitted once for each ancestor between it and the root, plus once on its own.

## Why it matters

Any XML that is not ALTO and nests its markup (TEI, JATS, DocBook, generic
`<section><p>...</p></section>`) gives duplicate chunks. Duplicates inflate
term frequencies for keyword scoring, repeat embeddings, waste storage and
return the same passage several times in a search. `.xml` files route to
`XMLReader` by default, so this is the default path for TEI too.

## Current evidence

Minimal reproducer (`main` at 13b457d, CPython 3.13, lxml installed):

```python
# nested.xml: <doc><sec><p>alpha <b>beta</b> gamma</p></sec></doc>
[c["text"] for c in DocumentReader.create("nested.xml").get_raw_chunks()]
# ['alpha beta gamma', 'alpha beta gamma', 'beta']
```

Real data: two TEI files with footnotes in the archive. Against the words of
the document read once (`itertext()` of the root), word precision of the
reader's output was 0.16 and 0.24: about 6x and 4x the document's words.
The header title alone came out 4 times (once each for the header, the file
description, the title statement and the title).

## Root cause / current understanding

Verified: `_xpath_elements(root, ".//*")` returns ancestors and descendants
alike, and `_element_text_content` joins `element.itertext()` (all
descendant text). The docstring documents "all elements with text"; it does
not say the text repeats.

## Expected behavior

With the defaults, every text node of the document appears in exactly one
chunk, in document order. Explicit `block_xpath` / `text_xpath` keep their
meaning.

## Affected paths and ownership

- `scikitplot/corpus/_readers/_xml.py` (`XMLReader.block_xpath` default,
  `get_raw_chunks`, `_element_text_content`).
- `TEIReader` sets its own XPath and is not affected; see the open decision
  "TEI routing" in the corpus backlog, which this note interacts with.
- Owner: `maintenances/corpus`.

## Constraints and non-goals

- Changes the default output of `XMLReader`: a maintainer decision.
- Do not change the ALTO path (`alto="auto"`) or explicit XPath behavior.
- Same result with and without lxml.

## Edge cases to cover

Mixed content (`text` and `tail`), inline markup inside a word
(`Elementar<hi>form</hi>`), empty elements, comments and processing
instructions, namespaced and default-namespace documents, a single flat
level (`<root><item/>...</root>`, output must not change), very deep
nesting (no quadratic text building), CDATA.

## Proposed direction

Options, recommended first:

(a) Default to block-level leaves: select elements that have direct text
(`text` or a child's `tail`) and whose text-bearing descendants are only
inline; emit each block once. Inline children are not selected.
(b) Keep `.//*` but take only an element's direct text (`text` plus
children's `tail`), so each text node is emitted once; inline words are
split from their sentence (worse for retrieval).
(c) Keep the behavior and document it; add a de-duplicating option.

## Verification / acceptance criteria

- The reproducer gives one chunk `'alpha beta gamma'` under (a).
- Over a TEI fixture, the multiset of words in the output equals that of
  the root's `itertext()` (precision and recall 1.0).
- A test fails on the current code (mutation check: restore `.//*` with
  `itertext()`).
- Reader snapshot before and after: only `.xml` files change.

## Documentation impact

`XMLReader` docstring (Notes, Examples), the corpus reader user guide page.

## Release-note promotion

Required: `scikitplot.corpus` / `fix`.
