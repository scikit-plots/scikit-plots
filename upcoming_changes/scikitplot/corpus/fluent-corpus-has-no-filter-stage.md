---
title: "corpus: FluentCorpus has no filter stage, and a reader filter_ crashes"
status: open
kind: "api"
area: "scikitplot/corpus"
discovered_during: "Hamlet custom-components gallery example (PR 967), main 2026-11-24"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "enhancement"
towncrier_fragment: ""
---

# corpus: FluentCorpus has no filter stage, and a reader filter_ crashes

## Summary

`CorpusPipeline(filter_=...)` applies a quality policy (a `FilterBase`),
but a `FluentCorpus` plan cannot carry one: there is no `.filter()` stage,
and passing it through the reader stage, `.reader({"filter_": policy})`,
fails at `run()` with `TypeError: DocumentReader.create() got multiple
values for keyword argument 'filter_'`.

## Why it matters

Which documents enter a corpus is a policy decision that belongs in the
plan, next to the chunker and the enricher. Today a user who needs one
must drop the plan and build the pipeline by hand, and the reader-kwargs
route fails with an error that does not say why.

## Current evidence

- `dir(FluentCorpus)` lists no `filter` or `replace_filter`.
- `scikitplot/corpus/_runtime.py` builds `CorpusPipeline(...)` without
  `filter_`; `CorpusPipeline.run` passes its own `filter_` to
  `DocumentReader.create` and also `**reader_kwargs`, hence the clash.
- `galleries/examples/corpus/plot_corpus_hamlet_custom_components_script.py`
  step 6: the plan keeps 12 speeches, the manual pipeline with the same
  policy 11.

## Root cause / current understanding

The plan's stage list predates filters as a configurable component; the
runtime materializer never learned to pass one.

## Expected behavior

`FluentCorpus().filter(policy)` (and `replace_filter`) stores the filter
in the plan, `materialize()` passes it to `CorpusPipeline(filter_=...)`,
and a `filter_` in reader options is either applied or refused with a
message naming `.filter()`.

## Affected paths and ownership

- `scikitplot/corpus/_fluent.py` (or the module defining `FluentCorpus`)
- `scikitplot/corpus/_runtime.py` (`materialize`)
- `scikitplot/corpus/_pipeline.py` (`filter_` vs `reader_kwargs`)

## Constraints and non-goals

The plan fingerprint must include the filter's identity (a frozen
dataclass fingerprints stably). No change for plans without a filter.

## Edge cases to cover

- A filter that rejects every document (empty corpus, clear warning)
- `filter_` given both as a stage and in reader options
- `add()` after `run()` uses the same filter
- A filter that is not a `FilterBase`

## Proposed direction

Add the stage with the same shape as `enricher`; in `CorpusPipeline.run`,
let an explicit reader option override or refuse, never pass both.

## Verification / acceptance criteria

- The gallery example's plan keeps the same 11 speeches as the manual run
- A test for the duplicate-keyword case

## Documentation impact

Update step 6 of the Hamlet custom-components example to use `.filter()`.

## Release-note promotion

Required: `scikitplot.corpus` / `enhancement`, one bullet, when the change lands.
