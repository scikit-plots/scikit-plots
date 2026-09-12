# R173T23 — Presented files use the snippet card's markup, not a lookalike

Status: COMPLETE

Base: R173T22 one artifact control, same package.

## The divergence

R173T22 gave both surfaces the same segmented *container*. Inside it they still
built different trees: the snippet card had `icon → info(name, type)`, while a
presented file had `icon → copy(name, meta, badge) → "Preview"`, styled by a
parallel set of rules.

Parallel trees kept in step by hand do not stay in step, and these already had
not: the presented file had grown a trailing "Preview" word the snippet card
does not need (the card *is* the affordance), a badge sitting after the
metadata rather than with the type, and a different truncation point for the
same filename.

The preview segment now uses `ai-md-artifact-card` with the snippet card's own
`ai-md-artifact-icon` / `-info` / `-name` / `-type` classes. The remaining rules
cover only what genuinely differs: the type line carries a short badge **and**
the live revision/state text the ledger refreshes at click time, either of which
may be empty at different moments, so neither may leave a gap alone.

The diff stat stays beside the filename, unchanged — it is what makes the row
readable without opening it.

## The footer became the wide version of a file row

```
|            Download all N files            │ Download patch series |
```

Same segmented control, same builder. A single-file section keeps just the
patch export, since the row above it already offers everything else.

## Verification

- browser wrapper gate: **147/147** (134 assertions in the owning harness);
- architecture gates: **450/450**, one new mutant caught:
  `presented-file-forks-its-own-card-markup`, which reverts the shared classes
  and is exactly the regression this checkpoint removes.

## Note

Three surfaces now share one card implementation, one segmented-control
builder, and one overflow menu. A fourth has to reuse them: the harness asserts
a single `ai-md-artifact-group` construction and a single menu keydown
implementation in the file.
