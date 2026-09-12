# R173T25 — The one-file footer stops looking like a different feature

Status: COMPLETE

Base: R173T24 artifact accent and CSS mutation, same package.

## The inconsistency

With two or more files the footer is a segmented control:

```
|            Download all N files            │ Download patch series |
```

With one file it was a bare button beneath a bordered file row — no border, no
span, no chrome — which reads as a different feature rather than the same one
with less in it. A reader who produces one file and then two should not have to
relearn the footer.

The solo control now spans the section and wears the group's chrome. Nothing is
segmented, because with one file there is nothing to bundle; the shape is the
same, the contents are honest about the count.

## Wording follows the count

`Download patch series` became `Download patch` when there is one file. A
series of one file is just a patch, and calling it a series makes a reader look
for the other files it supposedly contains.

The accessible names changed with it, in both directions:

- one file: `Download <path> as a git patch` — says *which* file;
- many: `Download all N tracked files as one git patch series` — says *how many*.

The old label, `Download every tracked file as one git patch series`, was
count-blind and wrong in the single-file case in the same way the visible text
was. It is asserted gone.

## Verification

- browser wrapper gate: **147/147** (140 assertions in the owning harness);
- architecture gates: **456/456**, two new mutants, both caught:
  `solo-patch-footer-loses-its-chrome` and `one-file-export-called-a-series`.

## Note on where this run has got to

Four artifact surfaces — snippet cards, presented-file rows, the multi-file
footer and now the single-file footer — share one card implementation, one
segmented-control builder, one overflow menu, one accent token, and one set of
count-aware labels. The harnesses assert single implementations of the group
and the menu, so a fifth surface reuses them rather than copying.
