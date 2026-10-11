# Family / ownership boundaries

## `scikitplot.cexternals._editdistance`

Owned by the C-external subsystem. Levenshtein may call its public `distance`
alias but does not own its Cython/C++ build, algorithms or tests.

## RapidFuzz

Optional MIT-licensed accelerator. Never imported by merely importing the
Levenshtein facade.

## external `Levenshtein`

Optional GPL-2.0-or-later backend. Supported only through an explicit backend
request; never part of automatic selection.

## `scikitplot.corpus`

Optional consumer. The adapter dependency points from Levenshtein to Corpus
only at scorer invocation time. Corpus must not become an import requirement of
the facade.
