# `scikitplot.annoy` maintenance

This directory contains developer-only state and verification for the Annoy
Python/Cython subsystem. Runtime source remains under `scikitplot/annoy/`.

The key architectural rule is that `annoy` and `cexternals/_annoy` are different
compiled owners. Read `FRESH_CHAT_HANDOFF.md` before changing either boundary.
