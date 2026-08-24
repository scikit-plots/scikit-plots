# `_sphinx_llm` — Sphinx machine-representation subsystem

`_sphinx_llm` is the planned scikit-plots owner for static, machine-consumable
Sphinx documentation.

The current tree contains two deliberately separated layers:

1. `sphinx_llm/` — the NVIDIA-derived baseline. A01 verified all 13 upstream
   source/test/license files byte-for-byte against pinned commit
   `2a971d7da6a5d7df81f7bff3612ee1822a060c17`; they are
   `UPSTREAM_PRESERVED`.
2. `core/`, `adapters/`, and `curation/` — downstream ownership shells whose
   production implementation remains intentionally deferred; `compat/` now owns
   the bounded A02 config-parity shim while later compatibility work remains
   checkpoint-gated.

A01 is **COMPLETE** via `PINNED_UPSTREAM_CI_EQUIVALENT`: local vendor/test/license bytes, the pinned `docs/source` fixture, `pyproject.toml`, `uv.lock`, and NVIDIA's test workflow are byte-verified against the same pinned commit, and the official Python-3.13/Sphinx-9 job is successful. The local exact-lock environment remains `ENVIRONMENT_BLOCKED`; that result is preserved as a reproduction gap and is not relabeled GREEN.

Target responsibilities:

- preserve NVIDIA `sphinx-llm` build/lifecycle strengths;
- produce canonical per-page Markdown from resolved Sphinx semantics;
- generate a curated `llms.txt` and optional bounded `llms-full.txt`;
- preserve rich extension/directive semantics through explicit adapters;
- publish manifest, compatibility, and provenance artifacts;
- expose a bounded public build-time integration surface when needed;
- let runtime consumers use static artifacts instead of private builder code;
- retain a lower-fidelity **offline** HTML compatibility path for static pages
  that cannot be represented through normal Sphinx semantics.

Start with [`MAINTAINING.md`](MAINTAINING.md). A02 has verified NVIDIA's pinned
10-cell compatibility matrix GREEN and now contains a downstream config-parity
shim under `compat/` that keeps `sphinx_llm/**` byte-identical. The shim captures
the effective primary config, integrity-checks a short-lived child snapshot, and
restores it before semantic reading. A02 includes a read-only post-CI closure-evidence verifier that recomputes the ten-cell artifact, checks coherent CircleCI pipeline/workflow/job/project/revision provenance for `scikit-plots/scikit-plots`, and never edits maintenance state. Because repository CI may drift while the campaign is open, `_maintenance/render_a02_circleci_rebase.py` can render the canonical A02 parameter/jobs/workflow into a separate candidate from the maintainer's current `.circleci/config.yml`; it refuses in-place mutation and the candidate is checked with `verify_a02_circleci_integration.py --candidate`. A02 remains **BLOCKED** because the real downstream Sphinx/`ifconfig` matrix is
still 0/10 GREEN. A CI-neutral 10-cell plan, isolated per-cell runner, and
fail-closed aggregator now live under `_maintenance/`; they avoid installing the
full scikit-plots runtime merely to test this Sphinx boundary. Keep A03+ feature
work locked until the aggregate evidence is 10/10 GREEN.
