---
title: "Native Hugging Face and Bitbucket Data Center feedback adapters"
status: open
kind: "other"
area: "scikitplot/_externals/_sphinx_ext/_sphinx_feedback/_service"
discovered_during: "round 29 (2026-10-11); narrowed in round 30 after GitLab and Bitbucket Cloud shipped"
release_note: "required"
towncrier_section: "scikitplot._externals._sphinx_ext._sphinx_feedback"
towncrier_type: "feature"
towncrier_fragment: ""
---

# Native Hugging Face and Bitbucket Data Center feedback adapters

## Summary

Round 30 extracted the reviewed-change state machine into
`_service/_forge.py` (GitHub now runs on it, its 27 tests unchanged) and
shipped `gitlab` (gitlab.com and self-managed) and `bitbucket` (Cloud) on the
same engine. Two forges remain: `huggingface` is accepted by the schema and
refused at start (`IMPLEMENTED_PROVIDERS`); Bitbucket Data Center is refused
by `api_url` validation because its REST API (`/rest/api/1.0`) differs from
Cloud's. Both are reachable today through `webhook` or `custom`.

## Why it matters

Documentation teams on the Hugging Face Hub, or on an intranet Bitbucket
Data Center, get a native reviewed path instead of operating a receiver.

## Current evidence

- `load_service_config` with a `huggingface` primary: "configured feedback
  provider adapter(s) are not implemented in this release: huggingface".
- `bitbucket` with `api_url=https://bitbucket.corp.example/rest/api/1.0`:
  "bitbucket api_url must be https://api.bitbucket.org/2.0".

## Root cause / current understanding

Adapters not written. A new forge now needs only a `ForgeOps` subclass (eight
operations) plus the parity scenarios in `tests/test_forge_providers.py`.

## Expected behavior

- `huggingface`: the Hub's commit API can open a pull request
  (`create_pr`); `ForgeOps` maps onto a PR branch (`refs/pr/<n>`), the
  dataset file API and the discussions API. The review-number namespace
  differs from git branches, so `review_branch()` needs a per-forge mapping.
- Bitbucket Data Center: `ForgeOps` over `/rest/api/1.0/projects/{key}/repos/{slug}`
  (branches, browse/raw, `files` commit via `PUT /browse/{path}`,
  `compare/commits`, pull requests).

## Affected paths and ownership

`_service/_forge.py` (only if HF needs the branch mapping), new
`_service/_huggingface.py` / `_service/_bitbucket_dc.py`, `_config.py`,
`_core.py` `_NETWORK_SUBMITTERS`, the proxy mirror list.

## Constraints and non-goals

No adapter ships without the shared parity scenarios (fresh, replay, broken
pipe after branch/file/review, conflict, smuggled change, extra history,
merged replay, foreign review host, redirect refusal, missing base,
credential missing) and a read-only `check-provider` path.

## Edge cases to cover

HF repo types (model/dataset/space) and gated repos; Data Center personal vs
HTTP access tokens; self-signed intranet CAs (operator CA bundle, never
disabled verification).

## Proposed direction

Bitbucket Data Center first (git semantics match the engine), then Hugging Face
with the branch-mapping change.

## Verification / acceptance criteria

The parity suite parametrized over the new forge passes; `check-provider`
reports `ok` against a real repository.

## Documentation impact

README "Storage providers" table; `_example_conf.py` section 9.

## Release-note promotion

Feature fragment under `scikitplot._externals._sphinx_ext._sphinx_feedback`.
