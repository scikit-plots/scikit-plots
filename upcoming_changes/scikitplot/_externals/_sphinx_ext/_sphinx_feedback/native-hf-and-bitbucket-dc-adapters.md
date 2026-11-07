---
title: "Native Bitbucket Data Center and Hugging Face feedback adapters (Data Center shipped; Hugging Face refused with its reason)"
status: promoted
kind: "other"
area: "scikitplot/_externals/_sphinx_ext/_sphinx_feedback/_service"
discovered_during: "round 29 (2026-10-11); narrowed in round 30 after GitLab and Bitbucket Cloud shipped; resolved in PR 909 (Bitbucket Data Center shipped, Hugging Face refused with its reason)"
release_note: "required"
towncrier_section: "scikitplot._externals._sphinx_ext._sphinx_feedback"
towncrier_type: "feature"
towncrier_fragment: "docs/source/whats_new/upcoming_changes/scikitplot._externals._sphinx_ext._sphinx_feedback/909.feature.rst"
---

# Native Bitbucket Data Center and Hugging Face feedback adapters (Data Center shipped; Hugging Face refused with its reason)

## Summary

Round 30 extracted the reviewed-change state machine into
`_service/_forge.py` (GitHub now runs on it, its 27 tests unchanged) and
shipped `gitlab` (gitlab.com and self-managed) and `bitbucket` (Cloud) on the
same engine. PR 909 added Bitbucket Data Center and decided Hugging Face: it
stays refused at start, now with the reason, because the Hub's pull-request
model cannot keep the reviewed-change guarantees (see both resolutions below).
The Hub stays reachable through `webhook` or `custom`.

## Why it matters

Documentation teams on the Hugging Face Hub, or on an intranet Bitbucket
Data Center, get a native reviewed path instead of operating a receiver.

## Current evidence

- (updated in PR 909) `load_service_config` with a `huggingface` primary or
  mirror: "feedback provider huggingface is not supported: Hub pull requests
  are numbered by the Hub and its API has no lookup by feedback ID (it
  documents listing only by author, type and status), so a retry could open a
  second review; deliver feedback to the Hub through the webhook or custom
  provider". Before:
  "... adapter(s) are not implemented in this release: huggingface".
- (resolved in PR 909) `bitbucket` with
  `api_url=https://bitbucket.corp.example/rest/api/1.0` now runs the Data
  Center adapter.

## Root cause / current understanding

Adapters not written. A new forge now needs only a `ForgeOps` subclass (eight
operations) plus the parity scenarios in `tests/test_forge_providers.py`, and
an exact lookup of its review by branch: Bitbucket Data Center has one
(`pull-requests?at=refs/heads/<branch>`), the Hub does not (PR numbers are
assigned by the Hub; see the Hugging Face resolution).

## Expected behavior

- (decided in PR 909) `huggingface`: refused at start with the reason above.
  The direction first planned here (the commit API's `create_pr` plus a
  per-forge `review_branch()` mapping onto `refs/pr/<n>`) cannot find the
  review again by feedback ID; see the Hugging Face resolution.
- (shipped in PR 909) Bitbucket Data Center: `ForgeOps` over
  `/rest/api/1.0/projects/{key}/repos/{slug}` (branches, browse/raw, `files`
  commit via `PUT /browse/{path}`, `compare/commits`, pull requests).

## Affected paths and ownership

Changed in PR 909: `_service/_forge.py` (multipart bodies), new
`_service/_bitbucket_dc.py`, `_service/_bitbucket.py`, `_config.py`
(product selection, `UNSUPPORTED_PROVIDER_REASONS`), `_core.py` (dispatch
guard text), `_cli.py`, the proxy mirror list. No `_service/_huggingface.py`.

## Constraints and non-goals

No adapter ships without the shared parity scenarios (fresh, replay, broken
pipe after branch/file/review, conflict, smuggled change, extra history,
merged replay, foreign review host, redirect refusal, missing base,
credential missing) and a read-only `check-provider` path.

## Edge cases to cover

HF repo types (model/dataset/space) and gated repos. (Handled in PR 909 for
Data Center: personal, project and repository HTTP access tokens are all sent
as `Authorization: Bearer`; a private intranet CA is trusted through
`SSL_CERT_FILE`, verification is never disabled.)

## Proposed direction

Bitbucket Data Center first (git semantics match the engine), then Hugging Face
with the branch-mapping change. (Done for Data Center; superseded for Hugging
Face by the decision recorded below.)

## Verification / acceptance criteria

The parity suite parametrized over the new forge passes; `check-provider`
reports `ok` against a real repository.

## Documentation impact

README "Storage providers" table; `_example_conf.py` section 9.

## Release-note promotion

Feature fragment under `scikitplot._externals._sphinx_ext._sphinx_feedback`.

## Resolution (PR 909): Bitbucket Data Center

Shipped as a ``bitbucket`` target whose ``api_url`` is a Data Center REST root
(``https://<host>[/<context>]/rest/api/1.0``; :func:`_config.bitbucket_flavor`
picks the product, anything else is refused): ``_service/_bitbucket_dc.py``
(``BitbucketDataCenterOps``), selected by ``_bitbucket.bitbucket_ops`` in the
submit path and in ``check-provider``; ``_forge._bounded_json`` gained a
``files`` argument for the multipart file write; ``_sphinx_feedback`` is
0.10.0 and the proxy mirror follows it, carrying the new module. Personal
repositories (``~user/<slug>``, other web paths) are refused by the ``repo``
pattern, as the README says. Fragment:
``docs/source/whats_new/upcoming_changes/scikitplot._externals._sphinx_ext._sphinx_feedback/909.feature.rst``.

- Call shapes from Atlassian's REST reference index (5.16), the resource
  descriptions generated from it, the ``ChangeType`` Javadoc and the HTTP access
  token page (sources listed in the module docstring); the JSON key of a change's
  type is not quoted in them, so ``check-provider`` reads the base head's changes
  and fails when their shape differs (read-only).
- Tests: the 14 shared parity cases now run over three forges (42), plus 14
  Data Center cases (product selection and the refused personal repository,
  branch and review paging, compare paging, capped changes, only ``ADD`` is
  added, trusted change paths, ``/overview`` links, the review number, another
  target or repository is not ours, the read-only shape check) and a
  ``check-provider`` test; 16 mutations of the adapter and its config rules are
  each caught; feedback suite 422 -> 451 passed, 0 failed. The AI-assistant
  tree with the proxy's locked package versions: 2830 passed, 2 skipped,
  before and after.
- Not done here: a run against a real Data Center server (none reachable from
  this environment). Run ``check-provider`` against one before enabling it.

## Resolution (PR 909): Hugging Face refused, with its reason

The engine's guarantees need two things from a forge: an exact lookup of the
review a feedback ID opened (so a retry after a broken connection, or a
replay, finds it instead of opening a second one) and a proof that the review
adds exactly the event file. The Hub documents neither:

- Pull requests are refs named by the Hub (``refs/pr/<n>``), not branches or
  forks (Hub documentation, "Pull requests and Discussions"); a new one gets
  its number from the server (the discussion-creation response's ``num``).
- The discussions listing is filtered by ``author``, ``type`` and ``status``
  (the Hub client guide). The OpenAPI specification also lists ``search`` and
  ``sort`` with no documented behaviour, so neither can prove absence: a
  missed match would open a duplicate review.
- ``compare/{compare}`` ("the diff between the two revisions") and the
  discussion details' ``diff`` have no documented format, so a pull request
  cannot be shown to add only the event file.

A claim branch per feedback recording the review number would make lookups
exact, but at the cost of an extra branch per event, empty draft pull requests
when a creation response is lost, and a scope proof still resting on an
undocumented diff; the maintainer chose to keep ``huggingface`` refused.

- ``_config.UNSUPPORTED_PROVIDER_REASONS`` holds the reason;
  ``IMPLEMENTED_PROVIDERS`` is derived from it, and the start-up refusal (for a
  primary or a mirror) states it. ``_core``'s dispatch guard (reached only by a
  configuration that bypassed the loader) says the provider has no adapter.
- README: the reason and the ``webhook`` / ``custom`` routes; the providers
  table row says "not supported".
- Tests: the refusal test is rewritten to the reasoned message, for a primary
  and a mirror, and pins ``huggingface`` as the only provider without an
  adapter; a new test pins the dispatch guard's code
  (``provider_not_implemented``). Five mutations (the old message, a missing
  reason, a primary-only check, the guard's text, the guard's code) are each
  caught; feedback suite 451 -> 452 passed, 0 failed; the AI-assistant tree
  with the proxy's locked package versions: 2830 passed, 2 skipped.
