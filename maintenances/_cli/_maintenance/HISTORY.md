# History

## 2026-11-14 — `api` command: the installed public API (PR 942)

Maintainer request: usage skills should show the API of the version a user
has, not a list frozen at one version. Step 1 of three (942 the command, 943
the skills stop shipping `references/API.md`, 944 Rule 77 reworded for the
directory's credential hold).

- `scikitplot/_api_listing.py` (standard library): the row renderer of
  `tools/maint_tools/generate_usage_skills.py` (kind, canonical signature,
  first sentence of the own docstring, Rule 168) moved into the package;
  the generator keeps its helper names as wrappers adding Markdown
  escaping. Before the switch: the old generator's functions and the
  module agree on 1,034 public names of 22 importable modules (5 need the
  compiled build), 0 differences; the same check against `main`'s
  generator after the switch: 0 differences.
- `python -m scikitplot api [MODULE] [--name N] [--format ...]`: only
  `scikitplot` and modules below it are imported (usage error, exit 2,
  otherwise); a submodule in `__all__` is listed without being imported
  (`find_spec`): listing the root takes 0.01 s and imports 3 modules
  instead of about 70 (BLD-APIREF-002); a name that does not resolve is
  listed `unavailable` with its reason instead of failing the listing (a
  partial install, or build-generated names in an unbuilt checkout);
  `--name` resolves only that name and suggests close ones; a missing
  optional dependency exits 69 (unavailable), an unknown module 1 with a
  hint that the release may predate it.
- Kept as it was: a cut signature keeps `max_length - 4` characters plus
  `" ...)"`, one over the limit, so the skills stay byte-identical.
- 24 tests on a synthetic package mounted below `scikitplot`; 12 mutants
  killed (one, markdown ignored for a module listing, after an added
  case). Frontend parity matrix: three `api` cases. CLI suite: the same 12
  environmental failures as `main` (`show-config`, module runners: no
  build here); 168 -> 177 passed.

## 2026-09-28 — delegated exemption reviewed

`test_the_delegated_exemption_is_narrow` had failed since `cleanprompt` was
registered as a delegated command: the registration was deliberate and its
frontend parity is proven by cleanprompt's own lane, but the allow-list here
was never updated, so the test reported a widening nobody had reviewed. The
exemption set is now `{"mcp", "cleanprompt"}` with the reason written beside
it. The remaining `_cli` failures in the standalone archive are the absent
package root (`__version__`, `show_config`, `scikitplot._testing`).

## 2026-09-12 — maintenance onboarding

Created a dedicated `_cli` maintenance/skill domain. The runtime tree was left
byte-for-byte unchanged. Review identified two structural findings: declared
`CommandSpec.capabilities` metadata has no enforcement path despite its normative
docstring, and runtime source references a missing `EXTENDING.md`.

The archive is also intentionally incomplete for full command tests: project
`scikitplot.config` and `scikitplot.utils` are absent, and no TOML writer is
installed in the review environment. Those environmental/snapshot gaps are kept
separate from `_cli`-owned defects.

## 2026-10-13 - CLI-MAINT-001 resolved (round 33, R2-5 / F-32)

`CommandSpec.capabilities` was removed rather than enforced: nothing read it,
and the delegate import already reports a missing part as
`CapabilityMissingError` with the command's install hint. The runtime check
no longer reports it; `test_current_runtime_reports_known_contract_findings`
now asserts its absence. The plane's other failures (missing `EXTENDING.md`,
stale fingerprints) predate this change and are unchanged.
