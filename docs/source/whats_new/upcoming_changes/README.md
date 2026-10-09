# Scikit-Plots changelog fragments

This directory contains Towncrier **release-note fragments** for changes that
will ship in the next Scikit-Plots release.

It is deliberately separate from the repository-root `upcoming_changes/`
engineering follow-up ledger:

- `upcoming_changes/` records verified unresolved work that still needs a future
  implementation or documentation task;
- `docs/source/whats_new/upcoming_changes/` records concise, user-facing changes
  that are actually being delivered.

Do not symlink or merge those lifecycles.

## The section model

The release-note structure has two kinds of sections.

### Cross-cutting sections

Use these when the change is primarily about a repository-wide concern rather
than one package owner:

| Section path | Use for |
| --- | --- |
| `array-api` | Array API support spanning one or more modules |
| `build-packaging` | builds, wheels, source distributions, install/packaging behavior |
| `compatibility` | dependency/platform/version compatibility changes |
| `documentation` | documentation-only user-facing changes |
| `many-modules` | one coherent change that materially affects several package owners |
| `metadata-routing` | metadata-routing behavior spanning owners |
| `security` | security/privacy hardening or fixes |

### Source-owner sections

For normal module work, section selection is deterministic:

- a root-level `scikitplot/*.py` change uses `scikitplot`;
- a change owned by `scikitplot/<package>/...` uses
  `scikitplot.<package>`;
- a change owned by `libs/<component>/...` uses `libs.<component>`;
- a release-worthy contributor/tooling change in a root `tools/*` file uses
  `tools`, while `tools/<component>/...` uses `tools.<component>`.

Every immediate non-excluded Python package under `scikitplot/`, every immediate
`libs/` component, and every immediate non-excluded `tools/` component has a
configured Towncrier section. Nested implementation paths normally roll up to their
nearest owner. Selected deeper Python packages can opt into dedicated ownership via
`[tool.scikitplot.maintenance.towncrier.nested_owner_sections]` in `pyproject.toml`.
The base owner is retained, so siblings that are not explicitly selected still roll
up normally. For example:

```text
scikitplot/api/metrics/...
    -> scikitplot.api
scikitplot/corpus/_embeddings/...
    -> scikitplot.corpus
scikitplot/cexternals/_astropy/...
    -> scikitplot.cexternals._astropy        # explicitly tracked nested owner
scikitplot/cexternals/other_nested/...
    -> scikitplot.cexternals                 # default roll-up
scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/...
    -> scikitplot._externals._sphinx_ext._sphinx_ai_assistant
libs/rank-bm25/...
    -> libs.rank-bm25
tools/maint_tools/...
    -> tools.maint_tools
tools/list_changed_files.py
    -> tools
```

Do **not** add a Towncrier section for every nested implementation package. That
would make release tracking harder rather than safer. Add a nested owner only when
it is a durable release-note boundary with enough independent change volume to
justify separate tracking. Configure it in the policy mapping rather than manually
editing the generated `[[tool.towncrier.section]]` block.

When a bundled `libs/` change is only an implementation detail of a public
Scikit-Plots feature, prefer the corresponding `scikitplot.<package>` section.
Use a `libs.*` section when the bundled/source-library component itself is the
meaningful release owner. Likewise, the existence of `tools` sections does not
make every internal maintenance edit release-note-worthy. Use them when a tooling
change materially affects contributors, release/build behavior, or another
delivered workflow that belongs in the changelog.

## Choose the section before the type

Use this decision order:

1. Security/privacy issue? Use `security`.
2. Documentation-only delivered change? Use `documentation`.
3. Build/install/package behavior? Use `build-packaging`.
4. Dependency/platform/version compatibility? Use `compatibility`.
5. Cross-module Array API or metadata-routing behavior? Use the matching topic.
6. One coherent change across several unrelated module owners? Use
   `many-modules`.
7. Otherwise use the nearest source-owner section.

The section answers **where the change belongs**. The type answers **what kind
of change it is**.

## Fragment type

Create one fragment named:

```text
<PULL_REQUEST>.<TYPE>.rst
```

`<PULL_REQUEST>` is the Scikit-Plots pull-request number. The allowed type names
come from `[[tool.towncrier.type]]` in `pyproject.toml`:

| Type | Use for |
| --- | --- |
| `major-feature` | a substantial new user-facing capability |
| `feature` | a new user-facing capability |
| `efficiency` | material performance/resource-use improvement |
| `enhancement` | meaningful improvement to existing behavior |
| `fix` | correction of incorrect/broken behavior |
| `api` | public API contract, deprecation, rename, signature, or compatibility change |
| `other` | release-worthy change that does not fit the tagged categories |

Do not infer a type from the source directory alone.

## Fragment content

In almost all cases, write exactly **one ReStructuredText bullet**. Describe the
verified effect for users or contributors, not the implementation diff.

Prefer:

```rst
- :func:`api.some_public_function` now handles <verified case> without
  <previous user-visible failure>. By :user:`Contributor <github-handle>`
```

Use the actual public object when one exists. Do not invent an API name merely
to make the fragment look complete. Internal paths can be useful evidence while
reviewing a change, but should rarely be the release-note sentence itself.

## Validate and synchronize section coverage

`pyproject.toml` is the durable policy/configuration source. The default owner
model is one level deep; the optional `nested_owner_sections` mapping adds only
selected deeper Python-package owners. Its keys are exact parent packages and its
values are relative child package paths, including dotted relative paths when a
future owner is deeper than one additional level. Run the maintenance helper after
changing that mapping; do not hand-maintain the generated owner block or directories.

`[tool.scikitplot.maintenance.towncrier]` defines which repository roots generate
release-note owners, which owners are intentionally excluded, and which
cross-cutting sections are curated manually. The current repository tree supplies
the actual immediate owners.

Use the repository helper instead of rebuilding this mapping by hand:

```sh
python tools/maint_tools/generate_towncrier_sections.py check
python tools/maint_tools/generate_towncrier_sections.py list
```

The current structural ownership rule covers:

- the `scikitplot` top level and each immediate non-excluded Python package;
- each immediate `libs/` component;
- the `tools` top level and each immediate non-excluded `tools/` component;
- the explicit cross-cutting topics configured in `pyproject.toml`.

Nested implementation paths roll up to the nearest owner. Test-only or archived
trees such as `scikitplot.tests` and `tools.zz_yanked` are explicit policy
exclusions, not hidden exceptions in the helper.

The check fails when configuration, owner directories, fragment types, fragment
filenames, or fragment structure drift. To repair structural drift, preview first:

```sh
python tools/maint_tools/generate_towncrier_sections.py sync --prune-empty
```

After reviewing the plan, apply the same deterministic operation explicitly:

```sh
python tools/maint_tools/generate_towncrier_sections.py sync --apply --prune-empty
```

`sync` is read-only without `--apply`. It never removes a populated stale section;
those require manual review so an unreleased fragment cannot be deleted merely
because ownership changed. Machine-readable `--json` output is available for CI
and AI-assisted maintenance.

## From an engineering finding to a release fragment

When a root `upcoming_changes/` finding is implemented:

1. verify the implementation and relevant compatibility/security claims;
2. decide whether the result is release-note-worthy;
3. choose the section using the rules above;
4. choose the Towncrier type;
5. create `<PR>.<TYPE>.rst` in that section directory;
6. record the final fragment path in the root planning note while the handoff is
   active;
7. remove the root planning note when it is no longer needed as current work.

A documentation-only change can create a `documentation` fragment directly. It
does not need a root planning note unless the review exposes unresolved runtime,
build, compatibility, security, or docs-contract work.

## Preview the next changelog

With the documentation/release dependencies installed, preview without consuming
fragments:

```sh
towncrier build --draft --version 0.5.0
```

The configured target file, fragment directory, template, types, and sections
are defined in `pyproject.toml`. Use the actual next release version when it
changes.

## Safety and quality

- Never include credentials, private URLs, tokens, bearer capabilities, user
  data, or sensitive log content.
- Do not claim platform/dependency/integration support without current evidence.
- Describe public behavior rather than promising private implementation details.
- Keep one fragment focused on one coherent delivered change.
- Do not use a release fragment as a substitute for a detailed unresolved-work
  note.
