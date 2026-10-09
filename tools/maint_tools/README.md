# Maintenance tools

Repository-local maintenance helpers live here. They are intended to be safe for
humans, CI, and AI-assisted maintenance without requiring prior chat history.

## CLI policy for Python helpers

New Python maintenance CLIs should follow the repository's hardened CLI pattern
used by `scikitplot.mcp`:

- importing the module must not parse arguments or mutate the repository;
- build arguments in a `_parser()` function;
- expose `main(argv=None, *, stdout=None, stderr=None) -> int` when practical;
- finish with `raise SystemExit(main())` only under `if __name__ == "__main__"`;
- make read-only behavior the default;
- require an explicit flag such as `--apply` for mutations;
- return non-zero status for validation failures or refused unsafe actions;
- provide deterministic output and a machine-readable mode when automation
  benefits from it;
- avoid network access unless the command explicitly exists for a network task;
- validate repository-relative paths before writing or deleting files.

These rules make helpers testable, scriptable, and safe to call repeatedly.

## Towncrier ownership helper

`generate_towncrier_sections.py` is the canonical helper for release-note section
coverage. The durable ownership policy is stored in
`[tool.scikitplot.maintenance.towncrier]` in `pyproject.toml`.

Common commands:

```sh
# Validate policy, pyproject configuration, fragment directories and fragments.
python tools/maint_tools/generate_towncrier_sections.py check

# Show the expected sections derived from the current repository tree.
python tools/maint_tools/generate_towncrier_sections.py list

# Preview synchronization. This never writes.
python tools/maint_tools/generate_towncrier_sections.py sync --prune-empty

# Apply the same deterministic synchronization after reviewing the preview.
python tools/maint_tools/generate_towncrier_sections.py sync --apply --prune-empty

# Machine-readable forms for CI/AI callers.
python tools/maint_tools/generate_towncrier_sections.py check --json
python tools/maint_tools/generate_towncrier_sections.py sync --prune-empty --json
```

The helper currently derives owners from:

- immediate Python packages under `scikitplot/`;
- immediate components under `libs/`;
- immediate components under `tools/`;
- explicitly configured top-level owner sections; and
- selected deeper Python packages declared in
  `[tool.scikitplot.maintenance.towncrier.nested_owner_sections]`.

The nested-owner mapping is intentionally opt-in. Keys are exact parent package
paths and values are relative child package paths; dotted child paths are allowed
for deeper ownership boundaries. The normal base owner remains active, so unlisted
nested packages continue to roll up to the nearest default owner. For example:

```toml
[tool.scikitplot.maintenance.towncrier.nested_owner_sections]
"scikitplot.externals" = ["_probscale", "array_api_compat"]
"scikitplot._externals._sphinx_ext" = ["_sphinx_ai_assistant"]
```

The helper validates that every requested parent and child is an importable package
in the checkout. Missing paths, invalid dotted names, duplicate/redundant owners, and
conflicts with `exclude_owner_sections` fail explicitly rather than silently changing
release-note ownership. Explicit exclusions such as test-only or archived trees are
declared in `pyproject.toml`, not hidden in the script.

`sync --apply` may rewrite only the marked auto-generated owner block in
`pyproject.toml`, create missing section directories, and remove stale section
directories only when `--prune-empty` is supplied and those directories are
empty or contain only `.gitkeep`. It refuses to delete populated stale sections.

## API reference generator

`scikitplot/_build_utils/generate_apis_reference/` is the canonical maintenance
package for `docs/source/apis_reference.py`. Its adjacent
`apis_reference.py.in` is the durable, human-editable template; the docs file is
generated output and should be reproducible from that template. The package is
installed with Scikit-Plots, exposes importable parsing/customization helpers,
and keeps runtime symbol verification separate from offline rebuilds.

```sh
# Verify generated output exactly matches the canonical template.
python -m scikitplot._build_utils.generate_apis_reference rebuild --check

# Preview a full rebuild. No package API imports and no write.
python -m scikitplot._build_utils.generate_apis_reference rebuild

# Recreate the generated file even if it was deleted.
python -m scikitplot._build_utils.generate_apis_reference rebuild --apply

# Runtime verification requires an installed/importable Scikit-Plots build.
python -m scikitplot._build_utils.generate_apis_reference check
python -m scikitplot._build_utils.generate_apis_reference check --strict-coverage
python -m scikitplot._build_utils.generate_apis_reference plan
python -m scikitplot._build_utils.generate_apis_reference inventory
```

Do not hand-edit `docs/source/apis_reference.py` for durable changes. Edit
`generate_apis_reference/apis_reference.py.in`, run `rebuild`, and review the
diff. Advanced Python tooling may call `reference_blueprint_copy()` to derive an
isolated mutable mapping from the template and pass it to
`build_reference_source()`. Non-literal trusted expressions such as
`_get_guide(...)` and `_get_submodule(...)` are preserved as validated
`PythonExpression` values; rebuild does not execute them.

The existing `generate` command remains the installed-package symbol-list
planner: it detects stale names and unambiguous new exports without executing
the docs configuration. Its dry-run output is discovery evidence; accepted
symbol changes must be written back to the canonical `.py.in` template and a
clean `rebuild --check` must follow.

Runtime generation remains conservative: prefer `__all__` when present,
otherwise inventory locally defined public objects; exclude module objects by
default; auto-assign only unambiguous symbols; and report optional/compiled
import failures as unverified rather than using them as deletion evidence.
