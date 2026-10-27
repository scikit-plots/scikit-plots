# Maintaining the repository (`_repo`)

Repository-wide work that no `scikitplot/<submodule>` owns: CI workflows,
the `libs/` partial distributions and `libs/_tools`, PR and issue labeling,
release tooling, and the work records themselves
([`check_records.py`](_maintenance/tools/check_records.py)).

Unlike the other `maintenances/<area>/` folders, `_repo` has no
`scikitplot/` counterpart: it is the area of the repository as a whole. The
repository-wide procedure (read order, planes, evidence) is
[`maintenances/MAINTAINING.md`](../MAINTAINING.md).

- Active tasks: [`todo/todo.md`](todo/todo.md).
- Where other records live: [`maintenances/README.md`](../README.md).
- Every archive, with the areas it names: [`HISTORY_INDEX.md`](_maintenance/HISTORY_INDEX.md).
- Rules: [`plugins/scikitplot-dev/skills/scikitplot-engineering-rules/SKILL.md`](../../plugins/scikitplot-dev/skills/scikitplot-engineering-rules/SKILL.md).

## Completed (archived verbatim)

| Round | Archive |
|---|---|
| Partial distributions (`libs/<name>`) | [`partial_distributions.md`](_maintenance/history/partial_distributions.md) |
| 2, version-free declarations, Python 3.15, WebAssembly | [`r02_declarations.md`](_maintenance/history/r02_declarations.md) |
| 3, first CI run: zizmor, verify legs, Cython report | [`r03_first_ci.md`](_maintenance/history/r03_first_ci.md) |
| 4, Windows leg, core/parts compatibility, staging wheels | [`r04_windows_leg.md`](_maintenance/history/r04_windows_leg.md) |
| 5, second CI run, threads at run time, bundle manifest | [`r05_second_ci.md`](_maintenance/history/r05_second_ci.md) |
| 6, third CI run: the Windows leg | [`r06_third_ci.md`](_maintenance/history/r06_third_ci.md) |
| 7, fourth CI run: two Windows tests | [`r07_fourth_ci.md`](_maintenance/history/r07_fourth_ci.md) |
| 8, annoy wheels: the test step cannot find skinny | [`r08_annoy_wheels.md`](_maintenance/history/r08_annoy_wheels.md) |
| 31, PR and issue labeling | [`r31_labeling.md`](_maintenance/history/r31_labeling.md) |
| 32, CI red on PR 864 | [`r32_ci_pr864.md`](_maintenance/history/r32_ci_pr864.md) |

`partial_distributions_removed_files.txt` is the deletion list that shipped
with the partial-distributions drop-in (formerly `tasks/removed-files.txt`).

## Checks

```sh
python -B maintenances/_repo/_maintenance/tools/check_records.py
python -B maintenances/_repo/_maintenance/tests/test_records.py
```
