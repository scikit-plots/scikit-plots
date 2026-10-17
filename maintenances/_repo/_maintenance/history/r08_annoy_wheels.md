<!-- Archived verbatim from tasks/todo.md at 471e46c (increment 4R).
     Do not edit: maintenances/_repo/_maintenance/tools/check_records.py
     checks every archived part against records_manifest.json. -->

## Task: Round 8 - the annoy wheels: the test step cannot find scikit-plots-skinny

### Context
- Symptom (your log, `build_annoy`, manylinux x86_64, cp310): the wheel builds and is
  repaired; `CIBW_BEFORE_TEST` then runs `pip install --no-index --find-links
  /project/dist/libs scikit-plots-skinny` and pip answers "Location '/project/dist/libs' is
  ignored: it is either a non-existing path" and "No matching distribution".
- The pure job (`build --outdir dist/libs`, `twine check --strict`) is green.
- Head: `refs/pull/849/head` b745a35; the drop-in is a diff against it.

### Root cause
`{project}` is not the repository in this job. `package-dir` is the annoy sdist, and
cibuildwheel 4.1.0 (the pinned release, read from its source):
1. `__main__.py`: a `package_dir` that is a `.tar.gz` is extracted into a temporary
   directory, and the build runs after `contextlib.chdir(project_dir)` into it;
2. `platforms/linux.py`: `container.copy_into(Path.cwd(), "/project")`; the other platforms
   pass `project=Path.cwd()` / `"."` for the test step.

So `{project}` is the extracted sdist everywhere, and the sdist holds no `dist/libs` (its
top level: `LICENSE.txt MANIFEST.in PKG-INFO README.md README_ANNOY.md pyproject.toml
scikit_plots_annoy.egg-info scikitplot setup.cfg setup.py`). The workflow's comment said the
container "sees the repository ... as {project}": that premise was mine, and unverified.

### Fix
- `build_annoy`: `--find-links` names the host path of `dist/libs`. On Linux the build and
  test run in a container where cibuildwheel mounts the host's root at `/host`
  (`oci_container.py`: `--volume=/:/host` unless `disable_host_mount`), so the path is
  `/host${{ github.workspace }}/dist/libs`; on macOS and Windows the test runs on the host:
  `${{ github.workspace }}/dist/libs`.
- `build_annoy_pyodide`: same cause, same fix (`${{ github.workspace }}/dist/libs`); Pyodide
  builds and tests on the host.

### Evidence
- The cibuildwheel 4.1.0 source lines above.
- cibuildwheel's flow reproduced by hand: extract the sdist, change into it, install from
  `<extracted>/dist/libs`: "Failed to read --find-links directory ... No such file or
  directory"; from the host path of the built wheels: `scikit-plots-skinny 0.5.dev0`
  installed.
- The YAML as GitHub reads it (one line after folding): `pip install --no-index
  --find-links ${{ (matrix.buildplat[1] == 'manylinux' || matrix.buildplat[1] ==
  'musllinux') && format('/host{0}/dist/libs', github.workspace) || format('{0}/dist/libs',
  github.workspace) }} scikit-plots-skinny`.
- actionlint (expressions and contexts): no findings, before and after. zizmor 1.30.1: no
  findings.
- Not run: a container build (no Docker daemon here) and macOS, Windows, Pyodide.

### Acceptance Criteria
- [x] Cause read in the pinned tool's source and reproduced.
- [ ] `build_annoy` test step installs scikit-plots-skinny on every row (CI only).
