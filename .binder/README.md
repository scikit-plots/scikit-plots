## 🐋 [repo2docker](https://repo2docker.readthedocs.io/en/latest/)’s documentation

[repo2docker](https://repo2docker.readthedocs.io/en/latest/) - https://repo2docker.readthedocs.io/en/latest/

- binder examples - https://github.com/binder-examples
- [Configuration files supported by repo2docker](https://repo2docker.readthedocs.io/en/latest/configuration/#config-files)

[repo2docker](https://repo2docker.readthedocs.io/en/latest/) is a tool that automatically builds a Docker image from a code repository given a configuration file. This Docker image will contain all of the code, data and resources that are listed in the repository. All the software required to run the code will also be preinstalled from the configuration file.

A list of supported configuration files (roughly in the order of build priority) can be found in the next sections.

Configuration for research and data science workflows
- environment.yml - Install a conda environment
- install.R - Install R packages
- DESCRIPTION - Install as an R package
- Project.toml - Install a Julia environment

Configuration files for software development workflows
- Pipfile and/or Pipfile.lock - Install a Python environment
- requirements.txt - Install a Python environment
- pyproject.toml - Install Python packages
- setup.py - Install Python packages

System-wide configuration
- apt.txt - Install packages with apt-get
- runtime.txt - Specifying runtimes
- default.nix - the nix package manager
- Dockerfile - Advanced environments

Configuration files for post-build actions
- postBuild - Run code after installing the environment
- start - Run code before the user sessions starts


> Reference: https://repo2docker.readthedocs.io/en/latest/
> Reference: https://github.com/jupyterhub/repo2docker-action

---

## 1. What repo2docker is

`repo2docker` converts a Git repository into a reproducible Docker image.
It reads configuration files in `.binder/` (or the repo root) and generates
a complete Dockerfile internally — the user never writes a Dockerfile.

The builder used here is **CondaBuildPack**, selected because `.binder/runtime.txt`
declares `python-3.12`.

---

## 2. Configuration files consumed (from `.binder/`)

| File | Role | Dockerfile effect |
|------|------|-------------------|
| `runtime.txt` | Pins `python-3.12` | selects CondaBuildPack; sets `NB_PYTHON_PREFIX` |
| `apt.txt` | System packages | one `RUN apt-get install` layer (step 15/22) |
| `environment.yml` | Conda + pip deps | `mamba env update` layer (step 17/22) |
| `requirements.txt` | pip flags + packages | consumed **inside** `environment.yml`'s `pip:` block |
| `postBuild` | Post-install script | `chmod + RUN` layers (steps 18–19/22) |
| `start` | Pre-launch hook | `chmod` layer (step 20/22), copied to image |

---

## 3. The 22-step build graph (log-faithful)

The build log shows the graph expanded from an internal **19-step** base
to **22 steps** after processing `.binder/`. The three added steps are:

- `apt.txt` → step 15/22 (apt-get for system deps)
- `postBuild` → steps 18/22 (chmod) and 19/22 (RUN)

Step 20/22 (chmod start) existed in the 19-step base as step 17/19;
it is renumbered when the graph expands.

### Step-by-step breakdown

```
Step  1/22  FROM buildpack-deps:24.04
             Base: Ubuntu 24.04 LTS with build tools (curl, git, gcc).
             repo2docker always uses this as the upstream base.

Step  2/22  RUN apt-get install locales
             [was 2/19] Minimal first layer: only locales.
             Uses -qq (quiet) and --no-install-recommends.
             Cleans apt lists immediately: no package cache in layer.

Step  3/22  RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen
             [was 3/19] Separates locale config from locale package install.
             Two separate layers so locale-gen cache busts independently.

Step  4/22  RUN group setup (idempotent)
             [was 4/19] Safe group creation: renames existing GID 1000
             group to "jovyan" if present, otherwise creates it fresh.
             Never errors if group already exists.

Step  5/22  RUN user setup (idempotent)
             [was 5/19] Safe user creation: renames existing UID 1000
             user to "jovyan" if present, otherwise creates it.
             --no-log-init avoids lastlog/faillog bloat in large UIDs.

Step  6/22  RUN apt-get install gettext-base less unzip
             [was 6/19] Second apt layer: minimal runtime utilities.
             Kept separate from locales so caching is independent.

Step  7/22  COPY build_script_files/.../activate-conda.sh
             → /etc/profile.d/activate-conda.sh
             Injects conda PATH activation for all login shells.
             This file is generated at build time by repo2docker itself
             from its internal CondaBuildPack templates.

Step  8/22  COPY build_script_files/.../environment.lock
             → /tmp/env/environment.lock
             A pinned lock file for the BASE conda environment.
             Contains jupyterhub/notebook/ipykernel at fixed versions.
             Generated by repo2docker, NOT from the user's environment.yml.

Step  9/22  COPY build_script_files/.../install-base-env.bash
             → /tmp/install-base-env.bash
             Shell script that: downloads micromamba, installs it to
             /srv/conda/bin/, creates /srv/conda/envs/notebook from
             the lock file, sets NB_PYTHON_PREFIX and MAMBA_EXE.

Step 10/22  RUN TIMEFORMAT='time: %3R' bash -c \
                'time /tmp/install-base-env.bash' && \
                rm -rf /tmp/install-base-env.bash /tmp/env
             Executes the base env installer with timing output.
             Cleans up the script and lock file after use (layer hygiene).
             After this step:
               /srv/conda/bin/micromamba  ← MAMBA_EXE
               /srv/conda/envs/notebook   ← NB_PYTHON_PREFIX
               Python 3.12 available in the notebook env.

Step 11/22  RUN mkdir -p /srv/npm && chown -R jovyan:jovyan /srv/npm
             Creates the Node.js package prefix used by configurable-http-proxy.
             Owned by jovyan so npm install works without root.

Step 12/22  RUN if [ ! -d "/home/jovyan" ]; then \
                  /usr/bin/install -o jovyan -g jovyan -d "/home/jovyan"; \
                fi
             Guard: creates home dir if user creation (step 5) didn't.
             Idempotent — no-op if home already exists.

Step 13/22  WORKDIR /home/jovyan
             Sets the working directory. All subsequent COPY/RUN use this.

Step 14/22  RUN chown jovyan:jovyan /home/jovyan
             Ensures home dir ownership even if created by install in step 12.

Step 15/22  RUN apt-get install <apt.txt packages>           ← FROM apt.txt
             Installs: build-essential ccache cm-super cmake ffmpeg
             gfortran graphviz inkscape libgmp-dev liblapack-dev
             libmpfr-dev libopenblas-dev ninja-build pkg-config tesseract-ocr
             Note: uses --yes (not -qq) — intentional, repo2docker applies
             -qq only to the infra layers (steps 2 and 6), not user layers.

Step 16/22  COPY --chown=1000:1000 src/ /home/jovyan/
             Copies the repository source tree into the image.
             Owned by jovyan:jovyan (1000:1000).

Step 17/22  RUN TIMEFORMAT='time: %3R' bash -c \           ← FROM environment.yml
                'time ${MAMBA_EXE} env update \
                   -p ${NB_PYTHON_PREFIX} \
                   --file ".binder/environment.yml" && \
                 time ${MAMBA_EXE} clean --all -f -y && \
                 ${MAMBA_EXE} list -p ${NB_PYTHON_PREFIX}'
             CRITICAL: mamba targets NB_PYTHON_PREFIX (/srv/conda/envs/notebook),
             NOT the base env. Packages from environment.yml are installed
             into the pre-existing notebook env (update, not create).
             The pip: block in environment.yml triggers pip INSIDE mamba:
               pip install -r ../requirements/build.txt
               pip install -r requirements.txt
             The "WARNING: Cache entry deserialization failed" messages are
             from pip's own wheel cache inside the container — harmless,
             because pip falls back to downloading. They are NOT errors.
             mamba clean removes all downloaded package files (saves ~227MB).
             mamba list prints the final package manifest to the build log.

Step 18/22  RUN chmod +x .binder/postBuild                  ← FROM postBuild
             Ensures the postBuild script is executable before running.

Step 19/22  RUN ./.binder/postBuild
             Executes postBuild as root inside the build context.
             This script:
             1. Backs up examples/ and .binder/ to a temp dir
             2. Clears WORKDIR (/home/jovyan)
             3. Converts 66 Python gallery scripts → 109 Jupyter notebooks
             4. Restores .binder/ back to WORKDIR
             5. Creates notebooks/auto_examples symlink
             6. Installs scipy-openblas64 (best-effort, via pip)
             7. Verifies the OpenBLAS installation

Step 20/22  RUN chmod +x "/home/jovyan/.binder/start"       ← FROM start
             Makes the pre-launch hook executable.
             The start script itself is NOT run at build time — it runs
             at container startup, immediately before the notebook server.

             [implicit] USER jovyan
             repo2docker switches to the non-root user before the entrypoint
             COPY steps. This does not appear as a numbered layer in BuildKit
             progress output but IS present in the generated Dockerfile.

Step 21/22  COPY --chmod=0755 /python3-login \
                 /usr/local/bin/python3-login
             Copies repo2docker's own python3-login helper binary.
             Source is the builder image filesystem, not the user repo.

Step 22/22  COPY --chmod=0755 /repo2docker-entrypoint \
                 /usr/local/bin/repo2docker-entrypoint
             Copies repo2docker's container entrypoint script.
             This is what runs when the container starts; it invokes
             .binder/start "$@" before handing off to the notebook server.
```

---

## 4. Environment variables set by repo2docker

These are set inside the generated Dockerfile (not shown in the log progress):

| Variable | Value | Purpose |
|----------|-------|---------|
| `NB_USER` | `jovyan` | Non-root username |
| `NB_UID` | `1000` | User ID |
| `NB_GID` | `1000` | Group ID |
| `NB_PYTHON_PREFIX` | `/srv/conda/envs/notebook` | Conda env path |
| `MAMBA_EXE` | `/srv/conda/bin/micromamba` | Path to micromamba binary |
| `CONDA_DIR` | `/srv/conda` | Conda root |
| `PATH` | `${NB_PYTHON_PREFIX}/bin:${CONDA_DIR}/bin:...` | Env on PATH |
| `REPO_DIR` | `/home/jovyan` | Working dir (used by postBuild) |

---

## 5. The pip cache warning — root cause

```
WARNING: Cache entry deserialization failed, entry ignored
```

**Origin**: pip's wheel cache at `~/.cache/pip/` inside the container.
**Cause**: The cache contains entries written by a different pip version or
with a different serialization format than the pip being used now (pip 25.2
vs cached metadata from an older run or different build environment).
**Effect**: pip ignores the stale cache entry and downloads the package fresh.
**Action required**: None. This is not an error. Build succeeds normally.

To silence in future builds: add `--no-cache-dir` to pip flags in
`requirements.txt` (already partially addressed via `--find-links`).

---

## 6. Key repo2docker contracts for `.binder/` files

### `environment.yml` pip block ordering
```yaml
- pip:
  - -r ../requirements/build.txt   # path relative to REPO ROOT, not .binder/
  - -r requirements.txt            # relative to .binder/ (CWD during mamba run)
```
The `../requirements/build.txt` path works because mamba's CWD during
`env update` is `/home/jovyan` (WORKDIR), so `../requirements/` resolves
to `/home/requirements/`. **This path must match your actual repo layout.**

### `postBuild` execution context
- Runs as the **build user** (typically root during build, then jovyan after USER)
- CWD is `WORKDIR` (`/home/jovyan`)
- `REPO_DIR` env var is set to CWD
- Must be idempotent (re-running should not fail)

### `start` execution context
- Runs at **container startup**, not at build time
- Must end with `exec "$@"` to hand PID to the notebook server
- Any failure exits the session

---
