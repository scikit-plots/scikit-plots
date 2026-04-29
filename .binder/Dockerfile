# .binder/Dockerfile
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
#
# Binder-compatible Dockerfile.
# Reference: https://mybinder.readthedocs.io/en/latest/tutorials/dockerfile.html
#
# Local testing:
#   docker build -t binder-test .
#   # Simulate exactly what MyBinder runs:
#   docker run --rm -p 8888:8888 binder-test \
#       jupyter notebook --NotebookApp.default_url=/lab/ --ip=0.0.0.0 --port=8888
#   # Inspect uid and file ownership inside the container:
#   docker run --rm -it binder-test bash -c "whoami; id -u; ls -alh ~"

# ── Base image ────────────────────────────────────────────────────────────────
# Binder requirement: must pin an explicit digest or named tag — "latest" is
# forbidden because an updated image would silently break reproducibility.
#
# Two digest flavours:
#   Index digest  — multi-platform manifest list (what `docker pull` resolves).
#   Manifest digest — one exact platform image; use only when targeting a
#                     single architecture (e.g. linux/amd64 on Binder).
#
# Use the index digest here for portability across architectures.
#
# To refresh this digest after the upstream image is updated:
#   docker pull scikitplot/scikit-plots:latest-python-3.11
#   docker inspect --format '{{index .RepoDigests 0}}' \
#       scikitplot/scikit-plots:latest-python-3.11
#
# FROM scikitplot/scikit-plots:latest-python-3.11
FROM scikitplot/scikit-plots@sha256:02160298fd356e43cf728131cd5048479938b8b653fff15074f8961e2d96b985

LABEL maintainer="The scikit-plots developers <scikit-plots.github.io/dev>"

# pipefail: the exit status of a pipeline is the status of the last command to
# fail, not the last command.  Without this, `false | true` silently succeeds.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ── Non-root user (uid 1000) ──────────────────────────────────────────────────
# Binder requirement: container processes must not run as root; uid must be 1000.
#
# NB_GID=100 is the "users" group — the established convention used by
# repo2docker and the jupyter/docker-stacks family of images.  Using a
# well-known GID makes file sharing between host bind-mounts and the container
# predictable across environments.
ARG NB_USER=jovyan
ARG NB_UID=1000
ARG NB_GID=100

# Consolidate into a single ENV layer: each separate ENV instruction creates an
# additional image layer.  The KEY=VALUE form is the current Docker convention.
ENV USER=${NB_USER} \
    NB_UID=${NB_UID} \
    NB_GID=${NB_GID} \
    HOME=/home/${NB_USER}

# Create the user only when the base image does not already define it.
# The id -u guard prevents failure on images (e.g. jupyter/scipy-notebook)
# that ship a "jovyan" user with uid 1000 out of the box.
RUN id -u "${NB_USER}" >/dev/null 2>&1 || \
    adduser --disabled-password \
            --gecos "Default user" \
            --uid  "${NB_UID}" \
            "${NB_USER}"

# ── Jupyter packages ──────────────────────────────────────────────────────────
# Binder requirement: notebook and jupyterlab must be installed via pip.
# jupyterhub is additionally required for authenticated Binder deployments
# (JupyterHub single-user server mode).
#
# Placed BEFORE the COPY step so Docker can cache this layer independently of
# source-code changes.  Editing a notebook or gallery script will not re-trigger
# a full package download on the next build.
#
# User note:
#   `python -m pip` is preferred over the bare `pip` command.  In environments
#   where multiple Python versions are installed (e.g. conda + system Python),
#   bare `pip` resolves via PATH and may invoke a different interpreter's pip
#   than `python` itself, leading to packages being installed into the wrong
#   site-packages tree.  `python -m pip` always uses the pip that belongs to
#   the current `python` binary, eliminating the ambiguity.
#
# Developer note:
#   --no-cache-dir disables pip's HTTP response cache (~/.cache/pip).
#   This keeps the image layer size-minimal and fully reproducible: two
#   builds from the same Dockerfile always download the same bytes without
#   relying on a warm local cache.
#   NOTE: the correct flag is --no-cache-dir, NOT --no-cache.
#         --no-cache is not a recognised pip option and is silently ignored,
#         leaving the cache active and the build non-reproducible.
# ── Jupyter packages ──────────────────────────────────────────────────────────
# Binder requirement: notebook and jupyterlab must be installed via pip.
# jupyterhub is additionally required for authenticated Binder deployments
# (JupyterHub single-user server mode).
#
# Placed BEFORE the COPY step so Docker can cache this layer independently of
# source-code changes.  Editing a notebook or gallery script will not re-trigger
# a full package download on the next build.
#
# User note:
#   `python -m pip` is preferred over the bare `pip` command.  In environments
#   where multiple Python versions are installed (e.g. conda + system Python),
#   bare `pip` resolves via PATH and may invoke a different interpreter's pip
#   than `python` itself, leading to packages being installed into the wrong
#   site-packages tree.  `python -m pip` always uses the pip that belongs to
#   the current `python` binary, eliminating the ambiguity.
#
# Developer note:
#   --no-cache-dir disables pip's HTTP response cache (~/.cache/pip).
#   This keeps the image layer size-minimal and fully reproducible: two
#   builds from the same Dockerfile always download the same bytes without
#   relying on a warm local cache.
#   NOTE: the correct flag is --no-cache-dir, NOT --no-cache.
#         --no-cache is not a recognised pip option and is silently ignored,
#         leaving the cache active and the build non-reproducible.
#
# CRITICAL — install as root, not as NB_USER:
#   When pip runs as a non-root user whose home directory is $HOME
#   (/home/jovyan), and the base image's system Python is root-owned,
#   pip falls back to a user-scheme install:
#
#     ~/.local/bin/jupyter
#     ~/.local/bin/jupyterhub-singleuser
#     ~/.local/lib/python3.x/site-packages/{notebook,jupyterlab,jupyterhub}/
#
#   postBuild then calls:
#     find "${WORKDIR:?}" -mindepth 1 -delete      # WORKDIR == HOME == /home/jovyan
#
#   That single find deletes ~/.local/ entirely, destroying every binary and
#   package that pip just installed.  The build succeeds (postBuild reuses
#   the base image's older jupyter for its extension calls), but at spawn-time
#   MyBinder invokes jupyterhub-singleuser — which is gone.  The container
#   exits immediately and JupyterHub reports "Spawn failed:".
#
#   Installing as root targets the system site-packages tree
#   (/usr/local/lib/python3.x/site-packages/), which is entirely outside
#   $HOME and therefore unaffected by postBuild's workspace reset.
USER root
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir notebook jupyterlab jupyterhub

# ── Repository contents ───────────────────────────────────────────────────────
# Binder requirement: all repo contents must be present in $HOME and owned by
# the notebook user.  Docker's COPY instruction always creates root-owned files
# regardless of the active USER directive; the chown step corrects that.
#
# Developer note: COPY invalidates this layer on every source change.  Placing
# it after the pip-install layer above preserves the package-install cache
# across source-only edits.
#
# Note: use COPY (not RUN git clone) here.  mybinder.org caches Docker layers;
# a git clone inside a RUN instruction does not invalidate the cache when the
# repository is updated, so stale content would be served from cache.
WORKDIR ${HOME}
COPY . ${HOME}
USER root
RUN chown -R "${NB_UID}:${NB_GID}" "${HOME}"
USER ${NB_USER}

# ── Entrypoint & default command ─────────────────────────────────────────────
# ENTRYPOINT [] clears any entrypoint inherited from the base image.
#
# Binder requirement: the container must accept the full jupyter invocation
# that MyBinder passes at launch time, including the port, token, and other
# arguments injected by JupyterHub:
#
#   docker run <image> jupyter notebook \
#       --NotebookApp.default_url=/lab/ \
#       --ip=0.0.0.0 \
#       --port=<N> \
#       --NotebookApp.token=<T>
#
# If the base image defines an ENTRYPOINT that does not exec "$@", Docker
# passes those arguments to that entrypoint instead of to jupyter.  The
# jupyter server never starts on the expected port, so JupyterHub's health
# check times out and "Spawn failed" is reported — even though the container
# itself is running.
#
# Clearing the entrypoint here ensures Docker treats CMD as the complete
# command.  MyBinder overrides CMD entirely at runtime, so this instruction
# has no effect on Binder deployments.  It is provided solely so that
# `docker run <image>` works out-of-the-box during local development.
#
# --no-browser suppresses the attempt to open a browser tab, which always
# fails inside a Docker container and produces a confusing error message.
ENTRYPOINT []
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser"]
