# docker/Dockerfile
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

######################################################################
## Dockerfile
## References: https://hub.docker.com/r/jupyter/base-notebook/dockerfile
## https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#image-relationships
## https://github.com/jupyter/docker-stacks/blob/main/images/base-notebook/Dockerfile#L58
##
## Use the official Jupyter Docker Stacks
## jupyter do not come with CUDA drivers or GPU support by default
## https://github.com/jupyter/docker-stacks/blob/main/images/docker-stacks-foundation/Dockerfile
## "ubuntu:focal" or "ubuntu:20.04"
## "ubuntu:jammy" or "ubuntu:22.04"
## "ubuntu:bookworm" or "ubuntu:24.04"
## "ubuntu:devel"
## "ubuntu:rolling"
## "ubuntu:latest"
## "jupyter/docker-stacks-foundation:latest"
## "jupyter/base-notebook:latest"
## "jupyter/minimal-notebook:latest"
## "jupyter/scipy-notebook:latest"
## "jupyter/r-notebook:latest"
## "jupyter/tensorflow-notebook:latest" or "tensorflow/tensorflow:latest-gpu-jupyter"
## "jupyter/pytorch-notebook:latest"
## "jupyter/datascience-notebook:latest"
## "jupyter/pyspark-notebook:latest"
## "jupyter/all-spark-notebook:latest"
######################################################################
## Images hosted on Docker Hub are no longer updated. Please, use quay.io image⁠: https://quay.io/repository/jupyter/tensorflow-notebook
## Pull this container with the following Docker command:
## docker pull quay.io/jupyter/tensorflow-notebook
######################################################################
## https://github.com/tensorflow/build/blob/master/tensorflow_runtime_dockerfiles/cpu.Dockerfile
## "tensorflow/tensorflow:latest-gpu-jupyter"
######################################################################

## ARG: You can override it during the docker build with --build-arg.
## Ubuntu 20.04 (focal), https://hub.docker.com/_/ubuntu/?tab=tags&name=focal
# ARG BASE_IMAGE=jupyter/scipy-notebook:latest@sha256:fca4bcc9cbd49d9a15e0e4df6c666adf17776c950da9fa94a4f0a045d5c4ad33
ARG BASE_IMAGE=jupyter/scipy-notebook:latest
ARG SCIKITPLOT_VERSION=0.4.0.post11
## ⚠️ Must match the CI build-arg name: PYTHON_VERSION=${{ matrix.python_version }}
ARG PYTHON_VERSION=3.12
ARG ENV_NAME
## runtime = lean image (no compilers); devel = full build toolchain
ARG VARIANT=runtime
## Optional: GPU / heavy BLAS packages — orthogonal to VARIANT, OFF by default
ARG EXTENDED_PACKAGES=false

## Use the argument to specify the base image
FROM ${BASE_IMAGE} AS base
## ⚠️ ARG may specified again because the FROM directive resets ARGs
## (but their default value is retained if set previously)

ARG SCIKITPLOT_VERSION
ARG PYTHON_VERSION
ARG ENV_NAME
## VARIANT must be redeclared after FROM; value carried through from pre-FROM default or --build-arg
ARG VARIANT=runtime

ENV SCIKITPLOT_VERSION=${SCIKITPLOT_VERSION}
## Use ENV with bash-style substitution to derive ENV_NAME=py39 / py311 / py312 / py314
## Replace '.' in PYTHON_VERSION so py3.12 → py312
ENV ENV_NAME=${ENV_NAME:-py${PYTHON_VERSION//./}}
## Promote VARIANT to ENV so every subsequent RUN and the post-create script can read it
ENV VARIANT=${VARIANT:-runtime}

## ── env-file paths (all promoted to ENV so the post-create script needs no Dockerfile changes) ─
## Layering: base always applied → runtime always applied → devel applied only when VARIANT=devel
ENV ENV_DIR=docker/env_conda
ENV ENV_FILE_BASE=docker/env_conda/environment.base.yml
ENV ENV_FILE_RUNTIME=docker/env_conda/environment.yml
ENV ENV_FILE_DEVEL=docker/env_conda/environment.devel.yml

# Needed if you want the conda env active for subsequent shell-form RUN steps
ARG MAMBA_DOCKERFILE_ACTIVATE=1
# Optional: run repo post-create scripts at build time
ARG RUN_POST_CREATE=1

## Set environment variable to avoid interactive prompts during builds
## Prevent apt install tzinfo from asking our location (assumes UTC)
ENV DEBIAN_FRONTEND=noninteractive

## Fail fast on unrecognised VARIANT so a typo (e.g. "runime") errors at build time
## rather than silently producing a half-baked image.
RUN case "${VARIANT}" in \
      runtime|devel) echo "VARIANT=${VARIANT} ✓" ;; \
      *) echo "Invalid VARIANT='${VARIANT}' (expected runtime|devel)" >&2; exit 2 ;; \
    esac

# MAINTAINER The scikit-plots developers <scikit-plots.github.io/dev>
# LABEL org.opencontainers.image.base.name="${BASE_IMAGE}"
LABEL maintainer="The scikit-plots developers <scikit-plots.github.io/dev>"
# LABEL org.opencontainers.image.authors="The scikit-plots developers <scikit-plots.github.io/dev>"
# LABEL org.opencontainers.image.vendor="The scikit-plots developers <scikit-plots.github.io/dev>"
# LABEL org.opencontainers.image.title="scikit-plots"
# LABEL org.opencontainers.image.description="Scikit-plots Docker image with Jupyter Notebook support"
# LABEL org.opencontainers.image.documentation="https://scikit-plots.github.io/dev"
# LABEL org.opencontainers.image.source="https://github.com/scikit-plots/scikit-plots"
# LABEL org.opencontainers.image.url="https://hub.docker.com/r/scikitplot/scikit-plots"
# LABEL org.opencontainers.image.licenses="BSD-3-Clause"
# LABEL org.opencontainers.image.version="latest"
# LABEL org.opencontainers.image.revision="latest"

######################################################################
## default user jupyter ${NB_UID}
## cut -d: -f1 /etc/passwd
######################################################################
## Switch to the root user to install dependencies
USER root

## Fix DL4006
## Needed for string substitution in RUN commands
## Fix: https://github.com/hadolint/hadolint/wiki/DL4006
## Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
# -c Runs commands in a non-interactive shell, not a login shell.
# -lc Runs commands in a login shell (still non-interactive), so bash reads login startup files.
# RUN bash -lc 'set -Eeuo pipefail; \
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

######################################################################
## Set system locale and timezone
######################################################################

# ARG NB_USER="jovyan"
# ARG NB_UID="1000"
# ARG NB_GID="100"
## Configure environment
# ENV CONDA_DIR=/opt/conda \
#     SHELL=/bin/bash \
#     NB_USER="${NB_USER}" \
#     NB_UID=${NB_UID} \
#     NB_GID=${NB_GID} \
#     LC_ALL=C.UTF-8 \
#     LANG=C.UTF-8 \
#     LANGUAGE=C.UTF-8
# ENV PATH="${CONDA_DIR}/bin:${PATH}" \
#     HOME="/home/${NB_USER}"

## Set the timezone for the container to UTC+3 (Europe/Moscow)
## ENV TZ=Europe/Moscow or ENV TZ=Europe/Istanbul
# ARG TZ=Etc/UTC
# ENV TZ=$TZ
## Install tzdata non-interactively, set timezone, and clean up to reduce image size
# RUN apt-get update \
#   ## Generate timezone data
#   && apt-get install -y --no-install-recommends tzdata && \
#   ## Configure timezone (use default if $TZ is not set)
#   ln -snf /usr/share/zoneinfo/${TZ:-Etc/UTC} /etc/localtime && \
#   echo "${TZ:-Etc/UTC}" > /etc/timezone && \
#   dpkg-reconfigure -f noninteractive tzdata; \
#   ## Cleanup to reduce image size
#   apt-get clean \
#   && rm -rf /var/lib/apt/lists/*;

## Set the system locale to en_US.UTF-8 (English, UTF-8 encoding)
# ENV LANG=en_US.UTF-8 \
#     LC_ALL=en_US.UTF-8 \
#     LANGUAGE=en_US.UTF-8
## Install minimal locales support and generate C.UTF-8 locale
# RUN apt-get update \
#   ## Generate locales (this may be required depending on the system)
#   && apt-get install -y --no-install-recommends locales \
#   ## Generate en_US.UTF-8 locale
#   && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
#   && locale-gen \
#   && update-locale LANG=en_US.UTF-8 \
#   && dpkg-reconfigure -f noninteractive locales; \
#   ## Cleanup to reduce image size
#   apt-get clean; \
#   rm -rf /var/lib/apt/lists/*;

## Set the system locale to C.UTF-8 (POSIX-compliant with Unicode support)
## Ensures that all locale categories (LC_*) are set to C.UTF-8, so the environment is fully UTF-8 enabled.
## LANGUAGE is used on win32
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV LANGUAGE=C.UTF-8
## Install minimal locales support and generate C.UTF-8 locale
## echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
## echo "C.UTF-8 UTF-8" >> /etc/locale.gen && \
## locale-gen
# RUN apt-get update \
#   ## Generate locales (this may be required depending on the system)
#   ## "C" locale (basic settings) while supporting the full range of Unicode characters (UTF-8).
#   && apt-get install -y --no-install-recommends locales \
#   ## Generate C.UTF-8 locale
#   && locale-gen C.UTF-8 \
#   && dpkg-reconfigure -f noninteractive locales; \
#   ## Cleanup to reduce image size
#   apt-get clean; \
#   rm -rf /var/lib/apt/lists/*;

## Set the default user and group for the Jupyter notebook
## whoami   # e.g., vscode
## id -u    # 0 = root, anything else = non-root
## -eq -ne -gt -lt -ge -le
## [ "$(id -u)" -eq 0 ] "👑 Already root. Running apt directly..."
## [ "$(id -u)" -ne 0 ] "🔒 Not root. Switching to root with gosu..."
## ARG NB_USER=jovyan
## ARG NB_UID=1000
## ARG NB_GID=100
ARG NB_PASS=jovyan
# ENV HOME=/home/${NB_USER}

## Ensure sudo is installed and allow ${NB_USER} to use it without a password
RUN apt-get update; \
  ## Install sudo, gosu and set up a passwordless sudo user
  apt-get install -y --no-install-recommends sudo gosu \
  ## && useradd -m -s /bin/bash $NB_USER && \
  ## $$ useradd --no-log-init --create-home --shell /bin/bash --uid "${NB_UID}" --no-user-group "${NB_USER}"
  ## Set the password for ${NB_USER} to "jovyan" (or any other password you prefer)
  ## ⚠️ Note: This is not secure, consider using a more secure method for production
  ## echo "yourpassword" | sudo chown -R $(whoami):$(id -gn whoami) ~
  ## docker run -it --rm scikitplot/scikit-plots:latest bash -c "echo 'jovyan' | sudo -S apt update"
  ## RUN echo "jovyan:yourpassword" | chpasswd
  && echo "$NB_USER:$NB_PASS" | chpasswd \
  ## Requires password to sudo for ${NB_USER}
  && echo "${NB_USER} ALL=(ALL) ALL" | tee -a /etc/sudoers \
  ## Allow passwordless sudo for ${NB_USER}
  ## echo 'ALL ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/nopasswd
  ## && echo "${NB_USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers \
  ; \
  ## Cleanup to reduce image size
  apt-get clean; \
  rm -rf /var/lib/apt/lists/*;

######################################################################
## OS dependencies
## Copy the permissions-fix script and make it executable
# COPY "./docker/scripts/fix-permissions" /usr/local/bin/fix-permissions
# RUN chmod a+rx /usr/local/bin/fix-permissions
######################################################################

## If Time Zone Error
# RUN sed -i '/jammy-updates/s/^/#/' /etc/apt/sources.list \
#     && sed -i '/jammy-security/s/^/#/' /etc/apt/sources.list

## ── Common runtime packages ─ installed in BOTH runtime and devel images ─────────────────────
## tini is already provided by the jupyter/scipy-notebook base image — no need to reinstall.
RUN apt-get update --yes; \
  apt-get install -y --no-install-recommends \
    --fix-broken --allow-downgrades \
    ## CA certificates for secure HTTPS connections
    ca-certificates \
    ## Provides the add-apt-repository command (useful for adding PPAs)
    software-properties-common \
    ## Text editors and utilities
    vim \
    nano \
    wget \
    curl \
    procps \
    ## Git and related tools — needed even in runtime for `pip install git+…`
    git \
    git-lfs \
    ## Archive utilities (bzip2 also needed to extract the micromamba executable)
    bzip2 \
    gzip \
    p7zip-full \
    p7zip-rar \
    tar \
    unrar \
    unzip \
    ## Documentation tools
    ## pandoc is used to convert notebooks to HTML files
    pandoc \
    ## Ensures sharp, scalable fonts in PDFs instead of bitmap (raster) fonts
    cm-super \
    ## File and directory utilities
    tree \
    ## Graph visualization software: dot -V
    graphviz \
    ## Vector graphics editor
    inkscape \
    ## Image viewer / converter
    imagemagick \
    ## Multimedia tools for matplotlib animations
    ffmpeg \
    ## Minimal build support (Makefile invocations etc.) — tiny, useful even at runtime
    make \
    ## Python package installer for Python 3
    python3-pip \
    || { echo "Failed to install common dependencies" >&2; exit 1; }; \
  ## Cleanup to reduce image size
  apt-get clean; \
  rm -rf /var/lib/apt/lists/*;

## ── Devel-only compiler toolchain ─ installed ONLY when VARIANT=devel ───────────────────────
## Provides the full build stack needed to compile Python C/C++/Fortran extensions from source.
## runtime images skip this entirely so they stay lean.
RUN if [ "${VARIANT}" = "devel" ]; then \
    echo "VARIANT=devel: installing compiler toolchain…"; \
    apt-get update --yes; \
    apt-get install -y --no-install-recommends \
      ## installs gcc, g++, make, libc-dev — foundation for compiling native extensions
      ## see https://cython.readthedocs.io/en/latest/src/quickstart/install.html
      build-essential \
      ## Tool for querying compile-time flags for installed libraries
      pkg-config \
      ## Fast build system (used by CMake, meson, etc.)
      ninja-build \
      ## Compiler caching — speeds up recompilation of C/C++ code
      ccache \
      ## Cross-platform build system generator
      cmake \
      ## Clang compiler suite (llvm)
      clang \
      ## Automatic C/C++ code formatter
      clang-format \
      ## Static analysis for C/C++
      clang-tidy \
      ## GNU C compiler (also pulled in via build-essential, explicit for clarity)
      gcc \
      ## GNU C++ compiler
      g++ \
      ## GNU Fortran compiler — required by scipy, numpy, etc.
      gfortran \
      || { echo "Failed to install devel compiler toolchain" >&2; exit 1; }; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*; \
  else \
    echo "VARIANT=runtime: skipping compiler toolchain."; \
  fi

## AdditinL OS AND Python packages (e.g. for GPU support, etc.)
ARG EXTENDED_PACKAGES
ENV EXTENDED_PACKAGES=${EXTENDED_PACKAGES:-false}

## Add a second conditional RUN block for large or optional packages:
RUN set -Eeuo pipefail; \
  if [ "$EXTENDED_PACKAGES" = "true" ]; then \
    echo "Installing extended packages..."; \
    ## GPU-Based Parallel Computing
    ## CUDA – NVIDIA's framework for parallel programming on GPUs.
    ## OpenCL – Open standard for parallel programming on heterogeneous platforms (CPUs, GPUs, FPGAs).
    ## ROCm – AMD’s alternative to CUDA for GPU acceleration.
    ## Vulkan (MoltenVK) – Alternative to OpenGL & CUDA
    ## Metal Performance Shaders (MPS) – Apple’s GPU Framework
    ## TensorFlow with Metal Backend (ML-Specific)
    apt-get update --yes; \
    apt-get install --yes --no-install-recommends \
      ## Linear algebra and Numerical Libraries
      ## Basic Linear Algebra Subprograms (BLAS)
      libblas-dev \
      ## OpenBLAS (OpenBLAS Linear Algebra Library) - optimized BLAS library
      libopenblas-dev \
      ## BLIS (BLAS-like Library Instantiation Software) - an optimized BLAS library
      libblis-dev \
      ## LAPACK (Linear Algebra PACKage) - library for linear algebra operations
      liblapack-dev \
      ## CPU-Based Parallel Computing (Multithreading & SIMD)
      ## Intel's Parallel Computing C++ Libraries
      ## OpenMP - Open standard for parallel programming on multi-core CPUs.
      libomp-dev \
      ## General-purpose C++ development
      ## TBB (Threading Building Blocks) - Intel's Threading Building Blocks for task-based parallelism in C++.
      libtbb-dev \
      ## Boost C++ Libraries - a set of portable C++ source libraries
      libboost-all-dev \
    || { echo "Failed to install extended packages"; }; \
    ## Install separately and ignore failure
    ## MKL (Intel Math Kernel Library) - Optimized math library for scientific computing with multi-threading support.
    apt-get install --yes --no-install-recommends \
      libmkl-dev || true; \
    ## Cleanup to reduce image size
    apt-get clean || true; \
    rm -rf /var/lib/apt/lists/* || true; \
  else \
    echo "Skipping extended packages."; \
  fi

## Switch back to the notebook default user "jovyan" to avoid accidental container runs as root
USER ${NB_UID}
# RUN sudo chown -R jovyan .
# COPY --chown="${NB_UID}:${NB_GID}" initial-condarc "${CONDA_DIR}/.condarc"
# Add an R mimetype option to specify how the plot returns from R to the browser
# COPY --chown=${NB_UID}:${NB_GID} Rprofile.site /opt/conda/lib/R/etc/

######################################################################
## WORKDIR
# RUN echo $HOME && whoami
######################################################################
## Set working directory inside the container
WORKDIR /tmp

COPY requirements ./requirements/
COPY docker ./docker/
# Install minimal due to disc size limit
## docker/env_conda/{environment.base.yml,environment.yml,environment.devel.yml} are
## already present at their real paths via 'COPY docker ./docker/' above.
## $ENV_FILE_BASE / $ENV_FILE_RUNTIME / $ENV_FILE_DEVEL point to them directly;
## no flat-copy alias needed.

ARG TARGETARCH
## ── Post-create env contract ──────────────────────────────────────────────────────────────
## The following ENV vars (promoted via ENV earlier) are available to all_post_create.sh
## as real shell environment variables — no Dockerfile changes needed when the script changes:
##
##   VARIANT          runtime | devel   → script decides which yml layers to apply
##   ENV_DIR          docker/env_conda
##   ENV_FILE_BASE    docker/env_conda/environment.base.yml   → always applied
##   ENV_FILE_RUNTIME docker/env_conda/environment.yml        → always applied
##   ENV_FILE_DEVEL   docker/env_conda/environment.devel.yml  → applied only when VARIANT=devel
##   ENV_NAME         e.g. py312  → conda env name (passed as --name)
##
## NOTE: TARGETARCH is only auto-injected by Docker BuildKit when --platform is passed
## (e.g., docker buildx build --platform linux/amd64). CI systems like Red Hat Quay
## do not inject it automatically. Fall back to `uname -m` when unset.
## Handle both Docker names (amd64/arm64) and Linux kernel names (x86_64/aarch64).
## Naming scheme                         amd64  arm64  x86_64  aarch64
## Docker/BuildKit TARGETARCH            ✅    ✅     ✗       ✗
## Linux uname -m                        ✗     ✗      ✅      ✅
## Red Hat Quay (no BuildKit injection)  ✗     ✗      ✅      ✅
RUN set -Eeuo pipefail; \
  if [ "${RUN_POST_CREATE}" = "1" ]; then \
    _arch="${TARGETARCH:-$(uname -m)}"; \
    case "${_arch}" in \
      amd64 | x86_64) export MICROMAMBA_API_PLATFORM=linux-64 ;; \
      arm64 | aarch64) export MICROMAMBA_API_PLATFORM=linux-aarch64 ;; \
      *) echo "Unsupported arch=${_arch} (TARGETARCH=${TARGETARCH:-unset})" >&2; exit 2 ;; \
    esac; \
    echo "Resolved arch=${_arch} → MICROMAMBA_API_PLATFORM=${MICROMAMBA_API_PLATFORM}"; \
    \
    POST_CREATE_STRICT=0 \
    POST_CREATE_ENV_REQUIRED=1 \
    POST_CREATE_RUN_FIRST_NOTICE=1 \
    POST_CREATE_RUN_POST_CREATE=1 \
    \
    POST_CREATE_RUN_CONDA=1 \
    POST_CREATE_RUN_MICROMAMBA=0 \
    POST_CREATE_PACKAGE_EXTRAS="" \
    \
    bash docker/scripts/all_post_create.sh; \
    { apt-get clean || true; }; \
    { python -m pip cache purge >/dev/null 2>&1 || true; }; \
    { conda clean --all -f -y || true; }; \
    { mamba clean --all -f -y || true; }; \
    { micromamba clean --all -f -y || true; }; \
    { jupyter lab clean || true; }; \
    { rm -rf /var/lib/apt/lists/* || true; }; \
    { rm -rf /tmp/* /var/tmp/* || true; }; \
    { \
      rm -rf ${HOME}/.cache/* || true; \
      rm -rf "${HOME}/.cache/pip" \
        "${HOME}/.cache/wheels" \
        "${HOME}/.cache/matplotlib" \
        "${HOME}/.cache/torch" \
        "${HOME}/.cache/huggingface" \
        "${HOME}/.cache/uv" \
        "${HOME}/.cache/pypoetry" \
        "${HOME}/.conda/pkgs" || true; }; \
  fi

######################################################################
## default environment (e.g. conda, mamba, pipenv)
######################################################################
## Install JupyterHub, JupyterLab, NBClassic and Jupyter Notebook
## Generate a Jupyter Server config
## Cleanup temporary files
## Correct permissions
## Do all this in a single RUN command to avoid duplicating all of the
## files across image layers when the permissions change
##
## Optionally, Ensure mamba as conda installations (if needed)
## RUN conda install -n base conda-forge::mamba -y
## RUN conda install -n base -c conda-forge mamba -y
## RUN mamba create -n "${ENV_NAME}" python=${PYTHON_VERSION} ipykernel -y
##
## ── Three-file layered install via $ENV_NAME and $ENV_FILE_* ─────────────────────────────
## These are superseded by all_post_create.sh (invoked above) which reads VARIANT and
## applies the correct yml layers automatically.  Left here as a manual-run reference.
## Actual file names:
##   base    → $ENV_FILE_BASE    = docker/env_conda/environment.base.yml
##   runtime → $ENV_FILE_RUNTIME = docker/env_conda/environment.yml        (always)
##   devel   → $ENV_FILE_DEVEL   = docker/env_conda/environment.devel.yml  (devel only)
##
# RUN { mamba env create -n "${ENV_NAME}" -f "${ENV_FILE_BASE}" || echo "Failed: base env"; } \
#   && { mamba env update -n "${ENV_NAME}" -f "${ENV_FILE_RUNTIME}" || echo "Failed: runtime env"; } \
#   && if [ "${VARIANT}" = "devel" ]; then \
#        mamba env update -n "${ENV_NAME}" -f "${ENV_FILE_DEVEL}" || echo "Failed: devel env"; \
#      fi \
#   && { mamba clean --all -f -y || true; } \
#   && { jupyter lab clean || true; } \
#   && { rm -rf ${HOME}/.cache || true; }

## Fix permissions on "$HOME" as jovyan
RUN { fix-permissions "/home/${NB_USER}" && fix-permissions "${CONDA_DIR}" || true; };

######################################################################
## JUPYTER
######################################################################

FROM base AS jupyter

## Set environment variable with a default port, allowing override
ENV JUPYTER_PORT=${JUPYTER_PORT:-8888}
## Expose the Jupyter notebook's default port
## Default value is 8888, can be overridden
EXPOSE ${JUPYTER_PORT:-8888}
EXPOSE 8888-8890
EXPOSE 5000-5001
EXPOSE 9000-9001

# Copy local files as late as possible to avoid cache busting
# COPY run-hooks.sh start.sh /usr/local/bin/
# COPY start-notebook.py start-notebook.sh start-singleuser.py start-singleuser.sh /usr/local/bin/

## Configure container entrypoint
## "tini" — the tini binary (often installed to /usr/bin/tini)
## -g — makes tini adopt grandchild processes too (ensures full reaping of orphaned zombies)
## -- — standard POSIX way to signal the end of options
# ENTRYPOINT ["tini", "-g", "--"]
# "start.sh" — your actual container start script (e.g. to start Jupyter, Conda env, etc.)
ENTRYPOINT ["tini", "-g", "--", "start.sh"]
## Configure container startup
## start-notebook.py reads JUPYTER_PORT from the environment automatically (ENV JUPYTER_PORT is set).
## Using exec-form without shell means variable expansion does NOT happen — drop the --port arg.
# CMD bash -c 'start-notebook.py --port "${JUPYTER_PORT:-8888}"'
CMD ["start-notebook.py"]

## Set the command to start Jupyter lab with the specified port
## CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]
## CMD bash -c 'source ~/.bash-bashrc 2>/dev/null || true; exec start-notebook.py --port "${JUPYTER_PORT:-8888}"'
## CMD bash -c '[ -f ~/.bash-bashrc ] && source ~/.bash-bashrc; exec start-notebook.py --port "${JUPYTER_PORT:-8888}"'

######################################################################
## WORKDIR jupyter "${HOME}"
# RUN echo $HOME && whoami
# Copy your project files to leverage Docker cache under (${HOME}, ~${HOME}/work, /work or /workspace)
######################################################################

USER ${NB_UID}

## Set working directory inside the container
WORKDIR "${HOME}"

COPY requirements ./requirements/
## Bring all three layered env yml files into $HOME for user convenience and reference.
## (The base stage already has them under /tmp/docker/env_conda/ — this makes them
##  immediately visible to the notebook user without needing to navigate to /tmp.)
COPY docker/env_conda/environment.base.yml ./
COPY docker/env_conda/environment.yml ./
COPY docker/env_conda/environment.devel.yml ./
COPY docker/env_pipenv/ ./env_pipenv/

# macOS Rosetta virtualization creates junk directory which gets owned by root further up.
# It'll get re-created, but as USER runner after the next directive so hopefully should not cause permission issues.
#
# More info: https://github.com/jupyter/docker-stacks/issues/2296
RUN rm -rf "/home/${NB_USER}/.cache/"

## Setup work directory for backward-compatibility (if needed)
## WORKDIR "${HOME}" "${HOME}/work" /home/$NB_USER/work
# WORKDIR "${HOME}/work"
