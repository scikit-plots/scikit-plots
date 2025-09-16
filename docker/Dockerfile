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
ARG BASE_IMAGE=jupyter/scipy-notebook:latest

## Use the argument to specify the base image
FROM ${BASE_IMAGE} AS base
## ⚠️ ARG may specified again because the FROM directive resets ARGs
## (but their default value is retained if set previously)

ARG SCIKITPLOT_VERSION=0.4.0.post3
ENV SCIKITPLOT_VERSION=${SCIKITPLOT_VERSION}

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

## Set environment variable to avoid interactive prompts during builds
## Prevent apt install tzinfo from asking our location (assumes UTC)
ENV DEBIAN_FRONTEND=noninteractive

## AdditinL OS AND Python packages (e.g. for GPU support, etc.)
ARG EXTENDED_PACKAGES=false
ENV EXTENDED_PACKAGES=${EXTENDED_PACKAGES:-false}

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
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

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
#   && rm -rf /var/lib/apt/lists/* \
#   ;

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
#   apt-get clean \
#   && rm -rf /var/lib/apt/lists/* \
#   ;

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
#   apt-get clean \
#   && rm -rf /var/lib/apt/lists/* \
#   ;

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
RUN apt-get update \
  ## Install sudo, gosu and set up a passwordless sudo user
  && apt-get install -y --no-install-recommends sudo gosu \
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
  apt-get clean \
  && rm -rf /var/lib/apt/lists/*

######################################################################
## OS dependencies
## Copy the permissions-fix script and make it executable
# COPY "./docker/scripts/fix-permissions" /usr/local/bin/fix-permissions
# RUN chmod a+rx /usr/local/bin/fix-permissions
######################################################################

## If Time Zone Error
# RUN sed -i '/jammy-updates/s/^/#/' /etc/apt/sources.list \
#     && sed -i '/jammy-security/s/^/#/' /etc/apt/sources.list

## Install common dependencies
RUN apt-get update --yes \
  && apt-get install -y --no-install-recommends \
    --fix-broken --allow-downgrades \
    ## CA certificates for secure HTTPS connections
    ca-certificates \
    ## Provides the add-apt-repository command (useful for adding PPAs)
    software-properties-common \
    # - `tini` is installed as a helpful container entrypoint,
    #   that reaps zombie processes and such of the actual executable we want to start
    #   See https://github.com/krallin/tini#why-tini for details
    # tini \
    ## Essential packages for building software
    ## for cython: https://cython.readthedocs.io/en/latest/src/quickstart/install.html
    ## installs gcc, g++, make, libc-dev for compiling native extensions
    build-essential \
    ## Tool for configuring compile-time flags for libraries
    pkg-config \
    ## Common useful utilities
    # zsh \
    # lsb-core \
    # lsb-release \
    ## Text editors and utilities
    vim \
    nano \
    wget \
    curl \
    procps \
    ## Git and related tools
    git \
    git-lfs \
    ## Archive utilities
    ## bzip2 is necessary to extract the micromamba executable.
    bzip2 \
    gzip \
    p7zip-full \
    p7zip-rar \
    tar \
    unrar \
    unzip \
    ## Documentation tools
    ## pandoc is used to convert notebooks to html files
    pandoc \
    ## Ensures that pdflatex generates sharp, scalable fonts in PDFs instead of bitmap (raster) fonts.
    cm-super \
    ## gives you all LaTeX packages
    # texlive-full \
    ## File and directory utilities
    ## Command-line utility for displaying directory structure
    tree \
    ## Multimedia tools for matplotlib animations
    ffmpeg \
    ## image viewer
    imagemagick \
    ## Build tools
    ## Utility for building and compiling software projects
    make \
    ## Cross-platform build system generator
    cmake \
    ## Build system that focuses on speed
    ninja-build \
    ## Python-related tools
    ## Python package installer for Python 3
    python3-pip \
    ## Python 3 virtual environments
    python3-venv \
    # Python development headers, useful for compiling certain packages
    python3-dev \
    ## Compilers and build dependencies
    ## compiler caching tool that speeds up recompilation of C/C++ code by caching previous compilation results.
    ccache \
    ## GNU Fortran compiler (required for some scientific libraries)
    gfortran\
    ## GNU C compiler, c/cpp 11.4.0 compiler version
    gcc \
    ## GNU C++ compiler
    g++ \
    ## Clang compiler
    ## llvm-17 clang-17
    clang \
    ## automatically formats C++ (and other)
    clang-format \
    ## static analysis of C++ code
    clang-tidy \
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
    ## Graph visualization software
    ## Graph visualization software: dot -V
    graphviz \
    ## newlines equivalent to single semicolon ; on terminal or in shell script.
    || { echo "Failed to install common dependencies"; exit 1; }; \
  ## Cleanup to reduce image size
  apt-get clean \
  && rm -rf /var/lib/apt/lists/*

## Add a second conditional RUN block for large or optional packages:
RUN if [ "$EXTENDED_PACKAGES" = "true" ]; then \
    echo "Installing extended packages..."; \
    ## GPU-Based Parallel Computing
    ## CUDA – NVIDIA's framework for parallel programming on GPUs.
    ## OpenCL – Open standard for parallel programming on heterogeneous platforms (CPUs, GPUs, FPGAs).
    ## ROCm – AMD’s alternative to CUDA for GPU acceleration.
    ## Vulkan (MoltenVK) – Alternative to OpenGL & CUDA
    ## Metal Performance Shaders (MPS) – Apple’s GPU Framework
    ## TensorFlow with Metal Backend (ML-Specific)
    apt-get update --yes && apt-get install --yes --no-install-recommends \
      ## General-purpose C++ development
      ## TBB (Threading Building Blocks) - Intel's Threading Building Blocks for task-based parallelism in C++.
      libtbb-dev \
      ## Boost C++ Libraries - a set of portable C++ source libraries
      libboost-all-dev \
      ## MKL (Intel Math Kernel Library) - Optimized math library for scientific computing with multi-threading support.
      # libmkl-dev \
      ## Graph visualization software
      ## Graph visualization software: dot -V
      graphviz \
      ## Vector graphics editor
      inkscape \
    || { echo "Failed to install extended packages"; }; \
    ## Install separately and ignore failure
    ## MKL (Intel Math Kernel Library)
    apt-get install --yes --no-install-recommends libmkl-dev || true; \
    ## Cleanup to reduce image size
    apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    ; \
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
COPY docker ./docker/
COPY docker/env_conda/environment.yml ./
COPY requirements ./requirements/

## micromamba and post create steps
# RUN bash -i -c ". ./docker/scripts/all_post_create.sh"
## Make sure there’s a space between the variable assignment and the . for sourcing.
RUN bash -i -c "SKIP_MICROMAMBA=${SKIP_MICROMAMBA:=true} . ./docker/scripts/all_post_create.sh"

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
## RUN mamba create -n "py312" python=3.11 ipykernel -y
# RUN { mamba env create -n "py312" -f "./docker/env_conda/base.yml" || echo "Failed to apply base environment"; } \
#   ## Clean mamba, If fails continue
#   && { mamba clean --all -f -y || true; } \
#   && { jupyter lab clean || true; } \
#   && { rm -rf "/home/${NB_USER}/.cache/yarn" || true; } \
#   && { rm -rf ${HOME}/.cache || true; }
## Update environment with default.yml (always applied), Fails and exits script cleanly
# RUN { mamba env update -n "py312" -f "./docker/env_conda/default.yml" || echo "Failed to apply default environment"; } \
#   ## Clean mamba, If fails continue
#   && { mamba clean --all -f -y || true; } \
#   && { jupyter lab clean || true; } \
#   && { rm -rf "/home/${NB_USER}/.cache/yarn" || true; } \
#   && { rm -rf ${HOME}/.cache || true; }
## Conditionally apply extended.yml (only if EXTENDED_PACKAGES=true), Fails and exits script cleanly
# RUN if [ "$EXTENDED_PACKAGES" = "true" ]; then \
#     echo "Installing extended packages..."; \
#     { mamba env update -n "py312" -f "./docker/env_conda/extended.yml" || echo "Failed to apply extended environment"; } \
#     ## Clean mamba, If fails continue
#     && { mamba clean --all -f -y || true; } \
#     && { jupyter lab clean || true; } \
#     && { rm -rf "/home/${NB_USER}/.cache/yarn" || true; } \
#     && { rm -rf ${HOME}/.cache || true; } \
#     ; \
#   else \
#     echo "Skipping extended packages."; \
#   fi

## Fix permissions on "$HOME" as jovyan
RUN { fix-permissions "/home/${NB_USER}" \
  && fix-permissions "${CONDA_DIR}"; } || true

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
# CMD bash -c 'start-notebook.py --port "${JUPYTER_PORT:-8888}"'
# CMD ["start-notebook.py"]
CMD ["start-notebook.py", "--port ${JUPYTER_PORT:-8888}"]

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
COPY docker/env_conda/environment.yml ./
COPY docker/env_pipenv/ ./env_pipenv/
COPY requirements ./requirements/

# macOS Rosetta virtualization creates junk directory which gets owned by root further up.
# It'll get re-created, but as USER runner after the next directive so hopefully should not cause permission issues.
#
# More info: https://github.com/jupyter/docker-stacks/issues/2296
RUN rm -rf "/home/${NB_USER}/.cache/"

## Setup work directory for backward-compatibility (if needed)
## WORKDIR "${HOME}" "${HOME}/work" /home/$NB_USER/work
# WORKDIR "${HOME}/work"
