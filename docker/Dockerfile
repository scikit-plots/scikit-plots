# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

## Dockerfile
## References: https://hub.docker.com/r/jupyter/base-notebook/dockerfile
## https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#image-relationships
## https://github.com/jupyter/docker-stacks/blob/main/images/base-notebook/Dockerfile#L58

## Use the official Jupyter Docker Stacks
## jupyter do not come with CUDA drivers or GPU support by default
## https://github.com/jupyter/docker-stacks/blob/main/images/docker-stacks-foundation/Dockerfile
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

## Ubuntu 20.04 (focal)
## https://hub.docker.com/_/ubuntu/?tab=tags&name=focal
## OS/ARCH: linux/amd64
## ARG ROOT_CONTAINER=ubuntu:focal-20200703@sha256:d5a6519d9f048100123c568eb83f7ef5bfcad69b01424f420f17c932b00dea76
ARG BASE_IMAGE=jupyter/scipy-notebook:latest

## Use the argument to specify the base image
FROM ${BASE_IMAGE}
# ARG may specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)

LABEL maintainer="The scikit-plots developers <scikit-plots.github.io/dev>"
# ARG NB_USER="jovyan"
# ARG NB_UID="1000"
# ARG NB_GID="100"

# Fix DL4006
# Needed for string substitution in RUN commands
# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

## Switch to the root user to install dependencies
USER root

## Set the desired locale to C with UTF-8 support
# ENV LANG en_US.UTF-8
## This sets the system language to C (the "POSIX" or "default" locale) with UTF-8 encoding,
## allowing it to handle all Unicode characters.
ENV LANG C.UTF-8
## Ensures that all locale categories (LC_*) are set to C.UTF-8,
## so the environment is fully UTF-8 enabled.
ENV LC_ALL C.UTF-8
## LANGUAGE is used on win32
ENV LANGUAGE C.UTF-8

## Set environment variable to avoid interactive prompts during builds
ENV DEBIAN_FRONTEND=noninteractive

## Install all OS dependencies for notebook server that starts but lacks all
## features (e.g., download as all possible file formats)
## Ensure sudo is installed and allow ${NB_USER} to use it without a password
RUN apt-get update \
  ## Generate locales (this may be required depending on the system)
  ## "C" locale (basic settings) while supporting the full range of Unicode characters (UTF-8).
  && apt-get install -y --no-install-recommends locales \
  && locale-gen C.UTF-8 \
  && dpkg-reconfigure locales \
  ## Install sudo and set up a passwordless sudo user
  && apt-get install -yq --no-install-recommends sudo gosu \
  # && echo "${NB_USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers \
  && echo "${NB_USER} ALL=(ALL) ALL" | tee -a /etc/sudoers \
  && echo "jovyan:jovyan" | chpasswd \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* \
  ;

# RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
#   locale-gen
## Copy a script that we will use to correct permissions after running certain commands
# COPY fix-permissions /usr/local/bin/fix-permissions
# RUN chmod a+rx /usr/local/bin/fix-permissions

## If Time Zone Error
# RUN sed -i '/jammy-updates/s/^/#/' /etc/apt/sources.list \
#     && sed -i '/jammy-security/s/^/#/' /etc/apt/sources.list
## Install common dependencies
RUN apt-get update --yes \
  && apt-get --fix-broken install --yes --no-install-recommends \
    --allow-downgrades \
    ## Common useful utilities
    ## Essential packages for building software
    ## for cython: https://cython.readthedocs.io/en/latest/src/quickstart/install.html
    build-essential \
    ## Provides the add-apt-repository command (useful for adding PPAs)
    software-properties-common \
    ## CA certificates for secure HTTPS connections
    ca-certificates \
    # zsh \
    # lsb-core \
    # lsb-release \
    ## Text editors and utilities
    vim \
    nano \
    wget \
    curl \
    ## Git and related tools
    git \
    git-lfs \
    ## Archive utilities
    unzip \
    gzip \
    ## bzip2 is necessary to extract the micromamba executable.
    bzip2 \
    p7zip-full \
    p7zip-rar \
    unrar \
    tar \
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
    ## Vector graphics editor
    inkscape \
    ## Graph visualization software
    ## Graph visualization software: dot -V
    graphviz \
    ## image viewer
    imagemagick \
    ## Build tools
    ## Utility for building and compiling software projects
    make \
    ## Cross-platform build system generator
    cmake \
    ## Build system that focuses on speed
    ninja-build \
    ## Tool for configuring compile-time flags for libraries
    pkg-config \
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
    ## General-purpose C++ development
    ## Boost C++ Libraries - a set of portable C++ source libraries
    libboost-all-dev \
    ## CPU-Based Parallel Computing (Multithreading & SIMD)
    ## Intel's Parallel Computing C++ Libraries
    ## OpenMP - Open standard for parallel programming on multi-core CPUs.
    libomp-dev \
    ## TBB (Threading Building Blocks) - Intel's Threading Building Blocks for task-based parallelism in C++.
    libtbb-dev \
    ## newlines equivalent to single semicolon ; on terminal or in shell script.
    || { echo "Failed to install common dependencies"; exit 1; } \
  ## Install separately and ignore failure
  ## MKL (Intel Math Kernel Library) - Optimized math library for scientific computing with multi-threading support.
  && apt-get install --yes --no-install-recommends libmkl-dev || true \
  ## GPU-Based Parallel Computing
  ## CUDA – NVIDIA's framework for parallel programming on GPUs.
  ## OpenCL – Open standard for parallel programming on heterogeneous platforms (CPUs, GPUs, FPGAs).
  ## ROCm – AMD’s alternative to CUDA for GPU acceleration.
  ## Vulkan (MoltenVK) – Alternative to OpenGL & CUDA
  ## Metal Performance Shaders (MPS) – Apple’s GPU Framework
  ## TensorFlow with Metal Backend (ML-Specific)
  && fix-permissions "$HOME" \
  && fix-permissions "$CONDA_DIR" \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* \
  ;

## Create a root-owned folder and copy the script there only if it exists
# COPY --chown="${NB_UID}:${NB_GID}" "./script/install_gpu_nvidia_cuda.sh" "/script/install_gpu_nvidia_cuda.sh"
COPY "./docker/script/install_gpu_nvidia_cuda.sh" "/script/install_gpu_nvidia_cuda.sh"
# RUN chmod a+rx "/script/install_gpu_nvidia_cuda.sh"
# Use a wrapper script to check for file existence before running it, ([ -f file ] && ... || ...)
RUN bash -c ' \
  [ -f /script/install_gpu_nvidia_cuda.sh ] \
  && chmod +x /script/install_gpu_nvidia_cuda.sh \
  # && /script/install_gpu_nvidia_cuda.sh \
  || echo "Script not found, skipping..." \
  '

## Switch back to the notebook default user jovyan to avoid accidental container runs as root
USER ${NB_UID}

# Setup work directory for backward-compatibility
RUN mkdir -p /home/$NB_USER/work || true && \
    fix-permissions /home/$NB_USER

## Optionally, Additional configurations or installations (if needed)
# RUN conda install -n base conda-forge::mamba
# RUN mamba env create -f environment.yml && mamba activate skplt-env
RUN mamba install --yes \
    -c defaults \
    -c conda-forge \
    # -c nvidia \
    # cuda \
    ## jupyter extensions
    'notebook>=6.0.3' \
    'jupyterhub>=1.1.0' \
    'jupyterlab>=2.1.5' \
    'jupyterlab-git' \
    'ipympl'\
    'ipywidgets' \
    'widgetsnbextension'\
    ## data manuplation
    'scipy' \
    'pandas' \
    'xlrd' \
    'openpyxl' \
    'pyarrow' \
    ## data visualiation
    'matplotlib-base' \
    'seaborn' \
    ## models
    'statsmodels' \
    'scikit-learn' \
    'catboost' \
    'xgboost' \
    ## model optimizations
    'optuna' \
    'hyperopt' \
    ## model saving
    'joblib' \
    'h5py' \
    ## model tracking
    'mlflow' \
    ## to generate a requirements.txt
    'pipreqs' \
    # testing
    'pytest' \
    'hypothesis' \
    # development
    'setuptools' \
    'wheel' \
    'build' \
    # C/C++ development
    'meson' \
    'ninja' \
    'meson-python' \
    'cython' \
    'pybind11' \
  ## Clean mamba and fix permissions
  && mamba clean --all -f -y || true \
  ## Fix permissions as jovyan
  # echo "yourpassword" | sudo chown -R $(whoami):$(id -gn whoami) ~
  && fix-permissions "/home/${NB_USER}" \
  && fix-permissions "${CONDA_DIR}" \
  ## yarn used by Yarn, a JavaScript package manager, to store its cache
  && rm -rf "/home/${NB_USER}/.cache" \
  && rm -rf "$CONDA_DIR/share/jupyter/lab/staging" \
  ;

## Set environment variable with a default port, allowing override
ENV JUPYTER_PORT=${JUPYTER_PORT:-8888}
## Expose the Jupyter notebook's default port
## Default value is 8888, can be overridden
EXPOSE ${JUPYTER_PORT:-8888}
EXPOSE 8888-8890
EXPOSE 5000-5001
EXPOSE 9000-9001

## Configure container startup
ENTRYPOINT ["tini", "-g", "--"]
# CMD ["start-notebook.sh"]
## Set the command to start Jupyter lab with the specified port
CMD start-notebook.py --port ${JUPYTER_PORT}

## Setup work directory for backward-compatibility (if needed)
WORKDIR "${HOME}/work"
