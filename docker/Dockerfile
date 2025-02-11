## Dockerfile
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

## Define the build argument with a default value
## Ubuntu 24.04 (noble)
## https://hub.docker.com/_/ubuntu/tags?page=1&name=noble
ARG BASE_IMAGE=jupyter/scipy-notebook:latest

## Use the argument to specify the base image
FROM ${BASE_IMAGE}
# ARG may specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)

# Needed for string substitution in RUN commands
# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set environment variable to avoid interactive prompts during builds
ENV DEBIAN_FRONTEND=noninteractive
# Set locale
ENV LANG C.UTF-8

# LABEL maintainer="Jupyter Project <jupyter@googlegroups.com>"
# ARG NB_USER="jovyan"
# ARG NB_UID="1000"
# ARG NB_GID="100"
## Switch to the root user to install dependencies
USER root
## Ensure sudo is installed and allow ${NB_USER} to use it without a password
RUN apt-get update \
  && apt-get install -y sudo gosu \
  # && echo "${NB_USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers \
  && echo "${NB_USER} ALL=(ALL) ALL" | tee -a /etc/sudoers \
  && echo "jovyan:jovyan" | chpasswd \
  ;

## If Time Zone Error
# RUN sed -i '/jammy-updates/s/^/#/' /etc/apt/sources.list \
#     && sed -i '/jammy-security/s/^/#/' /etc/apt/sources.list
## Install common dependencies
RUN apt-get update --yes \
  && apt-get install --yes --no-install-recommends \
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
    ## File and directory utilities
    ## Command-line utility for displaying directory structure
    tree \
    ## Graph visualization software
    ## Graph visualization software: dot -V
    graphviz \
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
    ## Linear algebra and numerical libraries
    ## Basic Linear Algebra Subprograms (BLAS)
    libblas-dev \
    ## OpenBLAS (OpenBLAS Linear Algebra Library) - optimized BLAS library
    libopenblas-dev \
    ## BLIS (BLAS-like Library Instantiation Software) - an optimized BLAS library
    libblis-dev \
    ## LAPACK (Linear Algebra PACKage) - library for linear algebra operations
    liblapack-dev \
    ## Parallel computing libraries
    ## OpenMP development libraries
    libomp-dev \
    ## TBB (Threading Building Blocks) - Intel’s C++ library for parallelism
    libtbb-dev \
    ## MKL (Intel Math Kernel Library) - Intel's high-performance library for math and linear algebra
    libmkl-dev \
    ## Boost C++ Libraries
    ## Boost (Boost C++ Libraries) - a set of portable C++ source libraries
    libboost-all-dev \
    ## Multimedia tools
    ## for matplotlib anim
    ffmpeg \
    ## Vector graphics editor
    inkscape \
    ## newlines equivalent to single semicolon ; on terminal or in shell script.
    || { echo "Failed to install common dependencies"; exit 1; } \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* \
  ;

## Create a root-owned folder and copy the script there only if it exists
# COPY --chown="${NB_UID}:${NB_GID}" "./script/install_gpu_nvidia_cuda.sh" "/script/install_gpu_nvidia_cuda.sh"
COPY "./docker/script/install_gpu_nvidia_cuda.sh" "/script/install_gpu_nvidia_cuda.sh"
# Use a wrapper script to check for file existence before running it, ([ -f file ] && ... || ...)
RUN bash -c ' \
  [ -f /script/install_gpu_nvidia_cuda.sh ] \
  && chmod +x /script/install_gpu_nvidia_cuda.sh \
  # && /script/install_gpu_nvidia_cuda.sh \
  || echo "Script not found, skipping..." \
  '

## Switch back to the notebook default user jovyan to avoid accidental container runs as root
USER ${NB_UID}

## Optionally, Additional configurations or installations (if needed)
# RUN conda install -n base conda-forge::mamba
# RUN mamba env create -f environment.yml && mamba activate skplt-env
RUN mamba install --yes \
    -c defaults \
    -c conda-forge \
    # -c nvidia \
    # cuda \
    ## jupyter extensions
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
  && fix-permissions "${CONDA_DIR}" \
  && fix-permissions "/home/${NB_USER}" \
  ## yarn used by Yarn, a JavaScript package manager, to store its cache
  && rm -rf "/home/${NB_USER}/.cache" "/home/${NB_USER}/.cache/yarn" \
  ;

## Set environment variable with a default port, allowing override
ENV JUPYTER_PORT=${JUPYTER_PORT:-8888}
## Expose the Jupyter notebook's default port
## Default value is 8888, can be overridden
EXPOSE ${JUPYTER_PORT:-8888}
EXPOSE 8888-8890
EXPOSE 5000-5001
EXPOSE 9000-9001

## Start the container
## Set the command to start Jupyter lab with the specified port
# CMD ["start-notebook.sh"]
CMD start-notebook.py --port ${JUPYTER_PORT}

## Setup work directory for backward-compatibility (if needed)
WORKDIR "${HOME}/work"
