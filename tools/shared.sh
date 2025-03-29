#!/bin/bash

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

get_dep() {
    package="$1"
    version="$2"
    if [[ "$version" == "none" ]]; then
        # do not install with none
        echo
    elif [[ "${version%%[^0-9.]*}" ]]; then
        # version number is explicitly passed
        echo "$package==$version"
    elif [[ "$version" == "latest" ]]; then
        # use latest
        echo "$package"
    elif [[ "$version" == "min" ]]; then
        echo "$package==$(python scikitplot/_min_dependencies.py "$package")"
    fi
}

show_installed_libraries(){
    # use conda list when inside a conda environment. conda list shows more
    # info than pip list, e.g. whether OpenBLAS or MKL is installed as well as
    # the version of OpenBLAS or MKL
    if [[ -n "$CONDA_PREFIX" ]]; then
        conda list
    else
        python -m pip list
    fi
}

# activate_environment() {
#     if [[ "$DISTRIB" =~ ^conda.* ]]; then
#         # For Conda environments, use the correct command
#         if [[ -d "$CONDA_PREFIX" ]]; then
#             source "$CONDA_PREFIX/bin/activate" || { echo "Conda environment activation failed"; exit 1; }
#         else
#             source activate "$VIRTUALENV"
#         fi
#     elif [[ "$DISTRIB" == "ubuntu" || "$DISTRIB" == "debian-32" ]]; then
#         # For Virtualenv on Ubuntu/Debian, ensure the virtual environment exists
#         if [[ -d "$VIRTUALENV/bin" ]]; then
#             source "$VIRTUALENV/bin/activate" || { echo "Virtualenv activation failed"; exit 1; }
#         else
#             echo "Error: Virtualenv not found at $VIRTUALENV"
#             exit 1
#         fi
#     fi
# }

activate_environment() {
    if [[ "$DISTRIB" =~ ^conda.* ]]; then
        source activate "$VIRTUALENV"
    elif [[ "$DISTRIB" == "ubuntu" || "$DISTRIB" == "debian-32" ]]; then
        source "$VIRTUALENV/bin/activate"
    fi
}

create_conda_environment_from_lock_file() {
    ENV_NAME=$1
    LOCK_FILE=$2
    # Because we are using lock-files with the "explicit" format, conda can
    # install them directly, provided the lock-file does not contain pip solved
    # packages. For more details, see
    # https://conda.github.io/conda-lock/output/#explicit-lockfile
    lock_file_has_pip_packages=$(grep -q files.pythonhosted.org "$LOCK_FILE" && echo "true" || echo "false")
    if [[ "$lock_file_has_pip_packages" == "false" ]]; then
        conda create --name "$ENV_NAME" --file "$LOCK_FILE"
    else
        python -m pip install "$(get_dep conda-lock min)"
        conda-lock install --name "$ENV_NAME $LOCK_FILE"
    fi
}
