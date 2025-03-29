#!/bin/bash

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

set -ex
######################################################################
## (All OS Platform) Handle Free-Threaded Python builds
######################################################################
# Function to handle free-threaded Python builds
handle_free_threaded_build() {
    log "Checking for free-threaded Python support..."
    # TODO: delete along with enabling build isolation by unsetting
    # CIBW_BUILD_FRONTEND when numpy is buildable under free-threaded
    # python with a released version of cython
    # Handle Free-Threaded Python builds (if applicable)
    # local FREE_THREADED_BUILD
    FREE_THREADED_BUILD=$(python -c "import sysconfig; print(bool(sysconfig.get_config_var('Py_GIL_DISABLED')))")
    if [[ $FREE_THREADED_BUILD == "True" ]]; then
        log "Free-threaded Python build detected. Installing additional build dependencies..."
        python -m pip install -U --pre pip
        python -m pip uninstall -y cython numpy
        python -m pip install -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple cython numpy || python -m pip install cython numpy
        # TODO: Remove meson installation from source once a new release
        # that includes https://github.com/mesonbuild/meson/pull/13851 is available
        python -m pip install git+https://github.com/mesonbuild/meson
        # python -m pip install git+https://github.com/serge-sans-paille/pythran
        python -m pip install meson-python ninja pybind11 pythran
    else
        log "No free-threaded Python build detected. Skipping additional dependencies."
    fi
}
# Install free-threaded Python dependencies if applicable
handle_free_threaded_build
