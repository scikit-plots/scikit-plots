#!/bin/bash

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# This script is used by .github/workflows/wheels.yml to run the full test
# suite, checks for license inclusion and that the openblas version is correct.
set -xe

PROJECT_DIR="$1"

python -m pip install threadpoolctl
python -c "import scikitplot; scikitplot.show_config()"

if [[ $RUNNER_OS == "Windows" ]]; then
    # GH 20391
    PY_DIR=$(python -c "import sys; print(sys.prefix)")
    mkdir "$PY_DIR/libs"
fi
if [[ $RUNNER_OS == "macOS"  && $RUNNER_ARCH == "X64" ]]; then
  # Not clear why this is needed but it seems on x86_64 this is not the default
  # and without it f2py tests fail
  export DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:/usr/local/lib"
  # Needed so gfortran (not clang) can find system libraries like libm (-lm)
  # in f2py tests
  export LIBRARY_PATH="$LIBRARY_PATH:/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib"
elif [[ $RUNNER_OS == "Windows" && $IS_32_BIT == true ]] ; then
  echo "Skip OpenBLAS version check for 32-bit Windows, no OpenBLAS used"
  # Avoid this in GHA: "ERROR: Found GNU link.exe instead of MSVC link.exe"
  rm /c/Program\ Files/Git/usr/bin/link.EXE
fi
# Set available memory value to avoid OOM problems on aarch64.
# See gh-22418.
export NPY_AVAILABLE_MEM="4 GB"


FREE_THREADED_BUILD="$(python -c "import sysconfig; print(bool(sysconfig.get_config_var('Py_GIL_DISABLED')))")"
if [[ $FREE_THREADED_BUILD == "True" ]]; then
    # TODO: delete when importing numpy no longer enables the GIL
    # setting to zero ensures the GIL is disabled while running the
    # tests under free-threaded python
    export PYTHON_GIL=0
fi

# Run full tests with -n=auto. This makes pytest-xdist distribute tests across
# the available N CPU cores: 2 by default for Linux instances and 4 for macOS arm64
# python -c "import sys; import scikitplot; sys.exit(not scikitplot.test(label='full', extra_argv=['-n=auto']))"
python -c "import sys; import scikitplot;"
python "$PROJECT_DIR/tools/wheels/check_license.py"
