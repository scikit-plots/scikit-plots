set -xe

PROJECT_DIR="${1:-$PWD}"


# remove any cruft from a previous run
rm -rf build

# Update license
echo "" >> $PROJECT_DIR/LICENSE
echo "----" >> $PROJECT_DIR/LICENSE
echo "" >> $PROJECT_DIR/LICENSE

# cat $PROJECT_DIR/LICENSES/ >> $PROJECT_DIR/LICENSES/
echo "$PROJECT_DIR/LICENSES/"; ls -R $PROJECT_DIR/LICENSES/ | sed '1d' | sed 's/^\(.*\)\/$/|-- \1/' | sed 's/^\([^|]\)/|-- \1/'
echo "" >> $PROJECT_DIR/LICENSE
echo "----" >> $PROJECT_DIR/LICENSE
echo "" >> $PROJECT_DIR/LICENSE

if [[ $RUNNER_OS == "Linux" ]] ; then
    cat $PROJECT_DIR/tools/wheels/LICENSE_linux.txt >> $PROJECT_DIR/LICENSE.txt
elif [[ $RUNNER_OS == "macOS" ]]; then
    cat $PROJECT_DIR/tools/wheels/LICENSE_osx.txt >> $PROJECT_DIR/LICENSE.txt
elif [[ $RUNNER_OS == "Windows" ]]; then
    cat $PROJECT_DIR/tools/wheels/LICENSE_win32.txt >> $PROJECT_DIR/LICENSE.txt
fi

# Check if the system is 32-bit or 64-bit and disable OpenBLAS for 32-bit wheels
if [[ $(python -c "import sys; print(sys.maxsize)") < $(python -c "import sys; print(2**33)") ]]; then
    echo "No BLAS used for 32-bit wheels"
    export INSTALL_OPENBLAS=false
elif [ -z "$INSTALL_OPENBLAS" ]; then
    export INSTALL_OPENBLAS=true
fi

# Install OpenBLAS from scipy-openblas64 if required
if [[ "$INSTALL_OPENBLAS" = "true" ]] ; then
    echo "Setting up OpenBLAS"
    echo PKG_CONFIG_PATH $PKG_CONFIG_PATH
    PKG_CONFIG_PATH=$PROJECT_DIR/.openblas
    
    # Cleanup any previous OpenBLAS directories and create a fresh one
    rm -rf "$PKG_CONFIG_PATH"
    mkdir -p "$PKG_CONFIG_PATH"
    
    # Install required CI dependencies
    python -m pip install --upgrade -r requirements/ci_requirements.txt
    
    # Configure OpenBLAS using scipy_openblas64 and create pkg-config file
    if python -c "import scipy_openblas64" &>/dev/null; then
        python -c "import scipy_openblas64; print(scipy_openblas64.get_pkg_config())" > "$PKG_CONFIG_PATH/scipy-openblas.pc"
        
        # Copy the shared objects into the OpenBLAS directory
        # Copy the shared objects to a path under $PKG_CONFIG_PATH, the build
        # will point $LD_LIBRARY_PATH there and then auditwheel/delocate-wheel will
        # pull these into the wheel. Use python to avoid windows/posix problems
        python <<EOF
import os, scipy_openblas64, shutil
srcdir = os.path.join(os.path.dirname(scipy_openblas64.__file__), "lib")
shutil.copytree(srcdir, os.path.join("$PKG_CONFIG_PATH", "lib"))

# Handle macOS-specific shared objects (if applicable)
srcdir = os.path.join(os.path.dirname(scipy_openblas64.__file__), ".dylibs")
if os.path.exists(srcdir):  # macOS delocate
    shutil.copytree(srcdir, os.path.join("$PKG_CONFIG_PATH", ".dylibs"))
EOF
        # pkg-config scipy-openblas --print-provides
    else
        echo "scipy_openblas64 module not found. Skipping OpenBLAS configuration."
    fi
fi

# Windows-specific setup for delvewheel if the OS is Windows
if [[ $RUNNER_OS == "Windows" ]]; then
    # delvewheel is the equivalent of delocate/auditwheel for windows.
    echo "Installing delvewheel and wheel for Windows"
    python -m pip install delvewheel wheel
fi

# TODO: delete along with enabling build isolation by unsetting
# CIBW_BUILD_FRONTEND when numpy is buildable under free-threaded
# python with a released version of cython
# Handle Free-Threaded Python builds (if applicable)
FREE_THREADED_BUILD="$(python -c"import sysconfig; print(bool(sysconfig.get_config_var('Py_GIL_DISABLED')))")"
if [[ $FREE_THREADED_BUILD == "True" ]]; then
    echo "Free-threaded build detected, installing additional build dependencies"
    
    # Install build tools like Meson, Ninja, and Cython (via nightly if needed)
    python -m pip install meson-python ninja
    python -m pip install -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple cython || python -m pip install cython
fi

echo "Build environment setup complete."