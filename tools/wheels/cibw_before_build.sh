#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status
set -x  # Print each command before executing it
echo "Running cibw_before_build.sh"
# remove any cruft from a previous run
rm -rf build

PROJECT_DIR="${1:-$PWD}"

# Update license
# cat $PROJECT_DIR/tools/wheels/LICENSE_win32.txt >> $PROJECT_DIR/LICENSE.txt
if [[ $RUNNER_OS == "Linux" ]] ; then
    cat $PROJECT_DIR/tools/wheels/LICENSE_linux.txt >> $PROJECT_DIR/LICENSE.txt
elif [[ $RUNNER_OS == "macOS" ]]; then
    cat $PROJECT_DIR/tools/wheels/LICENSE_osx.txt >> $PROJECT_DIR/LICENSE.txt
elif [[ $RUNNER_OS == "Windows" ]]; then
    cat $PROJECT_DIR/tools/wheels/LICENSE_win32.txt >> $PROJECT_DIR/LICENSE.txt
fi


# Check if the system is 32-bit or 64-bit to Install OpenBLAS
if [[ $(python -c "import sys; print(sys.maxsize)") < $(python -c "import sys; print(2**33)") ]]; then
    echo "32-bit wheels"
    # Install required CI dependencies with 32 bit Openblas
    python -m pip install -r requirements/ci32_requirements.txt
    python -c "import scipy_openblas32; print(scipy_openblas32.get_pkg_config())" > $PROJECT_DIR/scipy-openblas.pc
    export INSTALL_OPENBLAS64=false
elif [ -z "$INSTALL_OPENBLAS64" ]; then
    echo "64-bit wheels"
    export INSTALL_OPENBLAS64=true
fi

# Install OpenBLAS from scipy-openblas64 if required
if [[ "$INSTALL_OPENBLAS64" = "true" ]] ; then
    echo "Setting up OpenBLAS"
    echo PKG_CONFIG_PATH $PKG_CONFIG_PATH
    PKG_CONFIG_PATH=$PROJECT_DIR/.openblas
    
    # Cleanup any previous OpenBLAS directories and create a fresh one
    rm -rf "$PKG_CONFIG_PATH"
    mkdir -p "$PKG_CONFIG_PATH"
    
    # Install required CI dependencies with 32/64 bit Openblas
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
    python -m pip install -U --pre pip
    python -m pip install -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple numpy cython || python -m pip install cython
    # TODO: Remove meson installation from source once a new release
    # that includes https://github.com/mesonbuild/meson/pull/13851 is available
    python -m pip install git+https://github.com/mesonbuild/meson
    # python -m pip install git+https://github.com/serge-sans-paille/pythran
    python -m pip install ninja meson-python pybind11 pythran
fi

# if the OS is macos
if [[ $RUNNER_OS == "macOS" ]]; then

    PLATFORM=$(uname -m)
    echo $PLATFORM
    #########################################################################################
    # Install GFortran + OpenBLAS
    
    if [[ $PLATFORM == "x86_64" ]]; then
        #GFORTRAN=$(type -p gfortran-9)
        #sudo ln -s $GFORTRAN /usr/local/bin/gfortran
        # same version of gfortran as the openblas-libs
        # https://github.com/MacPython/gfortran-install.git
        curl -L https://github.com/isuruf/gcc/releases/download/gcc-11.3.0-2/gfortran-darwin-x86_64-native.tar.gz -o gfortran.tar.gz
      
        GFORTRAN_SHA256=$(shasum -a 256 gfortran.tar.gz)
        KNOWN_SHA256="981367dd0ad4335613e91bbee453d60b6669f5d7e976d18c7bdb7f1966f26ae4  gfortran.tar.gz"
        if [ "$GFORTRAN_SHA256" != "$KNOWN_SHA256" ]; then
            echo sha256 mismatch
            exit 1
        fi
      
        sudo mkdir -p /opt/
        # places gfortran in /opt/gfortran-darwin-x86_64-native. There's then
        # bin, lib, include, libexec underneath that.
        sudo tar -xv -C /opt -f gfortran.tar.gz
      
        # Link these into /usr/local so that there's no need to add rpath or -L
        for f in libgfortran.dylib libgfortran.5.dylib libgcc_s.1.dylib libgcc_s.1.1.dylib libquadmath.dylib libquadmath.0.dylib; do
          ln -sf /opt/gfortran-darwin-x86_64-native/lib/$f /usr/local/lib/$f
        done
        ln -sf /opt/gfortran-darwin-x86_64-native/bin/gfortran /usr/local/bin/gfortran
      
        # Set SDKROOT env variable if not set
        # This step is required whenever the gfortran compilers sourced from
        # conda-forge (built by isuru fernando) are used outside of a conda-forge
        # environment (so it mirrors what is done in the conda-forge compiler
        # activation scripts)
        export SDKROOT=${SDKROOT:-$(xcrun --show-sdk-path)}
    fi


    if [[ $PLATFORM == "arm64" ]]; then
        curl -L https://github.com/fxcoudert/gfortran-for-macOS/releases/download/12.1-monterey/gfortran-ARM-12.1-Monterey.dmg -o gfortran.dmg
        GFORTRAN_SHA256=$(shasum -a 256 gfortran.dmg)
        KNOWN_SHA256="e2e32f491303a00092921baebac7ffb7ae98de4ca82ebbe9e6a866dd8501acdf  gfortran.dmg"
      
        if [ "$GFORTRAN_SHA256" != "$KNOWN_SHA256" ]; then
            echo sha256 mismatch
            exit 1
        fi
      
        hdiutil attach -mountpoint /Volumes/gfortran gfortran.dmg
        sudo installer -pkg /Volumes/gfortran/gfortran.pkg -target /
        type -p gfortran
    fi



    lib_loc=$(python -c"import scipy_openblas32; print(scipy_openblas32.get_lib_dir())")
    # Use the libgfortran from gfortran rather than the one in the wheel
    # since delocate gets confused if there is more than one
    # https://github.com/scipy/scipy/issues/20852
    install_name_tool -change @loader_path/../.dylibs/libgfortran.5.dylib @rpath/libgfortran.5.dylib $lib_loc/libsci*
    install_name_tool -change @loader_path/../.dylibs/libgcc_s.1.1.dylib @rpath/libgcc_s.1.1.dylib $lib_loc/libsci*
    install_name_tool -change @loader_path/../.dylibs/libquadmath.0.dylib @rpath/libquadmath.0.dylib $lib_loc/libsci*
    
    codesign -s - -f $lib_loc/libsci*
fi

echo "Build environment setup complete."