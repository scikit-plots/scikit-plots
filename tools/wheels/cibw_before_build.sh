#!/bin/bash

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

######################################################################
## INFO Logic Functions
# is_true  Returns 0 (true) for values matching the truthy set: 1, true, True, yes, etc.
# is_false Returns 0 (true) if the value is not in the truthy set.
######################################################################
# Common Condition Operators
# Operator Meaning Example
# -eq	Equal                           	[ "$a" -eq "$b" ]
# -ne	Not equal                           [ "$a" -ne "$b" ]
# -gt	Greater than                        [ "$a" -gt "$b" ]
# -lt	Less than                          	[ "$a" -lt "$b" ]
# -ge	Greater than or equal to            [ "$a" -ge "$b" ]
# -le	Less than or equal to               [ "$a" -le "$b" ]
# ==	String equality                     [[ "$a" == "$b" ]]
# !=	String inequality                   [[ "$a" != "$b" ]]
# -z	String is empty                     [ -z "$a" ]
# -n	String is not empty                 [ -n "$a" ]
# -f	File exists and is a regular file   [ -f "$file" ]
# -d	File exists and is a directory      [ -d "$dir" ]
# is_true() {
#     case "$1" in
#         1|true|True|TRUE|y|Y|yes|Yes|YES|'1') return 0 ;;  # True values (return success)
#         0|false|False|FALSE|n|N|no|No|NO|'0') return 1 ;;  # False values (return failure)
#         *) return 1 ;;  # Default to false for anything else
#     esac
# }
# is_false() {
#     ! is_true "$1"  # Invert the result of is_true
# }
######################################################################
## RUNNING
######################################################################
set -e  # Exit immediately if a command exits with a non-zero status
set -o pipefail  # Ensure pipeline errors are captured
set -u  # Treat unset variables as an error
set -x  # Print each command before executing it

# Set environment variables to make our wheel build easier to reproduce byte
# for byte from source. See https://reproducible-builds.org/. The long term
# motivation would be to be able to detect supply chain attacks.
#
# In particular we set SOURCE_DATE_EPOCH to the commit date of the last commit.
#
# XXX: setting those environment variables is not enough. See the following
# issue for more details on what remains to do:
# https://github.com/scikit-learn/scikit-learn/issues/28151
SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
export SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH"
export PYTHONHASHSEED=0
######################################################################
## Define color and style variables for styled output.
######################################################################
# Colors for Pretty Logs
BOLD='\033[1m'
RESET='\033[0m'  # No Color
RED='\033[1;31m'
GREEN='\033[1;32m'
BLUE='\033[1;34m'
YELLOW='\033[1;33m'
# MAGENTA='\033[1;35m'
# CYAN='\033[1;36m'
######################################################################
## Logging functions for consistent output styling.
######################################################################
log_info()    { echo -e "$(date '+%Y-%m-%d %H:%M:%S') ${BOLD}${BLUE}[INFO]${RESET} $1"; }
log_success() { echo -e "$(date '+%Y-%m-%d %H:%M:%S') ${BOLD}${GREEN}[SUCCESS]${RESET} $1"; }
log_warning() { echo -e "$(date '+%Y-%m-%d %H:%M:%S') ${BOLD}${YELLOW}[WARNING]${RESET} $1"; }
log_error()   { echo -e "$(date '+%Y-%m-%d %H:%M:%S') ${BOLD}${RED}[ERROR]${RESET} $1" >&2; exit 1;}
######################################################################
## Utility Functions
######################################################################
# Function to clean up build artifacts
clean_build() {
    log_info "Cleaning up any previous build artifacts..."
    rm -rf build || log_error "Failed to remove build artifacts."
}
######################################################################
## (All OS Platform) Append OS LICENSE
######################################################################
# Function to handle LICENSE setup
setup_license() {
    log_info "Updating LICENSE file for $RUNNER_OS..."
    # Define project directory
    local project_dir="$1"
    # Define the license file based on OS
    local os_license_file
    case $RUNNER_OS in
        Linux)
            os_license_file="$project_dir/tools/wheels/LICENSE_linux.txt"
            ;;
        macOS)
            os_license_file="$project_dir/tools/wheels/LICENSE_osx.txt"
            ;;
        Windows)
            os_license_file="$project_dir/tools/wheels/LICENSE_win32.txt"
            ;;
        *)
            log_warning "Unknown OS: $RUNNER_OS. Skipping OS LICENSE update."
            return
            ;;
    esac
    # Check if the file exists before appending
    [ -f "$os_license_file" ] && stat -c "LICENSE size: %s bytes" "$os_license_file" || echo "LICENSE file not found"

    if [[ -f $os_license_file ]]; then
        cat "$os_license_file" >> "$project_dir/LICENSE.txt" || log_warning "Failed to append LICENSE.txt file."
        log_info "Appended $os_license_file to: $(find "$project_dir" -name "LICENSE.txt" -print -quit)"
    else
        log_warning "OS LICENSE file not found: $project_dir. Skipping OS LICENSE adding."
    fi

    [ -f "$project_dir/LICENSE.txt" ] && du -h "$project_dir/LICENSE.txt" || echo "LICENSE.txt file not found"
}
######################################################################
## (All OS Platform) Handle Free-Threaded Python builds
######################################################################
# Function to handle free-threaded Python builds
handle_free_threaded_build() {
    log_info "Checking for free-threaded Python support..."
    # TODO: delete along with enabling build isolation by unsetting
    # CIBW_BUILD_FRONTEND when numpy is buildable under free-threaded
    # python with a released version of cython
    # Handle Free-Threaded Python builds (if applicable)
    local FREE_THREADED_BUILD
    # Detect whether Python is a free-threaded (no-GIL) build using sysconfig
    FREE_THREADED_BUILD=$(python -c "import sysconfig; print(bool(sysconfig.get_config_var('Py_GIL_DISABLED')))")
    # ⚠️ DO NOT QUOTE the regex (right-hand side) inside [[ =~ ... ]]!
    # ✅ This regex will match both "True" and "true"
    # 📌 "$FREE_THREADED_BUILD" must be quoted to avoid globbing/splitting errors
    # ${VAR:=default} — Assign default if unset or null, avoiding the "unbound variable" error even when not set.
    # ${VAR:-default} — Use default if unset or null, but do NOT assign, if you only want to use a default value temporarily but keep the variable untouched.
    if [[ "$FREE_THREADED_BUILD" == ^[tT]rue$ ]]; then
        log_info "Free-threaded Python (GIL disabled) build detected. Installing additional build dependencies..."
        # 👉 Add no-GIL/free-threaded Python specific build flags or config here
        # python -m pip install -U --pre pip
        # python -m pip uninstall -y cython numpy
        # python -m pip install -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple cython numpy || python -m pip install cython numpy
        # # TODO: Remove meson installation from source once a new release
        # # that includes https://github.com/mesonbuild/meson/pull/13851 is available
        # python -m pip install git+https://github.com/mesonbuild/meson
        # # python -m pip install git+https://github.com/serge-sans-paille/pythran
        # python -m pip install meson-python ninja pybind11 pythran

        # Numpy, scipy, Cython only have free-threaded wheels on scientific-python-nightly-wheels
        # TODO: remove this after CPython 3.13 is released (scheduled October 2024)
        # and our dependencies have free-threaded wheels on PyPI
        export CIBW_BUILD_FRONTEND='pip; args: --pre --extra-index-url "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple" --only-binary :all:'  #  --no-build-isolation
    else
        log_info "No free-threaded Regular Python (GIL enabled) build detected. Skipping additional dependencies."
    fi
}
######################################################################
## (All OS Platform) Install Scipy OpenBLAS
######################################################################
# Function to configure the OpenBLAS PKG_CONFIG_PATH
configure_openblas_pkg_config() {
    local project_dir="$1"  # Take the project directory as input
    # Define the pkg_config_path
    PKG_CONFIG_PATH="$project_dir/.openblas"
    OPENBLAS_LIB_DIR="$project_dir/.openblas/lib"
    # Check if directory exists before attempting to delete it
    if [ -d "$PKG_CONFIG_PATH" ]; then
        log_info "Removing existing OpenBLAS config directory..."
        rm -rf "$PKG_CONFIG_PATH"  # Remove existing config directory
    else
        log_info "No existing OpenBLAS config directory to remove."
    fi
    # Create a new OpenBLAS directory and log the success
    log_info "Creating OpenBLAS config directory..."
    mkdir -p "$PKG_CONFIG_PATH"
    log_info "OpenBLAS config directory created successfully."
    # Export PKG_CONFIG_PATH based on the OS and log
    case $RUNNER_OS in
        Linux|macOS)
            export PKG_CONFIG_PATH="$PKG_CONFIG_PATH"  ;;
        Windows)
            # Adjust the path format for Windows
            PKG_CONFIG_PATH=$(echo "$PKG_CONFIG_PATH" | sed 's/\//\\/g')
            export PKG_CONFIG_PATH ;;
        *)
            export CIBW_ENVIRONMENT="PKG_CONFIG_PATH=$PKG_CONFIG_PATH"
            log_success "Setting CIBW_ENVIRONMENT to: $CIBW_ENVIRONMENT"
            ;;
    esac
    log_success "Setting PKG_CONFIG_PATH to: $PKG_CONFIG_PATH"
    # Export LIBRARY PATH based on the OS and log
    case $RUNNER_OS in
        Linux)
            # requires using safe expansions like ${VAR:-}
            if [ -n "${LD_LIBRARY_PATH:-}" ]; then
                export LD_LIBRARY_PATH="$OPENBLAS_LIB_DIR:$LD_LIBRARY_PATH"
            else
                export LD_LIBRARY_PATH="$OPENBLAS_LIB_DIR"
            fi
            log_success "Setting LD_LIBRARY_PATH to: $LD_LIBRARY_PATH"
            ;;
        macOS)
            ## ${VAR:+value}
            export DYLD_LIBRARY_PATH="$OPENBLAS_LIB_DIR${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
            log_success "Setting DYLD_LIBRARY_PATH to: $DYLD_LIBRARY_PATH"
            ;;
        Windows)
            # Adjust the path format for Windows
            OPENBLAS_LIB_DIR=$(echo "$OPENBLAS_LIB_DIR" | sed 's/\//\\/g')
            export PATH="$OPENBLAS_LIB_DIR${PATH:+;$PATH}"
            log_success "Setting PATH to: $PATH"
            ;;
        *)
            export CIBW_ENVIRONMENT="LD_LIBRARY_PATH=$OPENBLAS_LIB_DIR"
            log_success "Setting CIBW_ENVIRONMENT to: $CIBW_ENVIRONMENT"
            ;;
    esac
}
# Function to handle Scipy OpenBLAS setup
install_requirements() {
    # Define the requirements file based on Platform
    local requirements_file="$1"
    log_info "Installing Python requirements from $requirements_file..."
    python -m pip install -U pip -r "$requirements_file" \
        || log_error "Failed to install requirements."
}
generate_openblas_pkgconfig() {
    # Define the Scipy OpenBLAS based on Platform
    local openblas_module="$1"
    # Generate OpenBLAS pkg-config file based on Platform
    log_info "Generating OpenBLAS pkg-config file using $openblas_module..."
    python -c "import $openblas_module; print($openblas_module.get_pkg_config())" > "$PKG_CONFIG_PATH/scipy-openblas.pc" \
        || log_error "Failed to generate pkg-config."
    log_success "Defined scipy-openblas to: $PKG_CONFIG_PATH/scipy-openblas.pc"
}
copy_shared_libs() {
    # Copy Scipy OpenBLAS shared libraries to the build directory
    # Copy the shared objects to a path under $PKG_CONFIG_PATH, the build
    # will point $LD_LIBRARY_PATH there and then auditwheel/delocate-wheel will
    # pull these into the wheel. Use python to avoid windows/posix problems
    # Define the Scipy OpenBLAS based on Platform
    local openblas_module="$1"
    log_info "Copying shared libraries for $openblas_module..."
    python <<EOF
import os, shutil, $openblas_module
srcdir = os.path.join(os.path.dirname($openblas_module.__file__), "lib")
shutil.copytree(srcdir, os.path.join("$PKG_CONFIG_PATH", "lib"))
srcdir = os.path.join(os.path.dirname($openblas_module.__file__), ".dylibs")
if os.path.exists(srcdir):  # macOS delocate
    shutil.copytree(srcdir, os.path.join("$PKG_CONFIG_PATH", ".dylibs"))
EOF
    # Check for pkg-config availability
    log_info "Verifying OpenBLAS pkg-config file..."
    if ! [ -x "$(command -v pkg-config)" ]; then
        log_info "pkg-config not found. Attempting to manually set OpenBLAS library paths."
    else
        log_info "pkg-config found. Verifying OpenBLAS detection..."
        pkg-config --libs scipy-openblas --print-provides || log_error "Failed to find OpenBLAS with pkg-config."
    fi
    log_success "$openblas_module setup completed successfully."
}
setup_openblas() {
    log_info "Starting Python Scipy OpenBLAS setup..."
    # Define project directory
    local project_dir="$1"
    # Detect the system architecture
    local arch
    arch=$(uname -m)
    arch=$(echo "$arch" | tr '[:upper:]' '[:lower:]')
    log_info "Running on platform: $RUNNER_OS (Architecture: $arch)"
    # Check and set INSTALL_OPENBLAS if not already set, initialized as an empty string
    # set +u  # Temporarily disable unbound variable check
    # set -u  # Re-enable unbound variable check
    INSTALL_OPENBLAS="${INSTALL_OPENBLAS:-}"
    if [[ -z "$INSTALL_OPENBLAS" ]]; then
        log_info "INSTALL_OPENBLAS is not set. Setting INSTALL_OPENBLAS=true."
        INSTALL_OPENBLAS=true
        export INSTALL_OPENBLAS
    else
        # log_info INSTALL_OPENBLAS is set or not
        log_info "INSTALL_OPENBLAS is already set: $INSTALL_OPENBLAS"
    fi
    # Skip setup if INSTALL_OPENBLAS is not enabled
    if [[ "$INSTALL_OPENBLAS" != "true" ]]; then
        log_info "INSTALL_OPENBLAS is disabled. Skipping setup."
        return
    fi
    # Configure the PKG_CONFIG_PATH
    configure_openblas_pkg_config "$project_dir"
    # Determine architecture and install the appropriate requirements
    # converts the value of $arch to lowercase.
    arch=$(echo "$arch" | tr '[:upper:]' '[:lower:]')
    case "$arch" in
        i686|x86)
            log_info "Detected 32-bit architecture."
            # Install CI 32-bit specific requirements and generate OpenBLAS pkg-config file
            install_requirements "requirements/ci32_requirements.txt"
            generate_openblas_pkgconfig "scipy_openblas32"
            copy_shared_libs "scipy_openblas32"
            ;;
        x86_64|amd64|*arm64*|aarch64)
            log_info "Detected 64-bit architecture."
            # Install CI 64-bit specific requirements and generate OpenBLAS pkg-config file
            install_requirements "requirements/ci_requirements.txt"
            generate_openblas_pkgconfig "scipy_openblas64"
            copy_shared_libs "scipy_openblas64"
            ;;
        *)
            log_info "Unknown architecture detected: $arch."
            log_error "Unable to proceed. Exiting..."
            ;;
    esac
}
######################################################################
## Windows
######################################################################
# Function to handle Windows-specific setup
setup_windows() {
    if [[ $RUNNER_OS == "Windows" ]]; then
        log_info "Windows platform detected. Installing delvewheel and wheel..."
        # delvewheel is the equivalent of delocate/auditwheel for windows.
        python -m pip install delvewheel wheel
    fi
}
######################################################################
## macOS-specific setup GFortran + OpenBLAS
######################################################################
# Function to handle macOS-specific setup
setup_macos() {
    # $(uname) == "Darwin"
    # Detect the system architecture
    local arch
    arch=$(uname -m)
    if [[ $RUNNER_OS == "macOS" ]]; then
        log_info "macOS platform detected: $arch"
        if [[ $arch == "x86_64" ]]; then
            log_info "Setting up GFortran for x86_64..."
            # GFORTRAN=$(type -p gfortran-9)
            # sudo ln -s $GFORTRAN /usr/local/bin/gfortran
            # same version of gfortran as the openblas-libs
            # https://github.com/MacPython/gfortran-install.git
            # Download and install gfortran for x86_64
            curl -L https://github.com/isuruf/gcc/releases/download/gcc-11.3.0-2/gfortran-darwin-x86_64-native.tar.gz -o gfortran.tar.gz
            GFORTRAN_SHA256=$(shasum -a 256 gfortran.tar.gz)
            KNOWN_SHA256="981367dd0ad4335613e91bbee453d60b6669f5d7e976d18c7bdb7f1966f26ae4  gfortran.tar.gz"
            if [ "$GFORTRAN_SHA256" != "$KNOWN_SHA256" ]; then
                log_info "SHA256 mismatch for gfortran tarball"
                exit 1
            fi
            sudo mkdir -p /opt/
            # places gfortran in /opt/gfortran-darwin-x86_64-native. There's then
            # bin, lib, include, libexec underneath that.
            sudo tar -xv -C /opt -f gfortran.tar.gz
            # Link gfortran libraries and binaries
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
        elif [[ $arch == "arm64" ]]; then
            log_info "Setting up GFortran for ARM64..."
            # Download and install gfortran for ARM64
            curl -L https://github.com/fxcoudert/gfortran-for-macOS/releases/download/12.1-monterey/gfortran-ARM-12.1-Monterey.dmg -o gfortran.dmg
            GFORTRAN_SHA256=$(shasum -a 256 gfortran.dmg)
            KNOWN_SHA256="e2e32f491303a00092921baebac7ffb7ae98de4ca82ebbe9e6a866dd8501acdf  gfortran.dmg"
            if [ "$GFORTRAN_SHA256" != "$KNOWN_SHA256" ]; then
                log_info "SHA256 mismatch for gfortran DMG"
                exit 1
            fi
            hdiutil attach -mountpoint /Volumes/gfortran gfortran.dmg
            sudo installer -pkg /Volumes/gfortran/gfortran.pkg -target /
            type -p gfortran
        fi
        # Install OpenBLAS
        # Define the Scipy OpenBLAS based on Platform
        local openblas_module
        # Determine architecture and install the appropriate requirements
        case "$arch" in
            i686|x86)
                log_info "Detected 32-bit architecture."
                openblas_module="scipy_openblas32"
                ;;
            x86_64|*arm64*)
                log_info "Detected 64-bit architecture."
                openblas_module="scipy_openblas64"
                ;;
            *)  log_error "Unknown architecture detected: $arch. Unable to proceed. Exiting..." ;;
        esac
        # Fix library paths for macOS openblas_lib_dir
        lib_loc=$(python -c"import $openblas_module; print($openblas_module.get_lib_dir())")
        log_info "OpenBLAS path: $lib_loc"
        # Use the libgfortran from gfortran rather than the one in the wheel
        # since delocate gets confused if there is more than one
        # https://github.com/scipy/scipy/issues/20852
        # shellcheck disable=SC2086
        install_name_tool -change @loader_path/../.dylibs/libgfortran.5.dylib @rpath/libgfortran.5.dylib $lib_loc/libsci*
        # shellcheck disable=SC2086
        install_name_tool -change @loader_path/../.dylibs/libgcc_s.1.1.dylib @rpath/libgcc_s.1.1.dylib $lib_loc/libsci*
        # shellcheck disable=SC2086
        install_name_tool -change @loader_path/../.dylibs/libquadmath.0.dylib @rpath/libquadmath.0.dylib $lib_loc/libsci*

        # shellcheck disable=SC2086
        codesign -s - -f $lib_loc/libsci*
    fi
}
######################################################################
## Main Script
######################################################################
# Main function to orchestrate all steps
main() {
    printenv
    log_info "Starting build environment setup..."
    # Define project directory
    local project_dir="${1:-$PWD}"
    log_info "Project directory: $project_dir"
    # Clean up previous build artifacts
    clean_build
    # Append LICENSE file based on the OS
    setup_license "$project_dir"
    # Install free-threaded Python dependencies if applicable
    # TODO: These are no longer needed since Cython, Pythran and NumPy all have public releases that support free-threading now.
    # Not needed anymore, but leave commented out in case we need to start pulling
    # in a dev version of some dependency again.
    # https://github.com/scipy/scipy/pull/23180
    handle_free_threaded_build
    # Set up Scipy OpenBLAS based on architecture
    setup_openblas "$project_dir"
    # Windows-specific setup delvewheel
    setup_windows
    # macOS-specific setup GFortran + OpenBLAS
    setup_macos
    log_success "Build environment setup complete!"
}
# Execute the main function
main "$@"
