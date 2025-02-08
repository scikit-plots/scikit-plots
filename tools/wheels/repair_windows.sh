#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status
set -x  # Print each command before executing it

# Repairing wheel script
WHEEL="$1"
DEST_DIR="$2"

# create a temporary directory in the destination folder and unpack the wheel
# into there
cwd=$PWD

pushd $DEST_DIR
mkdir -p tmp
pushd tmp
wheel unpack $WHEEL
pushd scikit_plots*  # Adjust this to the correct unpacked directory name

# To avoid DLL hell, the file name of libopenblas that's being vendored with
# the wheel has to be name-mangled. delvewheel is unable to name-mangle PYD
# containing extra data at the end of the binary, which frequently occurs when
# building with mingw.
# We therefore find each PYD in the directory structure and strip them.

for f in $(find ./scikit_plots* -name '*.pyd'); do strip $f; done


# now repack the wheel and overwrite the original
wheel pack .
mv -fv *.whl $WHEEL

cd $DEST_DIR
rm -rf tmp

# the libopenblas.dll is placed into this directory in the cibw_before_build
# script.
# Run delvewheel to repair the wheel and add OpenBLAS path
delvewheel repair --add-path $cwd/.openblas/lib -w $DEST_DIR $WHEEL
echo "Fixed libopenblas.dll at $cwd/.openblas/lib"

# # Detect architecture (32-bit or 64-bit)
# ARCHITECTURE=$(python -c "import platform; print(platform.architecture()[0])")
# if [[ "$ARCHITECTURE" == "32bit" ]]; then
#   echo "Detected 32-bit architecture"
#   OPENBLAS_DIR=$(python -c "import scipy_openblas32 as sop; print(sop.get_lib_dir())")
# elif [[ "$ARCHITECTURE" == "64bit" ]]; then
#   echo "Detected 64-bit architecture"
#   OPENBLAS_DIR=$(python -c "import scipy_openblas64 as sop; print(sop.get_lib_dir())")
# else
#   echo "Unknown architecture: $ARCHITECTURE"
#   exit 1
# fi

# # Show the OpenBLAS directory
# echo "OpenBLAS Directory: $OPENBLAS_DIR"

# # Run delvewheel to repair the wheel and add OpenBLAS path
# # the libopenblas.dll is placed into this directory in the cibw_before_build
# # script.
# delvewheel repair --add-path $OPENBLAS_DIR --no-dll libsf_error_state.dll -w $DEST_DIR $WHEEL
