#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status
set -x  # Print each command before executing it

# Repairing wheel script
# OPENBLAS_DIR=$(python -c"import scipy_openblas32 as sop; print(sop.get_lib_dir())")
WHEEL="$1"      # Path to the built wheel file to repair
DEST_DIR="$2"   # Directory where the repaired wheel should be placed

# create a temporary directory in the destination folder and unpack the wheel
# into there
cwd=$PWD  # Save current working directory

pushd "$DEST_DIR"       # Enter the destination directory
mkdir -p tmp            # Create a temporary directory inside DEST_DIR
pushd tmp               # Enter tmp directory

# Unpack the wheel into the tmp directory
wheel unpack "$WHEEL"

# Find the unpacked directory (assumes only one starts with 'scikit_plots')
# pushd scikit_plots*  # Adjust this to the correct unpacked directory name
UNPACKED_DIR=$(find . -maxdepth 1 -type d -name 'scikit_plots*' | head -n 1)
if [[ -z "$UNPACKED_DIR" ]]; then
  echo "Error: unpacked directory matching 'scikit_plots*' not found!"
  exit 1
fi

pushd "$UNPACKED_DIR"   # Enter the unpacked wheel directory

# To avoid DLL hell, the file name of libopenblas that's being vendored with
# the wheel has to be name-mangled. delvewheel is unable to name-mangle PYD
# containing extra data at the end of the binary, which frequently occurs when
# building with mingw.
# We therefore find each PYD in the directory structure and strip them.

# To avoid DLL hell, strip all .pyd files to remove extra data that blocks delvewheel's name mangling
# Use find with -exec to safely handle filenames with spaces
# for f in $(find ./scikitplot* -name '*.pyd'); do strip "$f"; done
# find . -maxdepth 1 -type d -name 'scikitplot*' -exec find {} -name '*.pyd' -exec strip "{}" \; \;
find . -name '*.pyd' -exec strip "{}" \;

popd  # Exit unpacked directory

# Repack the wheel with the modifications and overwrite the original wheel file
wheel pack "$UNPACKED_DIR"
mv -fv *.whl "$WHEEL"   # Move the repacked wheel back to the original wheel path

# cd "$DEST_DIR"
popd  # Exit tmp directory

rm -rf tmp  # Clean up temporary unpack directory

# Python 3.8 resolves to delvewheel 1.10.0, while newer delvewheel
# releases require Python >=3.9. On Windows, run the Python 3.8
# repair process explicitly in UTF-8 mode to avoid locale/cp1252
# decoding failures when delvewheel patches Python source files.
# if python -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 8) else 1)'; then
#     log_info "Python 3.8 detected. Enabling UTF-8 mode for delvewheel."
#     export PYTHONUTF8=1
#     # python -X utf8 -c "import sys; print(sys.flags.utf8_mode)"
#     python -c "import sys; print(sys.flags.utf8_mode)"
# fi

# Run delvewheel to repair the wheel and add the OpenBLAS DLL path
# Note the space separating DEST_DIR and WHEEL arguments here
# the libopenblas.dll is placed into this directory in the cibw_before_build
# script.
# Run delvewheel to repair the wheel and add OpenBLAS path
# delvewheel repair --add-path $OPENBLAS_DIR --no-dll libsf_error_state.dll -w $DEST_DIR $WHEEL
# PYTHONUTF8=1 python -m delvewheel ... one command
# python -X utf8 -m delvewheel ...      one Python invocation
# delvewheel repair --add-path "$cwd/.openblas/lib" --no-dll libsf_error_state.dll -w "$DEST_DIR" "$WHEEL"
#
# Windows + Python 3.8  -> python -X utf8 -m delvewheel repair
# Windows + Python 3.9+ -> python -m delvewheel repair
# Non-Windows           -> python -m delvewheel repair
# repair
# ├── Windows / Python 3.8
# │   └── python -X utf8 -m delvewheel         correct
# │
# └── Windows / Python >=3.9
#     └── python -m delvewheel                 normal
if [[ "${RUNNER_OS:-}" == "Windows" ]] \
  && python -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 8))'
then
  echo "Windows + Python 3.8 detected: running delvewheel in UTF-8 mode"
  python -X utf8 -m delvewheel repair \
      --add-path "$cwd/.openblas/lib" \
      --no-dll libsf_error_state.dll \
      -w "$DEST_DIR" \
      "$WHEEL"
else
  python -m delvewheel repair \
      --add-path "$cwd/.openblas/lib" \
      --no-dll libsf_error_state.dll \
      -w "$DEST_DIR" \
      "$WHEEL"
fi

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
