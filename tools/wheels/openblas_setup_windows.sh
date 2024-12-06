#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status
# Define the path to utils.sh (Ensure utils.sh exists in the same directory)
source ./utils; load_logs 

# Install dependencies using Chocolatey
echo "Installing required dependencies..."
choco install -y make llvm ninja git

# Clone the OpenBLAS repository
echo "Cloning OpenBLAS repository..."
git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS

# Set build options for ARM64 (defaults to dynamic if BUILD_TYPE is not set)
BUILD_TYPE="${BUILD_TYPE:-dynamic}" # Use 'static' for static library builds
echo "Configuring $BUILD_TYPE build settings..."

# Detect the system architecture
local arch
if [[ "$OSTYPE" == "msys" ]]; then
  # For Git Bash on Windows
  arch=$(wmic os get architecture | grep -o '[0-9]*')
else
  arch=$(uname -m)
fi

# Determine architecture and set targets
case "$arch" in
    i686|x86)
        log "Detected 32-bit architecture."
        ARCH="i686"          # 32-bit architecture
        TARGET="GENERIC"     # Generic target for compatibility
        ;;
    x86_64|arm64)
        log "Detected 64-bit architecture."
        ARCH="x86_64"        # 64-bit architecture (generic target)
        TARGET="GENERIC"     # Generic target for compatibility
        ;;
    *)
        log "Unknown architecture detected. Defaulting to x86_64."
        ARCH="x86_64"        # Default to 64-bit
        TARGET="GENERIC"     # Generic target for compatibility
        ;;
esac

# Build OpenBLAS
echo "Building OpenBLAS for $ARCH architecture..."
make CC=clang-cl FC=flang-new AR="llvm-ar" RANLIB="llvm-ranlib" TARGET=$TARGET ARCH=$ARCH MAKE=make

# Install the compiled libraries
echo "Installing OpenBLAS..."
make PREFIX=/usr/local install

# Verify installation
success "OpenBLAS installed successfully."