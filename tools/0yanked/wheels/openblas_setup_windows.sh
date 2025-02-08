#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Install dependencies using Chocolatey
echo "Installing required dependencies..."
# Install make, llvm, and ninja only if they are not already installed
choco install --no-progress --skip-automatic -y make llvm ninja
# Check if Git is installed
git --version || winget install --id Git.Git -e --source winget

# Clone the OpenBLAS repository
echo "Cloning OpenBLAS repository..."
git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS

# Set build options for ARM64 (defaults to dynamic if BUILD_TYPE is not set)
BUILD_TYPE="${BUILD_TYPE:-dynamic}" # Use 'static' for static library builds
echo "Configuring $BUILD_TYPE build settings..."

# Detect the system architecture
ARCH=$(uname -m)

# Determine architecture and set targets
case "$arch" in
    i686|x86)
        echo "Detected 32-bit architecture."
        ARCH="i686"          # 32-bit architecture
        TARGET="GENERIC"     # Generic target for compatibility
        ;;
    x86_64|arm64)
        echo "Detected 64-bit architecture."
        ARCH="x86_64"        # 64-bit architecture (generic target)
        TARGET="GENERIC"     # Generic target for compatibility
        ;;
    *)
        echo "Unknown architecture detected. Defaulting to x86_64."
        ARCH="x86_64"        # Default to 64-bit
        TARGET="GENERIC"     # Generic target for compatibility
        ;;
esac
echo "Target architecture: $ARCH"
echo "Build target: $TARGET"

# Build OpenBLAS
echo "Building OpenBLAS for $ARCH architecture..."
# Redirect make output to a file
# make CC=clang-cl FC=flang-new AR="llvm-ar" RANLIB="llvm-ranlib" TARGET=$TARGET ARCH=$ARCH MAKE=make > build.log 2>&1
# Suppress make output completely
# make CC=clang-cl FC=flang-new AR="llvm-ar" RANLIB="llvm-ranlib" TARGET=$TARGET ARCH=$ARCH MAKE=make > /dev/null 2>&1
# Suppress standard output (stdout):
# make CC=clang-cl FC=flang-new AR="llvm-ar" RANLIB="llvm-ranlib" TARGET=$TARGET ARCH=$ARCH MAKE=make > /dev/null
# Suppress error output (stderr):
# make CC=clang-cl FC=flang-new AR="llvm-ar" RANLIB="llvm-ranlib" TARGET=$TARGET ARCH=$ARCH MAKE=make 2>/dev/null
# Limit the log output
make CC=clang-cl FC=flang-new AR="llvm-ar" RANLIB="llvm-ranlib" TARGET=$TARGET ARCH=$ARCH MAKE=make | grep -i 'error\|warning'


# Install the compiled libraries
echo "Installing OpenBLAS..."
make PREFIX=/usr/local install

# Verify installation
echo "OpenBLAS installed successfully."
