#!/bin/bash

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

set -x  # Enable debugging
## newlines equivalent to single semicolon ; on terminal or in shell script.

## Installing NVIDIA...
if nvidia-smi > /dev/null 2>&1; then
  echo "GPU NVIDIA CUDA drivers found, skipping setup...";
elif [ "$ENABLE_GPU" = "true" ]; then
  echo "Running GPU NVIDIA CUDA drivers setup..."

  # Install necessary dependencies
  PKGS=(
    ca-certificates
    curl
    gnupg2
    lsb-core
    lsb-release
  )
  apt-get update \
  && apt-get install -y --no-install-recommends "${PKGS[@]}"

  ## Set NVARCH to match your architecture, e.g., "x86_64"
  ARCH="$(uname -m 2>/dev/null || echo 'x86_64')"
  NVARCH="${NVARCH:-$ARCH}"  # Fallback to x86_64 if empty
  ## Set the Ubuntu version for the repository (replace with your version if necessary)
  ## Fallback to Ubuntu if lsb_release fails
  LSB_VERSION="$(lsb_release -rs 2>/dev/null || echo '24.04')"
  UBUNTU_VERSION="${LSB_VERSION:-${UBUNTU_VERSION:-2404}}"
  ## Replace (tr, sed) dots, e.g., "22.04" -> "2204"
  # UBUNTU_VERSION="${UBUNTU_VERSION//./}"
  UBUNTU_VERSION="$(echo "$UBUNTU_VERSION" | tr -d '.')"
  echo "UBUNTU_VERSION is: $UBUNTU_VERSION"

  ## Add NVIDIA repository key and repo
  ## Pin the CUDA repository to ensure you use the latest driver
  curl -fsSL \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu"${UBUNTU_VERSION}"/"${NVARCH}"/cuda-ubuntu"${UBUNTU_VERSION}".pin \
    | tee /etc/apt/preferences.d/cuda-repository-pin-600 \
    ;
  curl -fsSL \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu"${UBUNTU_VERSION}"/"${NVARCH}"/cuda-keyring_1.1-1_all.deb \
    -o cuda-keyring.deb \
    ;
  dpkg -i cuda-keyring.deb && rm -f cuda-keyring.deb \
  ;

  ## Default the driver version to 565 if not already set
  NVIDIA_DRIVER_VERSION="${NVIDIA_DRIVER_VERSION:-565}"

  ## Include cuBLAS (CUDA Basic Linear Algebra Subprograms) like libcublas-dev
  PKGS=(
    ## Install only the latest NVIDIA CUDA Toolkit "nvcc" (compiler, libraries, headers, etc.)
    # nvidia-cuda-toolkit
    cuda-toolkit
    ## Install only the NVIDIA CUDA drivers "nvidia-smi" using the variable, it will default to 565
    "nvidia-driver-${NVIDIA_DRIVER_VERSION:-565}"
    ## Install the latest full NVIDIA CUDA drivers environment
    # cuda
    libcudnn8
    libcudnn8-dev
  )
  apt-get update \
  && apt-get install -y --no-install-recommends "${PKGS[@]}"

  ## Set environment variables
  ## Puts the nvidia-smi binary (system management interface) on path
  ## with associated library files to execute it
  echo "Setting up environment variables for CUDA..."
  export PATH="/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:${PATH}}"
  export LIBRARY_PATH="/usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

  # echo 'export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:${PATH}}' >> ~/.bashrc
  # echo 'export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}' >> ~/.bashrc
  # source ~/.bashrc

  ## Set NVIDIA container runtime capabilities
  ## https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#dockerfiles
  export NVIDIA_VISIBLE_DEVICES="all"
  export NVIDIA_DRIVER_CAPABILITIES="compute,utility"

  ## Verify installation of NVIDIA drivers and CUDA toolkit
  echo "Verifying installation...";
  nvidia-smi;   # Check GPU status and drivers
  nvcc --version;  # Verify CUDA installation

  ## Clean up
  echo "Cleaning...";
  apt-get clean
  rm -rf "/var/lib/apt/lists/*"

  ## .rpmdb used by Red Hat-based systems to keep track of installed packages
  # chown "${NB_UID}:${NB_GID}" "/home/${NB_USER}"
  # chmod u+rwx "/home/${NB_USER}/.rpmdb"
  # find "/home/${NB_USER}" -path "/home/${NB_USER}/.rpmdb" -prune -o -exec fix-permissions {}
  [ -d "/home/${NB_USER}/.rpmdb" ] && rm -rf "/home/${NB_USER}/.rpmdb"

else
  echo "Skipping GPU NVIDIA CUDA drivers setup...";
fi
