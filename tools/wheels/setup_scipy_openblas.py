# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def log(message: str) -> None:
    """
    Logs an informational message to the console.

    Parameters
    ----------
    message : str
        The message to be logged.

    """
    print(f"[INFO] {message}")


def error(message: str) -> None:
    """
    Logs an error message and raises a RuntimeError.

    Parameters
    ----------
    message : str
        The error message to be logged.

    """
    raise RuntimeError(f"[ERROR] {message}")


def success(message: str) -> None:
    """
    Logs a success message to the console.

    Parameters
    ----------
    message : str
        The success message to be logged.

    """
    print(f"[SUCCESS] {message}")


def run_command(
    command: list[str], check: bool = True, capture_output: bool = False
) -> Optional[str]:
    """
    Executes a shell command using subprocess.

    Parameters
    ----------
    command : list of str
        The command and its arguments as a list.
    check : bool, optional
        Whether to raise an exception if the command fails, by default True.
    capture_output : bool, optional
        Whether to capture and return the command's output, by default False.

    Returns
    -------
    Optional[str]
        The command output if `capture_output` is True, otherwise None.

    """
    try:
        result = subprocess.run(
            command, check=check, text=True, capture_output=capture_output
        )
        if capture_output:
            return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        error(f"Command '{' '.join(command)}' failed with error: {e}")
    return None


def configure_openblas_pkg_config(project_dir: str) -> None:
    """
    Configures the OpenBLAS PKG_CONFIG_PATH and library paths for the current platform.

    Parameters
    ----------
    project_dir : str
        Path to the project directory where OpenBLAS configuration will be set up.

    """
    pkg_config_path = Path(project_dir) / ".openblas"
    openblas_lib_dir = pkg_config_path / "lib"

    if pkg_config_path.exists():
        log(f"Removing existing OpenBLAS config directory: {pkg_config_path}")
        shutil.rmtree(pkg_config_path)

    log("Creating OpenBLAS config directory...")
    pkg_config_path.mkdir(parents=True, exist_ok=True)
    os.environ["PKG_CONFIG_PATH"] = str(pkg_config_path)

    if os.name == "posix":
        if "darwin" in os.uname().sysname.lower():
            os.environ["DYLD_LIBRARY_PATH"] = str(openblas_lib_dir)
        else:
            os.environ["LD_LIBRARY_PATH"] = (
                f'{openblas_lib_dir}:{os.environ.get("LD_LIBRARY_PATH", "")}'
            )
    elif os.name == "nt":
        os.environ["PATH"] = f'{openblas_lib_dir};{os.environ["PATH"]}'
    else:
        error("Unsupported operating system.")

    success(f'PKG_CONFIG_PATH set to {os.environ["PKG_CONFIG_PATH"]}')


def install_requirements(requirements_file: str) -> None:
    """
    Installs Python dependencies from a requirements file.

    Parameters
    ----------
    requirements_file : str
        Path to the requirements.txt file.

    """
    log(f"Installing requirements from {requirements_file}...")
    run_command(
        ["python", "-m", "pip", "install", "-U", "pip", "-r", requirements_file]
    )


def generate_openblas_pkgconfig(openblas_module: str, pkg_config_path: str) -> None:
    """
    Generates the OpenBLAS pkg-config file using a Python module.

    Parameters
    ----------
    openblas_module : str
        The Python module to use for generating the pkg-config file.
    pkg_config_path : str
        Path to the directory where the pkg-config file will be generated.

    """
    log(f"Generating OpenBLAS pkg-config file using {openblas_module}...")
    try:
        output = run_command(
            [
                "python",
                "-c",
                f"import {openblas_module}; print({openblas_module}.get_pkg_config())",
            ],
            capture_output=True,
        )
        if output:
            with open(Path(pkg_config_path) / "scipy-openblas.pc", "w") as f:
                f.write(output)
        success("Pkg-config file generated successfully.")
    except Exception as e:
        error(f"Failed to generate OpenBLAS pkg-config: {e}")


def copy_shared_libs(openblas_module: str, pkg_config_path: str) -> None:
    """
    Copies shared libraries from the OpenBLAS module to the configuration path.

    Parameters
    ----------
    openblas_module : str
        The Python module associated with OpenBLAS.
    pkg_config_path : str
        Path to the directory where shared libraries will be copied.

    """
    log(f"Copying shared libraries for {openblas_module}...")
    try:
        module_path = Path(__import__(openblas_module).__file__).parent
        srcdir = module_path / "lib"
        target_dir = Path(pkg_config_path) / "lib"
        shutil.copytree(srcdir, target_dir)

        # Handle macOS-specific dylibs
        srcdir_dylibs = module_path / ".dylibs"
        if srcdir_dylibs.exists():
            shutil.copytree(srcdir_dylibs, Path(pkg_config_path) / ".dylibs")
    except Exception as e:
        error(f"Failed to copy shared libraries: {e}")


def setup_openblas(project_dir: str) -> None:
    """
    Sets up OpenBLAS for SciPy based on the system architecture.

    Parameters
    ----------
    project_dir : str
        Path to the project directory where OpenBLAS configuration will be set up.

    """
    configure_openblas_pkg_config(project_dir)

    arch = os.uname().machine.lower()
    log(f"Detected architecture: {arch}")

    requirements_map = {
        "i686": "requirements/ci32_requirements.txt",
        "x86": "requirements/ci32_requirements.txt",
        "x86_64": "requirements/ci_requirements.txt",
        "arm64": "requirements/ci_requirements.txt",
    }

    openblas_modules_map = {
        "i686": "scipy_openblas32",
        "x86": "scipy_openblas32",
        "x86_64": "scipy_openblas64",
        "arm64": "scipy_openblas64",
    }

    if arch not in requirements_map:
        error(f"Unsupported architecture: {arch}")

    requirements_file = requirements_map[arch]
    openblas_module = openblas_modules_map[arch]

    install_requirements(requirements_file)
    pkg_config_path = Path(project_dir) / ".openblas"
    generate_openblas_pkgconfig(openblas_module, str(pkg_config_path))
    copy_shared_libs(openblas_module, str(pkg_config_path))
    success("OpenBLAS setup completed successfully.")


def main():
    """
    Main function to parse arguments and initiate OpenBLAS setup.

    RUN: python tools/wheels/setup_scipy_openblas.py --project_dir ./
    """
    parser = argparse.ArgumentParser(
        description="Setup OpenBLAS for SciPy on any platform."
    )
    parser.add_argument(
        "--project_dir", type=str, required=True, help="Path to the project directory"
    )
    # parser.add_argument(
    #     "--install_openblas", action="store_true", help="Flag to enable OpenBLAS installation"
    # )
    args = parser.parse_args()

    try:
        setup_openblas(project_dir=args.project_dir)
    except RuntimeError as e:
        error(str(e))


if __name__ == "__main__":
    main()
