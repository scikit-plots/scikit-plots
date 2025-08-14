# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=unused-import

"""build_app for Os Independent."""

import os  # noqa: F401
import platform
import shutil  # noqa: F401
import subprocess

__all__ = [
    "run_build_app",
]


def run_build_app():
    """build_app."""
    system = platform.system()

    if system not in ["Windows", "Linux", "Darwin"]:
        raise RuntimeError("Unsupported OS for build.")

    script_name = "app.py"
    out_file = "snsx_plot_explorer.exe" if system == "Windows" else "snsx_plot_explorer"
    build_cmd = ["pyinstaller", "--onefile", "--clean", "--name", out_file, script_name]

    print("📦 Building executable with PyInstaller...")  # noqa: T201
    subprocess.run(build_cmd, check=False)  # noqa: S603

    # Move the output to /dist-clean
    # shutil.move(f"dist/{out_file}", f"dist/{out_file}")


if __name__ == "__main__":
    run_build_app()
