# run_app.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Streamlit Launcher CLI for Scikit-Plots.

This module provides a command-line interface (CLI) for launching the Scikit-Plots
Streamlit app with configurable options such as host address, port, theme, and
whether to load a sample library.

---

Usage (CLI)::

    python -m scikitplot.streamlit.run_app st --address 0.0.0.0 --port 8501 --lib_sample

You can also configure Streamlit via a persistent config file::

    # ~/.streamlit/config.toml
    [server]
    port = 8501
    address = "0.0.0.0"
    headless = true  # Useful for headless environments (CI/CD, remote servers)

For Docker or WSL 2 environments::

    # ~/.wslconfig (on Windows)
    localhostforwarding=true

    # Docker: set address to 0.0.0.0 instead of localhost
    python -m scikitplot.streamlit.run_app st --address 0.0.0.0

---

Command-line Options (see `--help` for full list):

    --file_path    Path to the Streamlit app script
    --address      Host address to bind to (e.g., 0.0.0.0 for public)
    --port         Port to serve the app
    --dark_theme   Enable dark mode in UI
    --lib_sample   Use preloaded example dataset

This CLI is intended for both local development and deployment scenarios.

Requirements:
- Streamlit must be installed (`pip install streamlit`)
"""

import argparse

# import click
import os
import subprocess
import sys

from scikitplot import logger

__all__ = [
    "launch_streamlit",
    "launch_streamlit",
    "run_ui_app_st",
]

# ---------------------------
# Common app launcher Function
# ---------------------------


def launch_streamlit(
    file_path: str = "template_ui_app_st.py",
    address: str = "localhost",
    port: str = "8501",
    dark_theme: bool = False,
    lib_sample: bool = False,
):
    """
    Launch the Streamlit app using subprocess to call the Streamlit CLI.

    Parameters
    ----------
    file_path : str
        Path to the Streamlit app file (default is "template_ui_app_st.py").
    address : str
        Address on which to run the Streamlit app (default is 'localhost').
    port : str
        Port on which to run the Streamlit app (default is '8501').
    headless : bool
        Useful when running on remote servers or CI/CD pipelines without a GUI.
    dark_theme : str
        Theme for the UI. Accepts "light" or "dark" (default is "light").
    lib_sample : bool
        If True, uses a sample library setting (default is False).

    Notes
    -----
    This function builds the appropriate command and launches the Streamlit app
    as a subprocess, so users do not have to call Streamlit manually.
    """
    # Determine app path based on whether a sample library should be used
    file_path = (
        os.path.join(os.path.dirname(__file__), file_path)
        if lib_sample
        else os.path.join(os.getcwd(), file_path)
    )

    # if headless:
    # useful when running on remote servers or CI/CD pipelines without a GUI.
    # os.environ["STREAMLIT_HEADLESS"] = "true"

    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    os.environ["STREAMLIT_SERVER_ADDRESS"] = str(address)

    # Build command for launching the Streamlit app
    run_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        file_path,
        # "--server.port", str(port),
        # "--server.address", str(address),
    ]

    # Append the theme option if specified
    if dark_theme:
        theme = "dark"
        run_cmd += ["--", f"--theme={theme}"]

    # Execute the Streamlit app command in a subprocess
    subprocess.run(run_cmd, check=False)  # noqa: S603


# ---------------------------
# argparse variant
# ---------------------------


# argparse parser
def parse_arguments(args=None):
    """
    Parse the command-line arguments passed to the script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including file_path, theme, and lib_sample.
    """
    # Initialize the argument parser with a description
    parser = argparse.ArgumentParser(description="Launch the Streamlit UI app.")
    # Add argument for specifying the Streamlit app file path
    parser.add_argument(
        "--file_path",
        default="template_ui_app_st.py",
        help="Path to the Streamlit app file (default: 'template_ui_app_st.py')",
    )
    parser.add_argument(
        "--address",
        "-a",
        default="localhost",
        type=str,
        help="Address to run the Streamlit app on (default: 'localhost')",
    )
    parser.add_argument(
        "--port",
        "-p",
        default="8501",
        type=str,
        help="Port to run the Streamlit app on (default: '8501')",
    )
    # Add argument for selecting the theme (light or dark)
    # Theme option as a boolean flag, default is light
    parser.add_argument(
        "--dark_theme",
        "-d",
        # default="light",
        # choices=["light", "dark"],
        # help="Choose a UI theme for the app (default: 'light')",
        action="store_true",
        help=(
            "Choose a UI dark_theme for the app (default: False). "
            "Pass this flag to enable dark_theme 'dark' mode."
        ),
    )
    # Add argument for specifying if a sample library should be used
    # Sample library option as a boolean flag, default is False
    # Option 1: Default is False, flag enables it
    # If lib_sample should be False by default and True when passed:
    # parser.add_argument(
    #     "--lib_sample",
    #     "-ls",
    #     action="store_true",
    #     # default=False,
    #     help=(
    #         "Use a sample library setting (default: False). Pass this flag to enable."
    #     ),
    # )
    # Option 2: Default is True, flag disables it
    # If lib_sample should be True by default and False when passed:
    parser.add_argument(
        "--no_lib_sample",
        "-nls",
        action="store_false",
        dest="lib_sample",
        help="Disable the sample library setting (default: True).",
    )
    # Parse and return the arguments
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.debug(f"unknown: {unknown}")
        # pass
    return args or parser.parse_args(args=args)


def run_ui_app_st(args=None):
    """run_ui_app_st."""
    # Entry point: parse args and launch Streamlit
    # Parse command-line arguments
    args = parse_arguments(args=args)
    # Launch the Streamlit app with the parsed arguments
    launch_streamlit(
        args.file_path,
        args.address,
        args.port,
        args.dark_theme,
        args.lib_sample,
    )


# ---------------------------
# click variant for main cli
# ---------------------------

# @click.command()
# @click.option("--file_path", default="template_ui_app_st.py", help="Streamlit app file")
# @click.option("--address", "-a", default="localhost", help="Host address")
# @click.option("--port", "-p", default="8501", help="Port")
# @click.option("--dark_theme", "-d", is_flag=True, help="Enable dark theme")
# @click.option("--lib_sample", "-ls", is_flag=True, help="Use sample lib")
# def run_ui_app_st(file_path, address, port, dark_theme, lib_sample):
#     launch_streamlit(
#         file_path=file_path,
#         address=address,
#         port=port,
#         dark_theme=dark_theme,
#         lib_sample=lib_sample,
#     )

# ---------------------------
# Entrypoint
# ---------------------------

if __name__ == "__main__":
    # # Choose argparse or click variant
    # import sys
    # if "--click" in sys.argv:
    #     sys.argv.remove("--click")
    #     run_click()
    # else:
    #     run_argparse()
    run_ui_app_st()
