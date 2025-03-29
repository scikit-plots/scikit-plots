"""test_scripts"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import glob
import logging
import os
import subprocess

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


def run_scripts(script_dir):
    """Run all Python scripts found in the specified directory and its subdirectories."""
    # Use glob to find all Python scripts recursively
    scripts = glob.glob(os.path.join(script_dir, "**", "*.py"), recursive=True)
    if not scripts:
        logger.warning(f"No Python scripts found in {script_dir}.")

    for script in scripts:
        script_name = os.path.basename(script)
        logger.info(f"Running script: {script_name}")
        try:
            # Correct the path: only use relative paths within the `base_dir`
            relative_script_path = os.path.relpath(script, start=script_dir)

            # Run the script using subprocess with the correct relative path
            subprocess.run(
                f"python {relative_script_path}", shell=True, cwd=script_dir, check=True
            )
            logger.info(f"Successfully executed {script_name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error executing {script_name}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error while executing {script_name}: {e}")


def main():
    import os

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run Python scripts and optionally save plots."
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        default="galleries/examples",
        help="Base directory to search for scripts",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Flag to save plots (default is to display them)",
    )
    args = parser.parse_args()
    # Determine whether to save plots or not
    os.environ["SAVE_PLOTS"] = "1" if args.save_plots else ""
    logger.info(f"Starting script execution in directory: {args.base_dir}")

    # Run scripts in the base directory and its subdirectories
    run_scripts(args.base_dir)
    logger.info("All Python scripts execution process finished.")


if __name__ == "__main__":
    main()
