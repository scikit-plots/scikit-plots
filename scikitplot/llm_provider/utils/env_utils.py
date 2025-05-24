"""env_utils."""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=broad-exception-caught

import os


def run_load_dotenv(dotenv_path: str = "", override=True) -> None:
    """run_load_dotenv."""
    try:
        # pylint: disable=import-outside-toplevel
        from dotenv import load_dotenv  # type: ignore[reportMissingImports]

        # path = Path.cwd() / path
        # path = Path(path).expanduser().resolve()
        # path = os.path.abspath(os.path.join(os.getcwd(), path))
        dotenv_path = os.path.abspath(os.path.expanduser(dotenv_path))
        # Variables already in the environment are not overwritten:
        load_dotenv(
            dotenv_path=dotenv_path or f"{os.getcwd()}/.env",
            override=override,  # ensure .env values override existing ones
        )
    except Exception:
        pass
