"""env_utils."""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=broad-exception-caught

# import os
from ... import logger


def run_load_dotenv(dotenv_path: str = "", override=False) -> None:
    """run_load_dotenv."""
    try:
        # pylint: disable=import-outside-toplevel
        from dotenv import (  # type: ignore[reportMissingImports]
            find_dotenv,
            load_dotenv,
        )

        # path = Path.cwd() / path or f"{os.getcwd()}/.env"
        # path = Path(path).expanduser().resolve()
        # path = os.path.abspath(os.path.join(os.getcwd(), path))
        # path = os.path.abspath(os.path.expanduser(path))
        # returns the absolute path of the first .env file it finds
        # while searching upward from the current directory.
        logger.info(
            f"Found .env file at: {find_dotenv()}"
        )  # See where it looks for the .env file
        dotenv_path = dotenv_path or find_dotenv()
        # load_dotenv()  # Will load from `.env` in cwd by default
        # Variables already in the environment are not overwritten:
        load_dotenv(
            dotenv_path=dotenv_path,
            override=override,  # ensure .env values override existing ones
        )
    except Exception:
        pass
