# pylint: skip-file
# ruff: noqa: PGH004
# ruff: noqa
# flake8: noqa
# type: ignore
# mypy: ignore-errors

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# config.py
import streamlit as st
from typing import Optional, Literal

def configure_page(
    title: str = "Streamlit App",
    layout: Literal["centered", "wide"] = "wide",
    sidebar_state: Optional[Literal["auto", "expanded", "collapsed"]] = None,
    menu_items: Optional[dict[str, str]] = None,
) -> None: ...
