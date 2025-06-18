# config.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: skip-file
# ruff: noqa: PGH004
# ruff: noqa
# flake8: noqa
# type: ignore
# mypy: ignore-errors

import streamlit as st
from typing import Optional, Literal

__all__ = []

# Use st.cache_data for immutable data and st.cache_resource for reusable, expensive resources
# Use @st.fragment to create modular, reusable UI blocks with proper state handling

def configure_page(
    title: str = "Streamlit App",
    layout: Literal["centered", "wide"] = "wide",
    sidebar_state: Optional[Literal["auto", "expanded", "collapsed"]] = None,
    menu_items: Optional[dict[str, str]] = None,
) -> None: ...
