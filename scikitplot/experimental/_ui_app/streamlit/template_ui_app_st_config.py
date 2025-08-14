# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=unused-import
# pylint: disable=unused-argument
# pylint: disable=import-outside-toplevel
# pylint: disable=broad-exception-caught

"""config."""

from typing import TYPE_CHECKING

# from scikitplot import logger
from scikitplot._compat.optional_deps import LazyImport

if TYPE_CHECKING:
    from typing import Literal, Optional

__all__ = []

# import streamlit as st
st = LazyImport("streamlit", package="streamlit")

# Use st.cache_data for immutable data and st.cache_resource for reusable, expensive resources
# Use @st.fragment to create modular, reusable UI blocks with proper state handling
if st:
    __all__ += [
        "st_page_config",
    ]

    def st_page_config(
        title: str = "scikit-plots",
        layout: "Literal['centered', 'wide']" = "wide",
        sidebar_state: "Optional[Literal['auto', 'expanded', 'collapsed']]" = None,
        menu_items: "Optional[dict[str, str]]" = None,
    ) -> None:
        """
        Set Streamlit page configuration as the first command in the app.

        Parameters
        ----------
        title : str
            Title of the browser tab.
        layout : {"centered", "wide"}
            Layout of the app.
        sidebar_state : {"auto", "expanded", "collapsed"}, optional
            Initial state of the sidebar. If None, uses session_state or defaults to "collapsed".
        menu_items : dict, optional
            Custom items for the Streamlit hamburger menu. Keys: 'Get Help', 'Report a bug', 'About'
        """
        # Default fallback for sidebar state
        if sidebar_state is None:
            if "sidebar_state" not in st.session_state:
                st.session_state.sidebar_state = "collapsed"
            sidebar_state = st.session_state.sidebar_state

        # Set configuration - must be first Streamlit command
        st.set_page_config(
            page_title=title,
            layout=layout,
            initial_sidebar_state=sidebar_state,
            menu_items=menu_items
            or {
                "Report a bug": "https://github.com/scikit-plots/scikit-plots/issues",
                "Get Help": "https://scikit-plots.github.io/dev",
                "About": "### Scikit-Plot Explorer\nBuilt with Streamlit",
            },
        )
