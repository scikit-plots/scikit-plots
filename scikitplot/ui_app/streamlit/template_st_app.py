"""
Streamlit home UI template_st_app.

~/.streamlit/config.toml

├── template_st_app.py        ← this file
└── .streamlit/
    └── config.toml

"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=unused-import
# pylint: disable=unused-argument
# pylint: disable=ungrouped-imports
# pylint: disable=no-name-in-module
# pylint: disable=broad-exception-caught
# pylint: disable=line-too-long

# template_st_app.py
import importlib
import os

from scikitplot._compat.optional_deps import HAS_STREAMLIT, safe_import

if HAS_STREAMLIT:
    # import streamlit as st
    st = safe_import("streamlit")

    from scikitplot.ui_app.streamlit import (  # noqa: F401
        template_st_chat_ui,
        template_st_data_visualizer_ui,
        template_st_dataset_loader_ui,
        template_st_login_ui,
    )

    def discover_ui_pages(folder=".", module="scikitplot.snsx_explorer"):
        """discover_ui_pages."""
        pages = {
            "🏠 Home": "home",
        }
        for fname in os.listdir(folder):  # noqa: PTH208
            if fname.endswith("_ui.py") and "login" not in fname:
                sub_mod_name = fname[:-3]  # remove ext
                page_name = (
                    sub_mod_name.removeprefix("template_st_")
                    .removesuffix("_ui")
                    .replace("_", " ")
                    .title()
                )
                module = importlib.import_module(f"{module}.{sub_mod_name}")
                if hasattr(module, "run"):
                    pages[f"📄 {page_name}"] = getattr(module, "run")  # noqa: B009
        return pages

    # Define all available pages
    PAGES = {
        "💬 Assistant Chat": template_st_chat_ui.run_chat_ui,
        "📁 Dataset Load": template_st_dataset_loader_ui.run_dataset_loader_ui,
        "📄 Visualization": template_st_data_visualizer_ui.run_data_visualizer_ui,
        "🏠 Home Page": "home",
        # Add more entries like "📊 Visualize": run_visualizer_ui, etc.
        # "🔐 Login": template_st_login_ui.run_login_form_ui,
    }
    # PAGES.update(discover_ui_pages())

    def run_home_ui():
        """run_home_ui."""
        st.title("🏠 Welcome to scikit-plots")
        st.markdown(
            """
        Use the sidebar to explore:
        - **Load Data**: Import from local files, URLs, or databases.
        - Future modules: visualizations, reports, AI insights.
        """
        )
        st.success(
            "Try different view styles in the sidebar to test app navigation UI."
        )

    def sidebar_logo(
        url: str = "https://scikit-plots.github.io/dev",
        logo_url: str = "https://raw.githubusercontent.com/scikit-plots/scikit-plots/main/docs/source/logos/scikit-plots-logo.svg",
        title: str = "scikit-plots",
        max_height: str = "80px",
        font_size: str = "1.1rem",
        align: str = "center",  # Now explicitly takes 'left', 'center', or 'right'
    ):
        """
        Display a logo and title at the bottom of the Streamlit sidebar.

        Parameters
        ----------
        url : str
            Link to open when the logo/title is clicked.
        logo_url : str
            URL or path to the logo image.
        title : str
            Sidebar title displayed below the logo.
        max_height : str
            Maximum height of the logo image (e.g., '70px').
        font_size : str
            Font size of the title (e.g., '1.1rem').
        align : 'left', 'center', or 'right'
            Whether to center the logo and title.
        """
        # st.sidebar.markdown(
        #     f"""
        #     <div class="sidebar-logo">
        #         <a align=center href="https://scikit-plots.github.io/dev">
        #           <img src={
        #         "https://raw.githubusercontent.com/scikit-plots/scikit-plots"
        #         "/main/docs/source/logos/scikit-plots-logo.svg"
        #     }
        #           alt="Logo" width="auto">
        #         </a>
        #         <a align=center href="https://scikit-plots.github.io/dev">
        #           <h2 align=center>scikit-plots</h2>
        #         </a>
        #     </div>
        #     """,
        #     unsafe_allow_html=True,
        # )
        st.sidebar.markdown(
            f"""
        <style>
        /* Make sidebar a flex column container */
        section[data-testid="stSidebar"] > div:first-child {{
            /* position: fixed; */
            left: 0;
            top: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
            /* space-between pushes last child to bottom */
            justify-content: space-between;
            padding-top: 0rem;
            padding-bottom: 0rem;
        }}

        /* Optional: reduce overall sidebar top margin
        section[data-testid="stSidebar"] {{
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }} */

        .sidebar-logo {{
            text-align: {align};
            margin-top: auto; /* push to bottom */
            padding: 2rem 1rem 1rem 1rem;
        }}

        .sidebar-logo img {{
            max-height: {max_height};
            width: 100%;
            object-fit: contain;
        }}

        .sidebar-logo h2 {{
            margin: 0.2em 0 0 0;
            font-size: {font_size};
            color: inherit;
        }}
        /*
        @media (prefers-color-scheme: dark) {{
            .sidebar-logo img {{
                filter: brightness(0.9) invert(1);
            }}
        }}*/
        </style>

        <div class="sidebar-logo">
            <a href="{url}" target="_blank" style="text-decoration: none;">
                <img src="{logo_url}" alt="{title} Logo" onerror="this.style.display='none';">
                <h2>{title}</h2>
            </a>
        </div>
        """,
            unsafe_allow_html=True,
        )

    def run_app_ui():  # noqa: PLR0912
        """Launch the Streamlit login app."""
        ## Must be first Streamlit call
        st.set_page_config(
            page_title="scikit-plots",
            layout="wide",
            initial_sidebar_state=st.session_state.setdefault(
                "sidebar_state", "collapsed"
            ),
            menu_items={
                "Report a bug": "https://github.com/scikit-plots/scikit-plots/issues",
                "Get Help": "https://scikit-plots.github.io/dev",
                "About": "### Scikit-Plots Explorer\nBuilt with Streamlit",
            },
        )
        ## Show login page first
        ## ---- Login ----
        if not template_st_login_ui.run_login_form_ui():
            st.stop()  # 👈 prevent rest of the app from rendering

        ## ---- App Content ----
        st.sidebar.title("🔖 Navigation")

        page_keys = list(PAGES.keys())
        selected = page_keys[0]  # default to first
        ## Initialize session state with defaults (only once)
        # st.session_state.setdefault("selected_page", selected)
        if "selected_page" not in st.session_state:
            st.session_state.selected_page = selected

        # Allow user to choose navigation layout style
        view_style = st.sidebar.selectbox(
            "Choose Navigation Style",
            [
                "Tabs",
                "Selectbox",
                "Radio",
                "Buttons",
            ],
        )
        # Update selection
        if view_style == "Tabs":
            selected = st.selectbox(
                "🔍 Select a Page",
                page_keys,
            )
        elif view_style == "Selectbox":
            selected = st.sidebar.selectbox(
                "🔍 Select a Page",
                page_keys,
                index=page_keys.index(st.session_state.selected_page),
            )
        elif view_style == "Radio":
            selected = st.sidebar.radio(
                "🔍 Select a Page",
                page_keys,
                index=page_keys.index(st.session_state.selected_page),
            )
        elif view_style == "Buttons":
            st.sidebar.markdown("### 🔍 Select a Page")
            for page in page_keys:
                if st.sidebar.button(page, use_container_width=True):
                    selected = page
        st.session_state.selected_page = selected
        # Render the selected page
        page_func = PAGES[st.session_state.selected_page]
        if callable(page_func):
            page_func()
        elif page_func == "home":
            run_home_ui()
        # Add Logo
        sidebar_logo()

    # Run the app from command line
    if __name__ == "__main__":
        ## app entry-point
        run_app_ui()
