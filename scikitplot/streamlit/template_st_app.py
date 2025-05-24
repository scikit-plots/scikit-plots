"""Streamlit home UI template_st_app."""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=unused-import
# pylint: disable=unused-argument
# pylint: disable=ungrouped-imports
# pylint: disable=no-name-in-module
# pylint: disable=broad-exception-caught

# template_st_app.py
import importlib
import os

from scikitplot._compat.optional_deps import HAS_STREAMLIT, safe_import

if HAS_STREAMLIT:
    st = safe_import("streamlit")

    from scikitplot.streamlit import (  # noqa: F401
        template_st_chat_ui,
        template_st_data_loader_ui,
        template_st_data_visualizer_ui,
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
        "📄 Data Visualization": template_st_data_visualizer_ui.run_data_visualizer_ui,
        "📁 Data Load": template_st_data_loader_ui.run_data_loader_ui,
        "💬 Assistant Chat": template_st_chat_ui.run_chat_ui,
        "🏠 Home Page": "home",
        # "🔐 Login": template_st_login_ui.run_login_form_ui,
        # Add more entries like "📊 Visualize": run_visualizer_ui, etc.
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
        st.sidebar.title("🔍 Navigation")

        page_keys = list(PAGES.keys())
        selected = page_keys[0]  # default to last

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
                "🔖 Select a Page",
                page_keys,
            )
        elif view_style == "Selectbox":
            selected = st.sidebar.selectbox(
                "🔖 Select a Page",
                page_keys,
                index=page_keys.index(st.session_state.selected_page),
            )
        elif view_style == "Radio":
            selected = st.sidebar.radio(
                "🔖 Select a Page",
                page_keys,
                index=page_keys.index(st.session_state.selected_page),
            )
        elif view_style == "Buttons":
            st.sidebar.markdown("### 🔖 Select a Page")
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

    # Run the app from command line
    if __name__ == "__main__":
        ## app entry-point
        run_app_ui()
