# template_ui_app_st.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=unused-import
# pylint: disable=unused-argument
# pylint: disable=ungrouped-imports
# pylint: disable=broad-exception-caught
# pylint: disable=line-too-long

"""
Streamlit home UI template_ui_app_st.

~/.streamlit/config.toml

├── template_ui_app_st.py        ← this file
└── .streamlit/
    └── config.toml
"""

# from scikitplot import logger
from scikitplot._compat.optional_deps import LazyImport

__all__ = []

# import streamlit as st
st = LazyImport("streamlit", package="streamlit")

# Use st.cache_data for immutable data and st.cache_resource for reusable, expensive resources
# Use @st.fragment to create modular, reusable UI blocks with proper state handling
if st:
    from scikitplot.experimental._ui_app.streamlit import (
        template_ui_app_st_chat,
        template_ui_app_st_data_visualizer,
        template_ui_app_st_dataset_loader,
        template_ui_app_st_login,
    )

    __all__ += [
        "ui_app_st",
    ]

    # --- home_ui ---
    def st_home():
        """st_home."""
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

    ######################################################################
    ## discover ui
    ######################################################################

    # def discover_ui_pages(folder=".", module="scikitplot.snsx_explorer"):
    #     """discover_ui_pages."""
    #     pages = {
    #         "🏠 Home": "home",
    #     }
    #     for fname in os.listdir(folder):  # noqa: PTH208
    #         if fname.endswith("_ui.py") and "login" not in fname:
    #             sub_mod_name = fname[:-3]  # remove ext
    #             page_name = (
    #                 sub_mod_name.removeprefix("template_st_")
    #                 .removesuffix("_ui")
    #                 .replace("_", " ")
    #                 .title()
    #             )
    #             module = importlib.import_module(f"{module}.{sub_mod_name}")
    #             if hasattr(module, "run"):
    #                 pages[f"📄 {page_name}"] = getattr(module, "run")  # noqa: B009
    #     return pages
    # Define all available pages
    PAGES = {
        "💬 Assistant Chat": template_ui_app_st_chat.st_chat,
        "📁 Dataset Load": template_ui_app_st_dataset_loader.st_dataset_loader,
        "📄 Visualization": template_ui_app_st_data_visualizer.st_data_visualizer,
        "🏠 Home Page": st_home,
        # Add more entries like "📊 Visualize": run_visualizer_ui, etc.
        # "🔐 Login": template_st_login_ui.run_login_form_ui,
    }
    # PAGES.update(discover_ui_pages())

    ######################################################################
    ## Sidebar Logo
    ######################################################################

    def st_add_sidebar_logo(
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
        # HTML for custom styling and alignment
        # st.markdown(
        #     """
        #     <style>
        #     .center-align {
        #         margin: 0 auto;           /* Automatically center horizontally */
        #         text-align: center;       /* Center text */
        #     }
        #     </style>
        #     """, unsafe_allow_html=True)
        ## Content for left and right corners
        # st.markdown(
        #     '<div class="center-align">This content is centered</div>',
        #     unsafe_allow_html=True)
        # Placeholder
        # st.container A static layout block.
        # st.empty().container Dynamic and replaceable container.
        with st.sidebar, st.container(border=True):
            st.markdown(
                f"""
            <style>
            /* Make sidebar a flex column container */
            section[data-testid="stSidebar"] > div:first-child {{
                /* height: 100vh; */
                height: 100%;
                display: flex;
                flex-direction: column;
                /* justify-content: space-between; */
                margin: 0rem;
                padding: 0rem;
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

    ######################################################################
    ## add_sidebar
    ######################################################################

    def st_add_sidebar():
        """
        Make sidebar.

        You can add any content here: title, logo, menu, etc.
        """
        # Sidebar for controlling expanders and categories
        with st.sidebar, st.container(border=True):
            st.logo(
                image=(
                    "https://raw.githubusercontent.com/scikit-plots/scikit-plots"
                    "/main/docs/source/logos/scikit-plots-logo.svg"
                ),
                icon_image=(
                    "https://raw.githubusercontent.com/scikit-plots/scikit-plots"
                    "/main/docs/source/logos/scikit-plots-favicon.ico"
                ),
                link="https://scikit-plots.github.io/dev",
                size="small",  # "medium","large"
            )

        st.sidebar.title("🔖 Navigation")
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
        return view_style  # noqa: RET504

    ######################################################################
    ## CONTENT UI
    ######################################################################

    def ui_app_st():  # noqa: PLR0912
        """Launch the Streamlit login app."""
        page_keys = list(PAGES.keys())
        selected = page_keys[0]  # default to first
        ## Initialize session state with defaults (only once)
        # st.session_state.setdefault("selected_page", selected)
        if "selected_page" not in st.session_state:
            st.session_state.selected_page = selected

        ## Only main page, Must be first Streamlit call
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

        ## ---- Login UI ----
        ## Show login page first
        if not template_ui_app_st_login.st_login():
            st.stop()  # 👈 prevent rest of the app from rendering

        ## ---- add_sidebar UI top ----
        view_style = st_add_sidebar()

        # Placeholder
        with st.container(border=True):
            ## ---- App Content UI ----
            # Update Content by selection
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
            else:
                pass

        ## ---- add_sidebar bottom Logo ----
        st_add_sidebar_logo()

    # Run the app from command line
    if __name__ == "__main__":
        ## app entry-point
        ui_app_st()
