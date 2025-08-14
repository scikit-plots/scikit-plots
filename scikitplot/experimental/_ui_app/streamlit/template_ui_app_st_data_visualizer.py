# template_st_data_visualizer.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=broad-exception-caught
# pylint: disable=unused-argument
# pylint: disable=ungrouped-imports

"""
Streamlit-based UI logic for exploring the plotting.

This module launches a Streamlit web application that allows users to
interactively explore plotting functions from the `snsx` library —
used for Machine Learning Exploratory Data Analysis (ML-EDA) and
interpretability.

Functions are organized by category and displayed with metadata
including task type, plot type, explainability level, and more.

- https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps
"""

# """
# This Streamlit application provides an interactive UI for exploring
# plotting functions from the `snsx` library — a suite of tools for
# Machine Learning Exploratory Data Analysis (ML-EDA) and interpretability.

# It uses a dynamic catalog of function metadata (`snsx_catalog`) and
# organizes the available plotting utilities by category.

# Categories include:
# - Evaluation (e.g., ROC curve, confusion matrix)
# - Representation (e.g., PCA, UMAP)
# - Explanation (e.g., SHAP, LIME)
# - Training (e.g., loss curves, overfitting detection)
# - Others as defined in the catalog

# What Users Can Do:
# - Select from a list of categories to filter functions
# - View detailed function metadata:
#     - Function name and description
#     - Task type and plot type
#     - Whether it is supervised or unsupervised
#     - Level of explainability
#     - Example usage with parameter names

# The app reflects any changes or additions made to the `snsx_catalog`
# without requiring manual updates to the UI code.

# Setup and Usage:
# Follow the steps below to run and explore this application:

# 1. Install Dependencies:
#    Ensure you have Streamlit installed:
#        pip install streamlit

#    This app also depends on the `snsx` library. Make sure it's
#    installed and includes `snsx.catalog`.

# 2. Save the Script:
#    Save this module as `streamlit_app.py` or similar.

# 3. Run the App:
#    In your terminal, navigate to the file's directory and run:
#        streamlit run streamlit_app.py

#    This will start the Streamlit server and open the app in a browser.

# 4. Use the App:
#    - Select a category from the dropdown
#    - Review functions and their metadata
#    - Use the example code snippets in your ML-EDA workflows

# Extensibility:
# - You can add live plot previews by including example data and
#   executing the function directly.
# - Additional filters can be added (e.g., task type, explainability).
# - Future versions could support search and table views for easier browsing.
# """

# import petname
# petname.Generate(3, separator="-")  # e.g., 'green-fox-jump'
# import hashlib
# Matplotlib doesn't work well with threads.
from threading import RLock
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from scikitplot import logger
from scikitplot._compat.optional_deps import LazyImport

__all__ = []

_lock = RLock()

# import streamlit as st
st = LazyImport("streamlit", package="streamlit")

# Use st.cache_data for immutable data and st.cache_resource for reusable, expensive resources
# Use @st.fragment to create modular, reusable UI blocks with proper state handling
if st:
    from scikitplot.experimental._ui_app.streamlit import get_sns_data, snsx_catalog
    from scikitplot.experimental._ui_app.streamlit.template_ui_app_st_chat import (
        get_response,
        st_add_sidebar_api_key,
    )

    if TYPE_CHECKING:
        from typing import Optional

        from streamlit.delta_generator import DeltaGenerator

    __all__ += [
        "st_data_visualizer",
    ]

    ######################################################################
    ## get_plot_func
    ######################################################################

    # Cache expensive resources (e.g., models, DB conns), Assumes mutable (and reusable) objects
    @st.cache_resource
    def get_plot_func(function_meta: "dict[str, any]"):
        """
        Dynamically import a plotting function from a string path.

        Parameters
        ----------
        function_meta : dict
            Metadata about the function including keys like:
            - module: str
            - function: str
            - fallback_function: str
            - description: str
            - task_type: str
            - plot_type: str
            - supervised: bool
            - explainability_level: str
            - parameters: list of str
            - optional_parameters: dict

        Returns
        -------
        Callable
            The actual function object.

        Raises
        ------
        AttributeError
        """
        # Dynamically import function
        logger.info(f"Called function {function_meta['function']}")
        func = LazyImport(
            function_meta["function"],
            package="scikitplot",
        ).resolved
        if callable(func):
            return func
        # Dynamically import function
        logger.info(f"Called function {function_meta['fallback_function']}")
        func = LazyImport(
            function_meta["fallback_function"],
            package="scikitplot",
        ).resolved
        if callable(func):
            return func
        logger.warning(
            f"Function not callable, using fallback: {function_meta['fallback_function']}"
        )
        raise TypeError(f"Object {function_meta['fallback_function']} is not callable.")

    # -------------------- UI Components --------------------

    ######################################################################
    ## select_functions
    ######################################################################

    def select_category() -> str:
        """
        Display a dropdown for the user to select a category from the snsx catalog.

        Returns
        -------
        str
            The selected category name.
        """
        categories = sorted({f["category"] for f in snsx_catalog})
        return st.selectbox("Selected category", categories)

    # Cache pure data (e.g., DataFrames, results), Assumes immutable return values
    @st.cache_data
    def filter_by_category(category: str) -> list[dict]:
        """
        Filter functions from the snsx catalog by category.

        Parameters
        ----------
        category : str
            The category to filter by (e.g., "Evaluation", "Explanation").

        Returns
        -------
        list of dict
            List of function metadata dictionaries matching the selected category.
        """
        return [f for f in snsx_catalog if f["category"] == category]

    def multiselect_categories() -> list:
        """
        Display a multiselect for the user to select categories from the snsx catalog.

        Returns
        -------
        list
            The selected categories.
        """
        categories = sorted({f["category"] for f in snsx_catalog})
        return st.multiselect(
            "Selected categories",
            categories,
            default=categories[:2],
        )

    # Cache pure data (e.g., DataFrames, results), Assumes immutable return values
    @st.cache_data
    def filter_by_categories(categories: list) -> list[dict]:
        """
        Filter functions from the snsx catalog by multiple categories.

        Parameters
        ----------
        categories : list
            The list of categories to filter by.

        Returns
        -------
        list of dict
            List of function metadata dictionaries matching the selected categories.
        """
        return [f for f in snsx_catalog if f["category"] in categories]

    def select_functions(filtered_entries: list[dict]) -> list[dict]:
        """
        Display a multiselect for the user to choose specific functions.

        Parameters
        ----------
        filtered_entries : list of dict
            List of function metadata dictionaries.

        Returns
        -------
        list of dict
            List of selected function metadata dictionaries.
        """
        function_names = [f["function"] for f in filtered_entries]
        selected_functions = st.multiselect(
            "Select functions to display", function_names, default=function_names
        )
        # Filter the selected functions
        return [f for f in filtered_entries if f["function"] in selected_functions]

    def select_function_search(filtered_entries: list[dict]) -> list[dict]:
        """Add a search bar for filtering functions by name."""
        search_query = st.text_input("Search functions")
        if search_query:
            filtered_entries = [
                f
                for f in filtered_entries
                if search_query.lower() in f["function"].lower()
            ]
        return filtered_entries

    ######################################################################
    ## add_sidebar
    ######################################################################

    def add_sidebar():
        """
        Make sidebar.

        You can add any content here: title, logo, menu, etc.
        """
        # Sidebar for controlling expanders and categories
        with st.sidebar:
            # Add title to the sidebar
            # st.title(
            #     "scikit-plots"
            # )
            # Add some text in the sidebar
            # st.write(
            #     "scikit-plots"
            # )
            # Allow user to choose a category
            # selected_category = select_category()
            # Allow user to choose multiple categories
            selected_categories = multiselect_categories()
            # Sidebar checkbox for controlling expanders
            expand_meta = st.checkbox("Expand all metadata", value=False)
            expand_live = st.checkbox("Expand all live interaction", value=False)
        # Sidebar for controlling expanders and categories
        with st.sidebar:
            st_add_sidebar_api_key()
        # Add a checkbox to the sidebar for demonstration
        # show_message = st.sidebar.checkbox("Show message", value=True)
        # if show_message:
        #     st.sidebar.write("Checkbox is checked!")
        return selected_categories, expand_meta, expand_live

    # def set_hide_sidebar():
    #     """hide_sidebar."""
    #     # st.sidebar.empty()
    #     st.session_state.sidebar_handler = ["collapsed"]
    #     st.rerun()

    ######################################################################
    ## CONTENT
    ## UI and plot encapsulated in fragment
    ## @st.fragment provides an implicit container; use st.container only for nested control
    ######################################################################

    # Redraw Minimization
    @st.fragment
    def render_metadata_section(
        function_meta: "dict[str, any]",
        expanded: bool = False,
        placeholder: "Optional[DeltaGenerator]" = None,
    ) -> None:
        """
        Render the metadata section for the function.

        Parameters
        ----------
        function_meta : dict
            Dictionary containing metadata.
        expanded : bool
            Whether the expander should be expanded by default.
        placeholder : st.delta_generator.DeltaGenerator | None
            Streamlit object.
        """
        # st.container A static layout block.
        # st.empty().container Dynamic and replaceable container.
        with (
            placeholder
            or st.container(
                height=None,
                border=None,
                key=f"metadata_container_{function_meta['function']}",
            ),
            st.expander(
                # Using st.expander with st.markdown inside it
                # $\color{color-code}{your-text-here} \space$
                # \Huge, \huge, \LARGE, \Large, \normalsize, \small, \tiny
                # \boldsymbol{...} — bolds the content inside.
                # \color{blue} — colors the content. oe :blue[text]
                # \boldsymbol{...} — only works for math symbols, not plain text.
                # Metadata Expander: Controlled by `expand_meta`
                # rf"👉 ${{\small \color{{blue}} \textbf{{[INFO]}}}}$ "
                label=(
                    r"${\small \color{blue} \textbf{\text{[INFO]}}}$ "
                    f"{function_meta['function'].rsplit('.')[-1]}"
                ),
                expanded=expanded,
                icon="👉",
            ),
        ):
            st.markdown(f"##### :blue-background[{function_meta['function']}]")
            st.markdown(f"**Fallback**: {function_meta['fallback_function']}")
            st.markdown(f"**Module**: {function_meta['module']}")
            st.markdown(f"**Description**: {function_meta['description']}")
            st.markdown(f"**Task Type**: {function_meta['task_type']}")
            st.markdown(f"**Plot Type**: {function_meta['plot_type']}")
            st.markdown(
                f"**Supervised**: {'✅' if function_meta['supervised'] else '❌'}"
            )
            st.markdown(
                f"**Explainability**: {function_meta['explainability_level'].capitalize()}"
            )
            # Display the example code snippet for the user
            ex_code = (
                f"{function_meta['function'].rsplit('.')[-1]}(\n"
                f"  {', '.join(function_meta['parameters'])}\n"
                ")"
            )
            st.code(
                ex_code,
                language="python",
                line_numbers=True,
                wrap_lines=True,
                height=None,
            )

    # Redraw Minimization
    @st.fragment
    def render_bot_message(
        function_meta: "dict[str, any]",
        idx: int = 0,
        msg: str = "Explain Plot...",
        placeholder: "Optional[DeltaGenerator]" = None,
    ) -> None:
        """
        Render a chatbot-style message placeholder.

        Parameters
        ----------
        function_meta : dict
            Function metadata dictionary.
        idx : int
            chatbot-style message plot index.
        msg : str
            chatbot-style message.
        placeholder : st.delta_generator.DeltaGenerator | None
            Streamlit object.
        """
        bot_chat_key = f"ask_ai_bot_{function_meta['function']}_{idx}"
        btn_chat_key = f"ask_ai_btn_{function_meta['function']}_{idx}"
        # bot_key = f"bot_msg_{function_meta['function']}"
        # Placeholder
        bot_placeholder = st.container(
            height=None,
            border=None,
            key=bot_chat_key,
        )
        with placeholder or bot_placeholder:
            # col1, col2 = st.columns([1, 9], vertical_alignment="center")
            # with col1:
            # Add a "Ask AI" button
            # you can use a unique identifier (function name, loop index, hash, etc.)
            if st.button(
                "Ask AI",
                icon=":material/forum:",
                use_container_width=True,
                key=btn_chat_key,
            ):
                with st.spinner("Response..."):
                    response = get_response(
                        st.session_state.messages,
                        model_provider=st.session_state.model_provider,
                        model_id=st.session_state.model_id,
                        api_key=st.session_state["api_key"],
                    )
                st.session_state[f"{function_meta['function']}_response"] = response
            # with col2:
            # st.chat_message("assistant").write("Hi there,")
            message = st.chat_message("assistant")
            message.write(
                st.session_state.get(f"{function_meta['function']}_response", msg)
            )
            # Separate visually
            # st.markdown("---")
            # st.divider()  # visually sleek
            # st.markdown("### 🔹 Begin Section A")
            # st.caption("🔻 Start of config")
            # st.toast(f"## Total clicks: {st.session_state.clicks}")

    # Redraw Minimization
    @st.fragment
    def render_plot_output(
        function_meta: "dict[str, any]",
        expanded: bool = False,
        placeholder: "Optional[DeltaGenerator]" = None,
    ) -> None:
        """
        Render the stored plots in the output placeholder.

        Parameters
        ----------
        function_meta : dict
            Metadata dict of the plotting function.
        expanded : bool, optional
            Whether to initially expand the live plot section.
        placeholder : st.delta_generator.DeltaGenerator | None
            Streamlit object.
        """
        # Display stored plot(s) (if available)
        for idx, fig in enumerate(st.session_state.get(function_meta["function"], [])):
            plot_key = f"plot_key_{function_meta['function']}_{idx}"
            # plot_key = f"plot_out_{function_meta['function']}"
            plot_placeholder = st.container(
                height=None,
                border=None,
                key=plot_key,
            )
            # Center
            col1, col2 = st.columns([7, 3], vertical_alignment="top")
            with placeholder or plot_placeholder:
                with col1:
                    # width to center area
                    _, mid, _ = st.columns([0.01, 0.985, 0.005])
                    with mid:
                        # plotting the figure
                        # fig.set_size_inches(6, 3.5)  # width=6 inches, height=3.5 inches
                        st.pyplot(
                            fig,
                            # clear_figure=None,
                            use_container_width=True,
                            dpi=150,
                        )
                with col2:
                    # width to center area
                    col3, _ = st.columns([0.99, 0.01])
                    with col3:
                        render_bot_message(function_meta, idx)

    # Redraw Minimization
    @st.fragment
    def display_function_details(
        function_meta: "dict[str, any]",
        expand_meta: bool = False,
        expand_live: bool = False,
        expand_all: bool = False,
        placeholder: "Optional[DeltaGenerator]" = None,
    ) -> None:
        """
        Display a complete section for a function, including metadata, plot controls, and output.

        Parameters
        ----------
        function_meta : dict
            Metadata about the function including keys like:
            - module: str
            - function: str
            - fallback_function: str
            - description: str
            - task_type: str
            - plot_type: str
            - supervised: bool
            - explainability_level: str
            - parameters: list of str
            - optional_parameters: dict
        expand_meta : bool, optional
            Whether to initially expand the metadata section.
        expand_live : bool, optional
            Whether to initially expand the live plot section.
        expand_all : bool, optional
            Whether to expand both sections by default.
        placeholder : st.delta_generator.DeltaGenerator | None
            Streamlit object.
        """
        # -------------------- Plot State --------------------
        ## Initialize session state with defaults (only once)
        # st.session_state.setdefault(function_meta["function"], [])
        if function_meta["function"] not in st.session_state:
            st.session_state[function_meta["function"]] = []

        # Outer container for modular rendering
        expan_label = f"▶️ Try it live - {function_meta['function'].rsplit('.')[-1]}"
        cont_key = f"live_container_{function_meta['function']}"
        btn_run_label = "Get Plot."
        btn_run_key = f"run_{function_meta['function']}"
        btn_clr_label = "Clear Plots!"
        btn_clr_key = f"clear_{function_meta['function']}"
        # Placeholder
        # you can use a unique identifier (function name, loop index, hash, etc.)
        live_placeholder = st.container(
            height=None,
            border=None,
            key=cont_key,
        )
        with placeholder or live_placeholder:
            # To place two buttons side-by-side (in the same horizontal row) in Streamlit
            # col1, mid, col2 = st.columns(3)
            col1, col2 = st.columns([0.5, 0.5])
            with (
                col1,
                st.expander(
                    # label=f"{function_meta['function'].rsplit('.')[-1]}",
                    label=expan_label,
                    expanded=(expand_all or expand_live),
                ),
            ):
                col3, col4 = st.columns([0.5, 0.5])
                with col3:
                    # Button to trigger plotting
                    # you can use a unique identifier (function name, loop index, hash, etc.)
                    if st.button(
                        btn_run_label,
                        icon=":material/order_play:",
                        use_container_width=True,
                        key=btn_run_key,
                        type="secondary",
                        # on_click=None,
                    ):
                        try:
                            # Example: if the function needs y_test and y_pred
                            y_true, y_pred, y_score = (
                                st.session_state["y_true"],
                                st.session_state["y_pred"],
                                st.session_state["y_score"],
                            )
                            # Dynamically import function
                            plot_func = get_plot_func(function_meta)
                            logger.info(f"{plot_func.__name__} function called.")
                            # 6 inches wide, 2 inches tall
                            # width=8 inches, height=6 inches
                            fig = plt.figure(
                                figsize=(
                                    function_meta.get("optional_parameters", {}).get(
                                        "figsize", (5, 2.5)
                                    )
                                ),
                            )
                            # fig, ax = plt.subplots(
                            #     figsize=(
                            #         function_meta.get(
                            #             "optional_parameters", {}
                            #         ).get("figsize", (5, 2.5))
                            #     ),
                            # )
                            # Show a spinner while the function runs or spinner decorator
                            with (
                                st.spinner("Generating plot...", show_time=True),
                                _lock,
                            ):
                                # Example: if the function needs y_test and y_pred
                                if function_meta["parameters"] == ["y_true", "y_score"]:
                                    # Plot with spinner
                                    plot_func(y_true, y_score, fig=fig)
                                    # Tight, small legend
                                    # ax.legend(fontsize=7)
                                    # Save to session state using unique key
                                    st.session_state[function_meta["function"]].append(
                                        fig
                                    )
                                elif function_meta["parameters"] == [
                                    "y_true",
                                    "y_pred",
                                ]:
                                    # Plot with spinner
                                    plot_func(y_true, y_pred, fig=fig)
                                    # Tight, small legend
                                    # ax.legend(fontsize=7)
                                    # Save to session state using unique key
                                    st.session_state[function_meta["function"]].append(
                                        fig
                                    )
                                else:
                                    st.warning(
                                        "Demo input for this function is not configured."
                                    )
                                    raise NotImplementedError
                                plt.legend(fontsize=7)  # sets the legend font size
                        except Exception as e:
                            st.error(f"Execution failed: {e}")
                with col4:
                    # Add a "Clear Plots" button
                    # you can use a unique identifier (function name, loop index, hash, etc.)
                    if st.button(
                        btn_clr_label,
                        icon=":material/delete:",
                        use_container_width=True,
                        key=btn_clr_key,
                    ):
                        # st.session_state.pop(fig_key, None)
                        del st.session_state[function_meta["function"]]
            # Right column for metadata
            with col2:
                render_metadata_section(
                    function_meta,
                    expanded=(expand_all or expand_meta),
                )
            # display chat and plot
            render_plot_output(function_meta)

    ######################################################################
    ## run Streamlit app
    ######################################################################

    def st_data_visualizer():
        """
        Launch the Streamlit scikit-plots Plot Explorer app.

        This function is the main entry point for the UI. It initializes
        the layout, handles user interaction, and displays metadata for
        all functions within the selected category.
        """
        ## Initialize session state with defaults (only once)
        # st.session_state.setdefault("running", False)
        if "running" not in st.session_state:
            # Initialize the flag in session state
            st.session_state["running"] = False

        ## Initialize session state with defaults (only once)
        if "dfs" not in st.session_state:
            st.session_state.dfs = st.session_state.get("dfs", get_sns_data())
        if "y_true" not in st.session_state:
            st.session_state["y_true"] = pd.Series()
        if "y_pred" not in st.session_state:
            st.session_state["y_pred"] = pd.Series()
        if "y_score" not in st.session_state:
            st.session_state["y_score"] = pd.Series()

        # (Optionally) set without login page
        if not st.session_state.get("authenticated", False):
            ## Initialize session state with defaults (only once)
            # st.session_state.setdefault("sidebar_state", "collapsed")
            if "sidebar_state" not in st.session_state:
                st.session_state.sidebar_state = "collapsed"

            ## Set up the page configuration
            ## layout ("centered" or "wide")
            ## initial_sidebar_state ("auto", "expanded", or "collapsed")
            ## Must be first Streamlit call
            st.set_page_config(
                page_title="scikit-plots",
                layout="wide",
                initial_sidebar_state=st.session_state.get(
                    "sidebar_state", "collapsed"
                ),
                menu_items={
                    "Report a bug": (
                        "https://github.com/scikit-plots/scikit-plots/issues"
                    ),
                    "Get Help": "https://scikit-plots.github.io/dev",
                    "About": "### Scikit-Plots Explorer\nBuilt with Streamlit",
                },
            )

        # Inject the CSS only once
        # st.markdown(
        #     """
        #     <style>
        #     div[data-testid="stVerticalBlock"] > div {
        #         /* overflow-y: auto; */
        #         overflow-y: hidden;
        #         overflow-x: hidden;
        #         border: 0px solid #CCC;
        #         border-radius: 0rem;
        #         padding: 0rem;
        #         margin: 0rem;
        #         box-sizing: border-box;
        #     }
        #     canvas {
        #         max-width: 100% !important;
        #         height: auto !important;
        #     }
        #     </style>
        # """,
        #     unsafe_allow_html=True,
        # )

        ## Call the sidebar function
        selected_categories, expand_meta, expand_live = add_sidebar()

        # To place two buttons side-by-side (in the same horizontal row) in Streamlit
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state["y_true"] = st.session_state.dfs.get(
                st.radio(
                    "Select Uploaded `y_true`:",
                    options=list(st.session_state.dfs.keys()),
                    index=0,  # None
                    help="Choose DataFrame for analysis (e.g. y_true, y_pred, y_score).",
                ),
                pd.Series(),
            )
            logger.info(st.session_state.y_true.shape)
        with col2:
            st.session_state["y_pred"] = st.session_state.dfs.get(
                st.radio(
                    "Select Uploaded `y_pred`:",
                    options=list(st.session_state.dfs.keys()),
                    index=1,
                    help="Choose DataFrame for analysis (e.g. y_true, y_pred, y_score).",
                ),
                pd.Series(),
            )
            logger.info(st.session_state.y_pred.shape)
        with col3:
            st.session_state["y_score"] = st.session_state.dfs.get(
                st.radio(
                    "Select Uploaded `y_score`:",
                    options=list(st.session_state.dfs.keys()),
                    index=2,
                    help="Choose DataFrame for analysis (e.g. y_true, y_pred, y_score).",
                ),
                pd.Series(),
            )
            logger.info(st.session_state.y_score.shape)

        ## Filter catalog to show only functions in the selected category
        ## filtered_entries = filter_by_category(selected_category)
        ## Filter catalog to show only functions in the selected categories
        filtered_entries = filter_by_categories(selected_categories)

        ## Add filter for selecting individual functions within the selected category
        selected_functions = select_functions(filtered_entries)

        ## Display details for each selected function
        ## -------------------- Render All --------------------
        for entry in selected_functions:
            # entry to function_meta
            display_function_details(entry, expand_meta, expand_live)

    # Run the app from command line
    if __name__ == "__main__":
        ## (Optionally) without login entry-point
        st_data_visualizer()
