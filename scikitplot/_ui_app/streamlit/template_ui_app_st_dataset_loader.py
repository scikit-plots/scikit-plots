# template_st_dataset_loader.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=broad-exception-caught
# pylint: disable=import-outside-toplevel
# pylint: disable=invalid-name

# ruff: noqa: PD901, N806

"""
template_st_dataset_loader_ui.
"""

# import petname
# petname.Generate(3, separator="-")  # e.g., 'green-fox-jump'
# import hashlib
import textwrap
import traceback
import uuid  # str(uuid.uuid4())

import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.model_selection import (
    train_test_split,
)

from scikitplot import logger
from scikitplot._compat.optional_deps import LazyImport
from scikitplot._datasets import (
    EXTENSION_LOADERS,
    load_data_meta,
    # load_data,
)

__all__ = []


def get_sns_data(
    load_sns: str = "iris",
    target: str = "species",
):
    """Fetc Data function."""
    import seaborn as sns

    # Load example data
    df = sns.load_dataset(load_sns)
    X = df.drop(columns=target)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = RandomForestClassifier().fit(X_train, y_train)
    y_score = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    result = {
        f"{load_sns} y_true": pd.Series(y_test, index=X_test.index),
        f"{load_sns} y_pred": pd.Series(y_pred, index=X_test.index),
        f"{load_sns} y_score": pd.DataFrame(
            y_score, index=X_test.index, columns=model.classes_
        ),
    }
    logger.info(f"seaborn {load_sns} data loaded.")
    return result


def total_memory_usage(obj):
    """
    Calculate total memory usage in MB of a DataFrame or Series.

    Handling both cases safely.
    """
    mem_usage = getattr(obj, "memory_usage", None)
    if mem_usage is None:
        # No memory_usage method
        return 0.0

    usage = mem_usage(deep=True)
    # DataFrame returns Series; sum it
    # Series returns int directly
    total_bytes = usage.sum() if isinstance(usage, pd.Series) else usage
    return total_bytes / 1e6  # Convert bytes to MB


# import streamlit as st
st = LazyImport("streamlit", package="streamlit")

# Use st.cache_data for immutable data and st.cache_resource for reusable, expensive resources
# Use @st.fragment to create modular, reusable UI blocks with proper state handling
if st:
    __all__ += [
        "st_dataset_loader",
    ]

    ######################################################################
    ## get data
    ######################################################################

    # Caches resources or objects like models, connections, etc.
    # @st.cache_resource
    # Caches pure functions that return data and don't have side effects
    # Data Fetching	Add @st.cache_data
    # https://docs.streamlit.io/develop/concepts/architecture/caching
    # Cache pure data (e.g., DataFrames, results), Assumes immutable return values
    @st.cache_data(show_spinner="Loading data...")
    def get_sns_data_cache():
        """Fetc Data function."""
        return get_sns_data()

    # ---------------------- Streamlit Module Interface ----------------------

    def clear_dfs():
        """Clear loaded data from Streamlit session state."""
        # Clear loaded data from Streamlit session state.
        # st.session_state["dfs"] = {}
        del st.session_state.dfs
        st.success("✅ Cleared loaded data(s) from memory.")

    def df_displayer(name, df):
        """Display dataframe."""
        st.markdown("---")
        st.markdown(f"### 📊 {name}")
        # Slider
        n_rows = st.slider(
            "Number of rows to show",
            1,
            len(df),
            10,
            key=str(uuid.uuid4()),
        )
        st.dataframe(getattr(df, "head", lambda n=10: None)(n_rows))
        st.markdown(f"- **Shape**: `{getattr(df, 'shape', None)}`")
        st.markdown(f"- **Columns**: {len(getattr(df, 'columns', []))}")
        st.write(f"- **Memory**: ~{total_memory_usage(df):.2f} MB")

        if st.checkbox(
            f"🔍 Show basic profiling for `{name}`",
            key=f"profiling_{name}",  # fix make unique
            value=False,
        ):
            try:
                desc_df = getattr(df, "describe", lambda **kwargs: pd.DataFrame())(
                    include="all"
                )
                st.write(desc_df)
                isnull_sum = (
                    getattr(df, "isnull", lambda: None)() or pd.Series()
                ).sum()
                st.write("Null values:", isnull_sum)
            except Exception:
                pass
        csv_data = getattr(df, "to_csv", lambda **kwargs: "")(index=False)
        st.download_button(
            label="💾 Download as CSV",
            data=csv_data,
            file_name=f"{name}_cleaned.csv",
            mime="text/csv",
        )

    def handle_database_load(db_uri, db_type, query):
        """handle_database_load."""
        try:
            dataset_meta = load_data_meta(db_uri, db_type=db_type, query=query)
            # to remove a key from a dictionary and return its value
            df = dataset_meta.pop("data", pd.DataFrame())  # noqa: PD901
            if not df.empty:
                store = {f"DB: {db_type}": df}
                st.session_state["dfs"] = {**store, **st.session_state["dfs"]}
                st.success("✅ Successfully Loaded Data.")
                st.write("**Metadata:**", dataset_meta)
        except Exception as e:
            st.write("Detected db_type:", db_type)
            st.write("Uploaded file:", db_uri)
            st.exception(f"❌ Database connection error: {e}")
            st.code(traceback.format_exc(), language="python")

    def handle_file_upload(file_path, query=""):
        """Handle file upload or URL input and return loaded data."""
        try:
            # 'streamlit.runtime.uploaded_file_manager.UploadedFile'
            upload_type = bool(getattr(file_path, "name", None))
            file_name = file_path.name if hasattr(file_path, "name") else str(file_path)
            dataset_meta = load_data_meta(
                file_path, upload_type=upload_type, query=query
            )
            # to remove a key from a dictionary and return its value
            df = dataset_meta.pop("data", pd.DataFrame())  # noqa: PD901
            if not df.empty:
                store = {file_name: df}
                st.session_state["dfs"] = {**store, **st.session_state["dfs"]}
                st.success("✅ Successfully Loaded Data.")
                st.write("**Metadata:**", dataset_meta)
        except Exception as e:
            st.write("Uploaded file:", file_path)
            st.exception(f"❌ Failed to load data: {e}")
            st.code(traceback.format_exc(), language="python")

    def st_upload_component(accept_multiple=False, sql_query=""):
        """
        Add a Streamlit component for uploading data files.

        Parameters
        ----------
        accept_multiple : bool
            Whether to allow multiple file uploads.
        sql_query : str
            query for database with pandas.

        Returns
        -------
        None
        """
        # Placeholder
        # st.container A static layout block.
        # st.empty().container Dynamic and replaceable container.
        with st.container(border=True):
            st.subheader("📤 Upload Dataset File(s)")
            file_types = list(EXTENSION_LOADERS.keys())
            accept_multiple = st.checkbox(
                "Enable multi-file upload",
                key="accept_multiple",
                value=False,
            )
            uploaded_files = st.file_uploader(
                label="Choose file(s) to upload:",
                type=file_types,
                accept_multiple_files=accept_multiple,
                help="Supported formats: " + ", ".join(EXTENSION_LOADERS.keys()),
            )
            # https://www.sqlitetutorial.net/sqlite-sample-database/
            # https://www.sqlitetutorial.net/tryit/#tracks
            # Download an example SQLite dataset, like the Sakila dataset, available on GitHub
            # https://github.com/jOOQ/sakila/blob/main/sqlite-sakila-db/sqlite-sakila-schema.sql
            # ✅ To Auto-Reset Even with Keys
            sql_query = st.text_area(
                (
                    "(Optional) SQL Query for DB files: Manually download and extract DB "
                    "chinook.db 0.8MB "
                    "https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip"
                ),
                textwrap.dedent(
                    """\
                    -- chinook https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip
                    SELECT * FROM tracks;""".rstrip(),
                ),
                # disabled = accept_multiple,
            )
            if uploaded_files:
                files_to_process = (
                    uploaded_files if accept_multiple else [uploaded_files]
                )
                for file in files_to_process:
                    handle_file_upload(file, sql_query)

    def st_url_component():
        """
        Add a Streamlit component for url data files.

        Returns
        -------
        None
        """
        # Placeholder
        with st.container(border=True):
            st.subheader("🗄️ Fetch Dataset File")
            file_url = st.text_input(
                (
                    "Enter A valid URL (e.g., https://raw.githubusercontent.com"
                    "/mwaskom/seaborn-data/master/iris.csv) OR A valid local DB "
                    "PATH Ensure file exist Under/Relative '$HOME' (e.g., ~/.db/chinook.db )"
                ),
                placeholder=(
                    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
                ),
                # value=(
                #     "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
                # ),
            ).strip()
            sql_query = st.text_area(
                "(Optional) SQL Query:",
                textwrap.dedent(
                    """\
                -- chinook https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip
                SELECT * FROM tracks;""".rstrip(),
                ),
            )
            if file_url:
                handle_file_upload(file_url, sql_query)

    def st_db_component():
        """
        Add a Streamlit component for db data files.

        Returns
        -------
        None
        """
        # Placeholder
        with st.container(border=True):
            st.subheader("🌐 Database Queried Dataset File")
            # connection_urls = {
            #     "duckdb": "duckdb:///path/to/file.duckdb",
            #     "mssql":
            #     "mssql+pyodbc://user:pass@host/dbname?driver=ODBC+Driver+17+for+SQL+Server",
            #     "mysql": "mysql+pymysql://user:pass@host/dbname",
            #     "postgresql": "postgresql://user:pass@host:port/dbname",
            #     "sqlite": "sqlite:///path/to/file.db",
            #     "sqlite": "oracle+cx_oracle://user:pass@host:port/dbname"
            # }
            # age = st.slider("Age", 0, 100, key="age")
            db_type = (
                st.selectbox(
                    "Database Type",
                    [
                        "SQLite",
                        "DuckDB",
                        "MSSQL",
                        "MySQL",
                        "PostgreSQL",
                        "Oracle",
                    ],
                )
                .lower()
                .strip()
            )
            msg = f"🔗 Enter a valid {db_type.upper()} DB URI (e.g., " + (
                "must under/relative 'HOME' -> "
                f"~/.db/chinook.{('db' if db_type == 'sqlite' else 'duckdb')} )"
                if db_type in ["duckdb", "sqlite"]
                else f"user:pass@host{('' if db_type.startswith('m') else ':port')}/dbname"
                f"{('?driver=ODBC+Driver+17+for+SQL+Server' if db_type == 'mssql' else '')} )"
            )
            db_uri = st.text_input(
                msg,
                placeholder=(
                    f"~/path/to/file.{db_type}"
                    if db_type in ["duckdb", "sqlite"]
                    else msg
                ),
            ).strip()
            sql_query = st.text_area(
                "SQL Query:",
                textwrap.dedent(
                    """\
                -- chinook https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip
                SELECT * FROM tracks;
                """.rstrip(),
                ),
            )
            if db_uri and db_type:
                handle_database_load(db_uri, db_type, sql_query)

    def st_dataset_loader():
        """
        Render the Streamlit UI for loading data.

        display df or dfs infos
        """
        ## Initialize session state with defaults (only once)
        # st.session_state.setdefault("dfs", {})
        if "dfs" not in st.session_state:
            st.session_state["dfs"] = get_sns_data_cache()

        # Placeholder
        with st.container(border=True):
            st.title("📁 Load Dataset File(s)")
            # st.subheader("📁 Load Dataset File(s)")
            st.caption("Supported extensions: " + ", ".join(EXTENSION_LOADERS.keys()))

        # Placeholder
        clear_btn_placeholder = st.container(border=True)
        # Placeholder
        dataset_loader_placeholder = st.container(border=True)

        with clear_btn_placeholder:
            if st.button(
                "🧹 Clear Loaded Data",
                icon=":material/delete:",
                # use_container_width=True,
                # on_click=clear_dfs,
            ):
                # Clear loaded data from Streamlit session state.
                del st.session_state.dfs
                # This effectively clears the container
                # by not rendering anything inside it
                dataset_loader_placeholder.empty()
                st.rerun()

        with dataset_loader_placeholder:
            # Mode selection
            # ---------------------- Mode Selection ----------------------
            input_opts = {
                "📁 Upload File(s) (CSV, Parquet, Pickle, etc.)": "upload",
                "🌐 Load dataset from a public URL (e.g. GitHub, S3, etc.)": "url",
                "🗄️ Query a Database (PostgreSQL, MySQL, Oracle, SQLite)": "db",
            }
            input_mode_label = st.radio(
                "Select your data source:",
                list(input_opts.keys()),
                help="Choose how you'd like to load your data for analysis.",
            )
            input_mode = input_opts[
                input_mode_label
            ]  # Map label to internal mode string

            # ---------------------- Handle Modes ----------------------
            if input_mode == "upload":
                st_upload_component()
            elif input_mode == "url":
                st_url_component()
            elif input_mode == "db":
                st_db_component()

        # Placeholder
        with st.container(border=True):
            # Show results
            if st.session_state.get("dfs", None):
                st.subheader("📑 Loaded Data Summary")
                for name, df in st.session_state["dfs"].items():
                    if not df.empty:
                        df_displayer(name, df)


# ---------------------- Main Entrypoint ----------------------

if __name__ == "__main__":
    st_dataset_loader()
