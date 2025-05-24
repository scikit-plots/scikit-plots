"""
template_st_data_loader.

├── template_st_data_loader.py        ← this file
└── .streamlit/
    └── config.toml
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=unused-import
# pylint: disable=unused-argument
# pylint: disable=no-name-in-module
# pylint: disable=broad-exception-caught

# template_st_data_loader.py
# template_st_data_uploader_ui.py

import textwrap
import traceback

import pandas as pd

from scikitplot._compat.optional_deps import HAS_STREAMLIT, safe_import
from scikitplot._datasets import EXTENSION_LOADERS, load_data

SUPPORTED_TYPES = EXTENSION_LOADERS

if HAS_STREAMLIT:
    st = safe_import("streamlit")

    # ---------------------- Streamlit Module Interface ----------------------

    def clear_loaded_data():
        """Clear loaded data from Streamlit session state."""
        st.session_state["loaded_data"] = {}
        st.success("✅ Cleared loaded data from memory.")

    def display_dataframe(name, df):
        """Display dataframe."""
        st.markdown("---")
        st.markdown(f"### 📊 {name}")
        n_rows = st.slider("Number of rows to show", 1, len(df), 10)
        st.dataframe(getattr(df, "head", lambda n=10: None)(n_rows))
        st.markdown(f"- **Shape**: `{getattr(df, 'shape', None)}`")
        st.markdown(f"- **Columns**: {len(getattr(df, 'columns', []))}")
        memory_mb = (
            getattr(df, "memory_usage", lambda **kwargs: pd.Series([]))(deep=True).sum()
        ) / 1e6
        st.write(f"- **Memory**: ~{memory_mb:.2f} MB")

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
            data_info = load_data(db_uri, db_type=db_type, query=query)
            df = data_info.get("data", pd.DataFrame())  # noqa: PD901
            store = {f"DB: {db_type}": df}
            if store:
                st.session_state["loaded_data"].update(store)
                st.success("✅ Loaded Data from Database.")
                # st.write("**Metadata:**", data_info.get("meta", {}))
        except Exception as e:
            st.write("Uploaded file:", db_uri)
            st.write("Detected db_type:", db_type)
            st.error(f"❌ Database connection error: {e}")
            st.exception(traceback.format_exc())

    def handle_file_upload(file_path, query=""):
        """Handle file upload or URL input and return loaded data."""
        try:
            # 'streamlit.runtime.uploaded_file_manager.UploadedFile'
            upload_type = bool(getattr(file_path, "name", None))
            data_info = load_data(file_path, upload_type=upload_type, query=query)

            df = data_info.get("data", pd.DataFrame())  # noqa: PD901
            store = {
                (file_path.name if hasattr(file_path, "name") else str(file_path)): df
            }
            if store:
                st.session_state["loaded_data"].update(store)
                st.success("✅ Loaded Data from upload or URL.")
                # st.write("**Metadata:**", data_info.get("meta", {}))
        except Exception as e:
            st.write("Uploaded file:", file_path)
            st.error(f"❌ Failed to load data: {e}")
            st.exception(traceback.format_exc())

    def uploader_component(accept_multiple=False, sql_query=""):
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
        list: List of (file name, loaded content) tuples.
        """
        # Placeholder
        with st.empty().container(border=True):
            st.subheader("📤 Upload Dataset File(s)")
            file_types = list(SUPPORTED_TYPES.keys())
            accept_multiple = st.checkbox(
                "Enable multi-file upload",
                key="accept_multiple",
                value=False,
            )
            uploaded_files = st.file_uploader(
                label="Choose file(s) to upload:",
                type=file_types,
                accept_multiple_files=accept_multiple,
                help="Supported formats: " + ", ".join(SUPPORTED_TYPES.keys()),
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

    def run_data_loader_ui():
        """
        Render the Streamlit UI for loading data.

        return df or dfs infos
        """
        ## Initialize session state with defaults (only once)
        # st.session_state.setdefault("loaded_data", {})
        if "loaded_data" not in st.session_state:
            st.session_state.loaded_data = {}

        st.session_state.display_dataframe = display_dataframe

        # Placeholder
        with st.empty().container(border=True):
            st.title("📁 Load Dataset File(s)")
            # st.subheader("📁 Load Dataset File(s)")
            st.caption("Supported extensions: " + ", ".join(SUPPORTED_TYPES.keys()))

        # Placeholder
        clear_btn_placeholder = st.empty().container(border=True)
        # Placeholder
        data_loader_placeholder = st.empty().container(border=True)

        with clear_btn_placeholder:
            if st.button(
                "🧹 Clear Loaded Data",
                icon=":material/delete:",
                # on_click=clear_loaded_data,
            ):
                # # Clear loaded data from Streamlit session state.
                # st.session_state["loaded_data"] = {}
                # st.success("✅ Cleared loaded data from memory.")
                del st.session_state.loaded_data
                # This effectively clears the container by not rendering anything inside it
                data_loader_placeholder.empty()
                st.rerun()

        with data_loader_placeholder:
            # Mode selection
            # ---------------------- Mode Selection ----------------------
            input_opts = {
                "🌐 Load dataset from a public URL (e.g. GitHub, S3, etc.)": "url",
                "📁 Upload File(s) (CSV, Parquet, Pickle, etc.)": "upload",
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
                uploader_component()

            elif input_mode == "url":
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

            elif input_mode == "db":
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

        # Placeholder
        with st.empty().container(border=True):
            # Show results
            if st.session_state.get("loaded_data", None):
                st.subheader("📑 Loaded Data Summary")
                for name, df in st.session_state["loaded_data"].items():
                    display_dataframe(name, df)


# ---------------------- Main Entrypoint ----------------------

if __name__ == "__main__":
    run_data_loader_ui()
