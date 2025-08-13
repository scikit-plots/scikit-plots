# pylint: disable=import-error
# pylint: disable=unused-import
# pylint: disable=unused-argument
# pylint: disable=no-name-in-module
# pylint: disable=import-outside-toplevel
# pylint: disable=broad-exception-caught
# pylint: disable=line-too-long

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""dataset_loader - Unified and extensible data loading module."""

import asyncio  # noqa: F401
import io
import mimetypes
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse
from urllib.request import urlopen

import pandas as pd

from .. import logger
from .._compat.optional_deps import safe_import

if TYPE_CHECKING:
    from typing import (  # noqa: F401
        IO,
        Callable,
        ClassVar,
        ModuleType,
        Optional,
        TypeVar,
        Union,
    )

    pandas = pd


# ---------------------- Helpers ----------------------


def detect_file_type(
    url_or_path,
    strict: bool = True,
) -> tuple[str | None, str | None]:
    """
    Python is used to guess the MIME type of a file.

    Based on its filename or URL, particularly its extension.
    - No extension	Returns (None, None)
    - mimetypes.guess_type("data/file.csv")
    - mimetypes.guess_type("https://example.com/file.csv")
    - # Output: ('text/csv', None)
    """
    # Uploaded file like streamlit
    if hasattr(url_or_path, "name"):
        mime_type, _ = mimetypes.guess_type(url_or_path.name)
    else:
        mime_type, _ = mimetypes.guess_type(url_or_path)
    logger.info(f"detect_file_type: {mime_type}")
    return mime_type  # e.g., 'text/csv', 'application/vnd.ms-excel'


def get_extension(file_name: str) -> str:
    """Extract the lowercase file extension."""
    # Uploaded file like streamlit
    # if hasattr(file_name, "name"):
    file_name = getattr(file_name, "name", file_name)
    # extension = file_name.name.split(".")[-1].lower()
    # extension = Path(file_name).suffix.lower()
    # List of all suffixes (e.g., ['.csv', '.gz'])
    extension = Path(file_name).suffixes
    # Single extension types
    extension = "".join(extension).lower()
    logger.info(f"get_extension: {extension}")
    return extension


def is_url(path: str) -> bool:
    """Check if the provided path is a URL."""
    # Determine if the given path is a URL
    try:
        result = urlparse(str(path))
        return result.scheme in ("http", "https")
    except Exception:
        return False


def get_file_from_zip(path):
    """get_file_from_zip."""
    # Handle zip archive: try extracting the first supported file
    with zipfile.ZipFile(path, "r") as zip_ref:
        file_list = [f for f in zip_ref.namelist() if not f.endswith("/")]
        if not file_list:
            raise ValueError("Zip archive contains no files")
        # TODO: Add logic for selection or default to first supported
        # Extract and return a file-like object
        return zip_ref.open(file_list[0])


def get_file_data(
    path: "Union[str, Path, io.BytesIO]", compression: "Optional[str]" = None
):
    """
    Retrieve a file-like object based on file path, URL, or stream.

    Parameters
    ----------
    path : str | Path | BytesIO
        File path, URL, or byte stream.
    compression : Optional[str]
        Compression type.

    Returns
    -------
    File-like object.
    """
    # Retrieve file-like object based on file type or URL
    if is_url(path):
        # Load from URL
        # with urllib.request.urlopen(url=str(path)) as response:
        response = urlopen(url=str(path))  # noqa: S310
        return io.BytesIO(response.read())
    elif isinstance(path, io.BytesIO):  # noqa: RET505
        # Already a byte stream
        return path
    elif str(path).endswith(".gz"):
        return safe_import("gzip").open(path, "rb")
    elif str(path).endswith(".bz2"):
        return safe_import("bz2").open(path, "rb")
    elif str(path).endswith(".zip"):
        get_file_from_zip(path)
    # Default binary open
    return open(path, "rb")  # noqa: PTH123


# ---------------------- Unified Loader ----------------------


def load_file_by_extension(file_obj, extension: str, **kwargs):
    """Generalized loader dispatcher based on file extension."""
    try:
        return default_loader(file_obj, extension, **kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load file with extension {extension}: {e}"
        ) from e


def load_stream_or_path(source, extension: str, **kwargs):
    """Load data from a stream, path, or uploaded file."""
    file_obj = (
        get_file_data(source)
        if isinstance(source, (str, Path)) and not is_url(source)
        else (
            io.BytesIO(source.read())  # like Streamlit uploads
            if hasattr(source, "read")
            else source
        )
    )
    return load_file_by_extension(file_obj, extension, **kwargs)


def load_uploaded_file(uploaded_file):
    """
    Load supported file types from Streamlit uploaded file.

    Parameters
    ----------
    uploaded_file : UploadedFile
        File uploaded via Streamlit.

    Returns
    -------
    pd.DataFrame or object: Loaded content.
    """
    ext = get_extension(uploaded_file.name)
    return load_stream_or_path(uploaded_file, ext)


def load_file_from_path(file_path):
    """
    Load supported file types from a local file path.

    Parameters
    ----------
    file_path : str
        Path to the file.

    Returns
    -------
    pd.DataFrame or object: Loaded content.
    """
    ext = get_extension(file_path)
    return load_stream_or_path(file_path, ext)


def load_file_from_url(url):
    """
    Load a file from a URL. Supports CSV, Parquet, Pickle, Excel.

    Parameters
    ----------
    url : str
        URL to the file.

    Returns
    -------
    pd.DataFrame or object: Loaded content.
    """
    ext = "." + url.lower().split("?")[0].split(".")[-1]
    return load_stream_or_path(url, ext)


# ---------------------- File Loaders ----------------------


def default_loader(file_like, extension, **kwargs):
    """Load data from a file-like object based on its extension."""
    if extension in EXTENSION_LOADERS:
        return EXTENSION_LOADERS.get(extension)(file_like, **kwargs)
    raise ValueError(f"Unsupported extension: {extension}")


# Simplified using safe_import and get_file_data
def make_loader(module, attr):
    """make_loader."""

    def loader(path, **kwargs):
        # pylint: disable=unnecessary-dunder-call
        return safe_import(module).__getattribute__(attr)(get_file_data(path), **kwargs)

    return loader


# pylint: disable=unnecessary-lambda-assignment
load_numpy = lambda path, **kwargs: safe_import("numpy").load(  # noqa: E731
    get_file_data(path), **kwargs
)
# pandas
load_csv = make_loader("pandas", "read_csv")
load_txt = lambda path, **kwargs: safe_import("pandas").read_csv(  # noqa: E731
    get_file_data(path), sep="\t", **kwargs
)
load_parquet = make_loader("pandas", "read_parquet")
load_excel = make_loader("pandas", "read_excel")
load_json = make_loader("pandas", "read_json")
load_feather = make_loader("pandas", "read_feather")
load_pandas_pickle = make_loader("pandas", "read_pickle")
# pyarrow
# import pyarrow.parquet as pq
# Convert to pandas DataFrame if needed
# load_pyarrow_parquet = make_loader("pyarrow.parquet", "read_table").to_pandas()
# pickle
load_pickle = lambda path, **kwargs: safe_import("pickle").load(  # noqa: E731
    get_file_data(path)
)  # noqa: ARG005
load_joblib = lambda path, **kwargs: safe_import("joblib").load(  # noqa: E731
    get_file_data(path)
)  # noqa: ARG005
load_cloudpickle = lambda path, **kwargs: safe_import(  # noqa: E731
    "cloudpickle"
).load(  # noqa: ARG005
    get_file_data(path)
)

# ---------------------- Database Loaders ----------------------


def clean_sql(query: str, remove_inline_comments: bool = True) -> str:
    """
    Clean SQL query by removing comments: single-line, inline, and block.

    Parameters
    ----------
    query : str
        Raw SQL query string.
    remove_inline_comments : bool, optional
        If True, removes inline `--` and block `/* */` comments. Default is True.

    Returns
    -------
    str
        Cleaned SQL string, safe for execution.
    """
    if not isinstance(query, str):
        raise TypeError("Query must be a string.")

    # Remove block comments (/* ... */), including multiline
    query = re.sub(r"/\*.*?\*/", "", query, flags=re.DOTALL)

    cleaned_lines = []
    for line in query.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        if remove_inline_comments:
            line = re.split(r"--", line, maxsplit=1)[0].rstrip()  # noqa: PLW2901
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def load_duckdb(path: "Union[str, Path]", query: str = "SELECT 1;", **kwargs):
    """
    Load data from a DuckDB database file 'path/to/file.db'.

    DuckDB Alternative (no SQLAlchemy).
    """
    duckdb = safe_import("duckdb")
    pd = safe_import("pandas")
    db_path = Path(path).expanduser().resolve()
    conn = duckdb.connect(database=str(db_path), read_only=True)
    try:
        return pd.read_sql_query(clean_sql(query), conn, **kwargs)
    finally:
        conn.close()


def load_sqlite(path: "Union[str, Path]", query: str = "SELECT 1;", **kwargs):
    """
    load_sqlite 'path/to/file.db'.

    SQLite Alternative (no SQLAlchemy Needs `sqlite:///`).
    sqlite3 does NOT support remote (hosted) databases — it's strictly local file-based.

    +----------------------------+---------------------------------------------------------------+
    | Connection string          | Meaning                                                       |
    +============================+===============================================================+
    | sqlite://                  | In-memory SQLite database (temporary, disappears when closed) |
    +----------------------------+---------------------------------------------------------------+
    | sqlite:///path.db          | SQLite DB stored at relative path ``path.db``                 |
    +----------------------------+---------------------------------------------------------------+
    | sqlite:////full/path.db    | SQLite DB stored at absolute path ``/full/path.db``           |
    +----------------------------+---------------------------------------------------------------+

    +---------------------+------------------------------------------+-----------------------------------------+
    | Function            | Purpose                                  | Allows DML/DDL (e.g., INSERT, DROP)     |
    +=====================+==========================================+=========================================+
    | pd.read_sql()       | General-purpose: SELECT, or even DDL/DML | ✅ (but not recommended for non-SELECT) |
    +---------------------+------------------------------------------+-----------------------------------------+
    | pd.read_sql_query() | Only for SELECT-type queries             | ❌ (raises error on non-SELECT)         |
    +---------------------+------------------------------------------+-----------------------------------------+
    """
    db_path = Path(path).expanduser().resolve()
    conn = safe_import("sqlite3").connect(db_path)
    return safe_import("pandas").read_sql_query(clean_sql(query), conn)


def preview_sqlite_tables(path):
    """preview_sqlite_tables 'path/to/file.db'."""
    db_path = Path(path).expanduser().resolve()
    try:
        conn = safe_import("sqlite3").connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables
    except Exception:
        return []


def load_sqlalchemy(
    path: str, db_type: str = "postgresql", query: str = "SELECT 1;", **kwargs
) -> "pandas.DataFrame":  # noqa: F821
    """
    Load data from a database using SQLAlchemy and return as a pandas DataFrame.

    Parameters
    ----------
    path : str
        Connection path or full URL depending on db_type.
        - For SQLite/DuckDB: path to database file.
        - For others: `user:pass@host:port/dbname`
    db_type : str
        One of: 'sqlite', 'postgresql', 'mysql', 'oracle', 'mssql', 'duckdb'.
    query : str
        SQL query to execute.
    **kwargs : dict
        Additional kwargs passed to pandas.read_sql().

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    +------------+---------------------------------------------------------------------------+
    | Database   | Connection URL format                                                     |
    +============+===========================================================================+
    | DuckDB     | duckdb:///path/to/file.duckdb                                             |
    +------------+---------------------------------------------------------------------------+
    | MSSQL      | mssql+pyodbc://user:pass@host/dbname?driver=ODBC+Driver+17+for+SQL+Server |
    +------------+---------------------------------------------------------------------------+
    | MySQL      | mysql+pymysql://user:pass@host/dbname                                     |
    +------------+---------------------------------------------------------------------------+
    | PostgreSQL | postgresql://user:pass@host:port/dbname                                   |
    +------------+---------------------------------------------------------------------------+
    | SQLite     | sqlite:///path/to/file.db                                                 |
    +------------+---------------------------------------------------------------------------+
    | Oracle     | oracle+cx_oracle://user:pass@host:port/dbname                             |
    +------------+---------------------------------------------------------------------------+

    +----------------------------+---------------------------------------------------------------+
    | Connection string          | Meaning                                                       |
    +============================+===============================================================+
    | sqlite://                  | In-memory SQLite database (temporary, disappears when closed) |
    +----------------------------+---------------------------------------------------------------+
    | sqlite:///path.db          | SQLite DB stored at relative path ``path.db``                 |
    +----------------------------+---------------------------------------------------------------+
    | sqlite:////full/path.db    | SQLite DB stored at absolute path ``/full/path.db``           |
    +----------------------------+---------------------------------------------------------------+

    - sqlite:///relative/path.db, sqlite:////absolute/path.db

    +---------------------+------------------------------------------+-----------------------------------------+
    | Function            | Purpose                                  | Allows DML/DDL (e.g., INSERT, DROP)     |
    +=====================+==========================================+=========================================+
    | pd.read_sql()       | General-purpose: SELECT, or even DDL/DML | ✅ (but not recommended for non-SELECT) |
    +---------------------+------------------------------------------+-----------------------------------------+
    | pd.read_sql_query() | Only for SELECT-type queries             | ❌ (raises error on non-SELECT)         |
    +---------------------+------------------------------------------+-----------------------------------------+


    Examples
    --------
    # SQLite
    load_sqlalchemy("data/mydb.sqlite", db_type="sqlite")

    # DuckDB
    load_sqlalchemy("data/mydb.duckdb", db_type="duckdb")

    # MSSQL
    load_sqlalchemy("user:pass@host/dbname?driver=ODBC+Driver+17+for+SQL+Server", db_type="mssql")

    # MySQL
    load_sqlalchemy("user:pass@localhost/mydb", db_type="mysql")

    # PostgreSQL
    load_sqlalchemy("user:pass@localhost:5432/mydb", db_type="postgresql")

    # Oracle
    load_sqlalchemy("user:pass@host:1521/orcl", db_type="oracle")
    """
    sa = safe_import("sqlalchemy")
    pd = safe_import("pandas")
    db_type = db_type.lower()
    engines = {
        "mssql": f"mssql+pyodbc://{path}",
        "mysql": f"mysql+pymysql://{path}",
        "postgresql": f"postgresql://{path}",
        "oracle": f"oracle+cx_oracle://{path}",
    }
    if db_type in ["duckdb", "sqlite"]:
        # Expand ~ and get absolute path
        # db_path = Path.home() / ".db" / path
        db_path = Path(path).expanduser().resolve()
        # Ensure ~/.db exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Add new entries safely
        engines.update(
            {
                # sqlite:///relative/path.db
                # sqlite:////absolute/path.db
                "sqlite": f"sqlite:///{db_path}",
                "duckdb": f"duckdb:///{db_path}",
            }
        )
    if db_type not in engines:
        raise ValueError(f"Unsupported SQLAlchemy DB type: {db_type}")
    engine = sa.create_engine(engines[db_type])
    # Test connection to a database using SQLAlchemy.
    with engine.connect() as conn:
        # Read the SQL file or Query
        # Run and load into DataFrame a single SELECT statement.
        return pd.read_sql_query(clean_sql(query), conn)
        # conn.execute(sa.text(query))  # use `text()` to safely wrap SQL
        # conn.commit()
        # return None


async def async_load_sqlalchemy(
    path: str,
    db_type: str = "postgresql",
    query: str = "SELECT 1;",
) -> "pandas.DataFrame":  # noqa: F821
    """
    Async version of load_sqlalchemy using SQLAlchemy 2.0 async engine.

    Parameters
    ----------
    path : str
        Connection path or full URL depending on db_type.
    db_type : str
        One of: 'sqlite', 'postgresql', 'mysql'.
    query : str
        SQL query to execute.

    Returns
    -------
    pd.DataFrame
    """
    sa = safe_import("sqlalchemy")
    pd = safe_import("pandas")
    db_type = db_type.lower()

    if db_type == "sqlite":
        conn_str = f"sqlite+aiosqlite:///{Path(path).absolute()}"
    elif db_type == "postgresql":
        conn_str = f"postgresql+asyncpg://{path}"
    elif db_type == "mysql":
        conn_str = f"mysql+aiomysql://{path}"
    else:
        raise ValueError(f"[async_load_sqlalchemy] Unsupported db_type: {db_type}")

    engine = sa.ext.asyncio.create_async_engine(conn_str, future=True)

    async with engine.connect() as conn:
        result = await conn.execute(sa.text(query))
        rows = result.fetchall()
        return pd.DataFrame(rows, columns=result.keys())


# ---------------------- Handle an uploaded ----------------------


def upload_handler(
    file_obj,
    return_file: bool = True,
    clean_tmp: bool = False,
    loader_func: "Optional[Callable[[str, Optional[str]], dict[str, any]]]" = None,
    query: "Optional[str]" = None,
    db_type: "Optional[str]" = None,
) -> "Union[dict[str, any], str, None]":
    """
    Handle an uploaded file object.

    - Writes it to a temporary file.
    - If `return_file=True`, immediately returns the temp file path.
    - If `return_file=False`, attempts to load data using `loader_func` or `load_data()`.

    Parameters
    ----------
    file_obj : file-like
        A file-like object with `.name` and `.read()` attributes.
    return_file : bool, default True
        If True, return path to the saved temp file immediately after writing.
    clean_tmp : bool, default False
        If True, deletes the temp file after loading (or failed load).
    loader_func : callable, optional
        Optional custom loader function.
        Fallbacks to `load_data()`.
    query : str, optional
        Optional query string passed to the
        default loader `load_data()` not passed to `loader_func()`.
    db_type : str, optional
        Optional ext "sqlite" or "duckdb" default depends on `file_obj` extension.

    Returns
    -------
    dict
        If `return_file=False`, returns the result of the loader function.
    str
        If `return_file=True`, returns the path to the temp file.
    None
        If loading fails or invalid input is provided.
    """
    logger.info(f"upload_handler: {type(file_obj)}")
    logger.info(f"upload_handler: {file_obj!r}")
    if not (hasattr(file_obj, "name") and callable(getattr(file_obj, "read", None))):
        raise NotImplementedError(
            "Provided object is not a supported file-like object."
        )

    file_name = getattr(file_obj, "name", "uploaded_file")
    file_suffix = db_type or os.path.splitext(file_name)[-1] or ""  # noqa: PTH122

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
            tmp.write(file_obj.read())
            tmp.flush()
            tmp_path = tmp.name
            logger.info(f"upload_handler: {tmp_path}")

        if return_file:
            return tmp_path

        # If not returning the file, attempt to load
        if loader_func:
            result = loader_func(tmp_path)
        else:
            result = load_data(tmp_path, query=clean_sql(query))
        return result

    except Exception:
        # Optionally log or raise depending on use case
        # print(f"[upload_handler] Failed to process file: {e}")
        return tmp_path if return_file else None
    finally:
        if clean_tmp and "tmp_path" in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------- Extension Dispatcher ----------------------

EXTENSION_LOADERS = {
    ".npy": load_numpy,
    # pandas
    ".zip": load_csv,  # assume zipped CSV as a fallback
    ".csv": load_csv,
    ".txt": load_txt,
    ".parquet": load_parquet,
    ".xls": load_excel,
    ".xlsx": load_excel,
    ".json": load_json,
    ".feather": load_feather,
    ".pd.pkl": load_pandas_pickle,
    ".pandas.pkl": load_pandas_pickle,
    # pickle
    ".pkl": load_pickle,
    ".pickle": load_pickle,
    ".joblib": load_joblib,
    ".cloudpkl": load_cloudpickle,
    ".cloudpickle": load_cloudpickle,
    # Upload Database fetch sql query
    ".db": load_sqlite,  # just path db
    ".sqlite": load_sqlite,  # just path db
    ".sqlite3": load_sqlite,  # just path db
    ".duckdb": load_duckdb,  # just path db
    # Connect Database fetch sql query
    ".oracle.sql": lambda path, **kwargs: load_sqlalchemy(
        path, db_type="oracle", **kwargs
    ),
    ".mysql.sql": lambda path, **kwargs: load_sqlalchemy(
        path, db_type="mysql", **kwargs
    ),
    ".postgres.sql": lambda path, **kwargs: load_sqlalchemy(
        path, db_type="postgresql", **kwargs
    ),
}

# ---------------------- Extension API ----------------------


def register_loader(extension: str, func, override: bool = False):
    """register_loader."""
    # Allow user to register custom loader functions
    ext = extension.lower()
    if ext in EXTENSION_LOADERS and not override:
        raise ValueError(
            f"Loader for '{ext}' already exists. Use override=True to overwrite."
        )
    EXTENSION_LOADERS[ext] = func


def get_loader_by_ext(ext: str) -> "Callable":
    """get_loader_by_ext."""
    return EXTENSION_LOADERS.get(
        ext.lower(),
        lambda path, **kwargs: load_stream_or_path(path, ext, **kwargs),
    )


# ---------------------- load_data ----------------------


def load_data_meta(
    path: "Union[str, Path]",
    extension: "Optional[str]" = None,
    upload_type: "Optional[str]" = None,
    db_type: "Optional[str]" = None,
    query: "Optional[str]" = None,
    **kwargs,
) -> dict:
    """Unified high-level data loading with meta interface."""
    # Generalized data loading interface with file type inference
    logger.info(f"load_data: {type(path)}")
    logger.info(f"load_data: {path!r}")
    if db_type:
        data = load_sqlalchemy(
            str(path).strip(),
            db_type=db_type,
            query=clean_sql(query) or "SELECT 1;",
            **kwargs,
        )
    else:
        if upload_type:
            path = upload_handler(path)
        ext = extension or get_extension(path)
        # Normalize ext
        ext = "." + ext.lower().lstrip(".")
        loader = get_loader_by_ext(ext)
        data = (
            # if loadable data load
            loader(path, **kwargs)
            if ext
            not in [
                ".db",
                ".sqlite",
                ".duckdb",
            ]
            # If sqlite db get table
            else loader(path, query=clean_sql(query), **kwargs)
        )
    # "meta": {"source": str(path), "rows": len(df), "columns": len(df.columns)},
    return {
        "data": data,
        "shape": getattr(data, "shape", None),
        "source": str(path),
        "is_url": is_url(path),
        "type": db_type or ext,
        "query": (
            (
                db_type
                or ext
                in [
                    ".db",
                    ".sqlite",
                    ".duckdb",
                ]
            )
            and str(clean_sql(query))
        ),
    }


def load_data(
    path: "Union[str, Path]",
    extension: "Optional[str]" = None,
    upload_type: "Optional[str]" = None,
    db_type: "Optional[str]" = None,
    query: "Optional[str]" = None,
    **kwargs,
) -> "pandas.DataFrame":
    """Unified high-level data loading interface."""
    return load_data_meta(
        path,
        extension,
        upload_type,
        db_type,
        query,
        **kwargs,
    ).get("data", pd.DataFrame())
