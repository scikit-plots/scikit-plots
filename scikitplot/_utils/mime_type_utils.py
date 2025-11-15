"""mime_type_utils.py."""

import os as _os
import pathlib as _pathlib
from mimetypes import guess_type


# TODO: Create a module to define constants to avoid circular imports
#  and move MLMODEL_FILE_NAME and MLPROJECT_FILE_NAME in the module.
def get_text_extensions():
    """get_text_extensions."""
    exts = [
        "txt",
        "log",
        "err",
        "cfg",
        "conf",
        "cnf",
        "cf",
        "ini",
        "properties",
        "prop",
        "hocon",
        "toml",
        "yaml",
        "yml",
        "xml",
        "json",
        "js",
        "py",
        "py3",
        "csv",
        "tsv",
        "md",
        "rst",
    ]
    return exts  # noqa: RET504


def _guess_mime_type(file_path):
    filename = _pathlib.Path(file_path).name
    extension = _os.path.splitext(filename)[-1].replace(".", "")  # noqa: PTH122
    # for MLmodel/mlproject with no extensions
    if extension == "":
        extension = filename
    if extension in get_text_extensions():
        return "text/plain"
    mime_type, _ = guess_type(filename)
    if not mime_type:
        # As a fallback, if mime type is not detected, treat it as a binary file
        return "application/octet-stream"
    return mime_type
