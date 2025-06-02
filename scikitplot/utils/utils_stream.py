"""stream_utils."""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=no-name-in-module

from collections.abc import Generator
from typing import Union

from .._compat.optional_deps import HAS_STREAMLIT, safe_import


def streamlit_stream_or_return(
    content: Union[str, Generator[str, None, None]],
) -> Union[str, None]:
    """
    Conditionally stream content in Streamlit if streaming is available.

    The content is an iterable (but not a string). Otherwise, return content as-is.

    Checks:
    - HAS_STREAMLIT is enabled
    - Streamlit object `st` has a `write_stream` method
    - content is iterable but not a string

    Parameters
    ----------
    content : Any
        The content to stream or return.

    Returns
    -------
    str or None
        The streamed content if streaming happens, else the original content.
    """
    if HAS_STREAMLIT:
        st = safe_import("streamlit")
        if (
            hasattr(st, "write_stream")
            and hasattr(content, "__iter__")
            and not isinstance(content, str)
        ):
            return st.write_stream(content)
    return content
