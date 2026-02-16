"""Developer Python template: notes for C++ bridge workflows.

This template is documentation-oriented. For actual C++ extension builds,
see the corresponding Cython template in the adjacent category.
"""

from __future__ import annotations


def notes() -> str:
    return (
        "For C++ integration, prefer building a small extension with language='c++' "
        "and linking extra_sources (e.g., .cpp). This devkit supports that."
    )
