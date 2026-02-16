# cython: language_level=3
"""
Multi-file template: include a `.pxi` helper.

User notes
----------
- Demonstrates `include "file.pxi"` (source inclusion).
- The helper file is shipped as template data and copied into the build
  directory automatically via metadata `support_paths`.

Developer notes
--------------
- Use `.pxi` for simple shared code snippets across templates.
"""

include "helper_square.pxi"

cpdef int square(int n):
    """Return n*n using an inlined helper in a `.pxi` file."""
    return _square_int(n)
