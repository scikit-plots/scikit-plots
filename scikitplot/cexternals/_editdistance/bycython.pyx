# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# distutils: language=c++
#
# scikitplot/cexternals/_editdistance/bycython.pyx

"""
Cython bindings for the bundled Levenshtein kernels.

Notes
-----
**User.** Elements are compared the way Python containers compare them:
identity first, then ``==``. Two strings compare by code point. Any other
sequence may hold any hashable elements (integers, tuples, mixed types);
``1 == 1.0`` counts as a match, two distinct ``float("nan")`` objects do not,
and the same NaN object does. An unhashable element raises ``TypeError``.

**Developer.** The kernels compare 64-bit integer codes. Upstream fed them
``hash(element)``, so elements whose hashes collide were treated as equal:
``hash(-1) == hash(-2)`` and ``hash(0) == hash(2**61 - 1)`` in CPython, which
made ``eval([-1], [-2])`` return 0. Codes are now assigned exactly:

* two ``str`` arguments are coded by code point, which is injective;
* otherwise both arguments are coded through **one** dict shared by the pair,
  so equal elements (as a dict decides equality) share a code and unequal
  elements never do.

Buffers are C++ ``vector`` objects owned by the calling frame. Upstream used
``malloc`` then ``hash()``; a raising ``hash`` leaked the first buffer (and a
``NULL`` from ``malloc`` was written through). Destructors now run on every
exit path, including a Python exception, and ``bad_alloc`` becomes
``MemoryError``.
"""

from libc.limits cimport UINT_MAX
from libc.stdint cimport int64_t
from libcpp cimport bool
from libcpp.vector cimport vector

cdef extern from "./_editdistance.h":
    unsigned int edit_distance(
        const int64_t *a, const unsigned int asize,
        const int64_t *b, const unsigned int bsize,
    ) except +
    bool edit_distance_criterion(
        const int64_t *a, const unsigned int asize,
        const int64_t *b, const unsigned int bsize,
        const unsigned int thr,
    ) except +
    unsigned int edit_distance_dp(
        const int64_t *str1, const size_t size1,
        const int64_t *str2, const size_t size2,
    ) except +


cdef int _encode_str(str text, vector[int64_t]& out) except -1:
    """Append the code points of ``text`` to ``out``."""
    cdef Py_UCS4 character
    out.reserve(<size_t>len(text))
    for character in text:
        out.push_back(<int64_t>character)
    return 0


cdef int _encode_items(object items, dict codes, vector[int64_t]& out) except -1:
    """Append one exact code per element of ``items`` to ``out``.

    ``codes`` maps each distinct element (identity, then ``==``) to a dense
    integer; the same dict must be used for both sides of a comparison.
    """
    cdef object code
    for item in items:
        code = codes.get(item)
        if code is None:
            code = len(codes)
            codes[item] = code
        out.push_back(<int64_t>code)
    return 0


cdef int _encode_pair(
    object a,
    object b,
    vector[int64_t]& left,
    vector[int64_t]& right,
) except -1:
    """Encode ``a`` and ``b`` into exact, mutually consistent integer codes."""
    cdef dict codes
    if isinstance(a, str) and isinstance(b, str):
        _encode_str(<str>a, left)
        _encode_str(<str>b, right)
    else:
        codes = {}
        _encode_items(a, codes, left)
        _encode_items(b, codes, right)
    if left.size() > UINT_MAX or right.size() > UINT_MAX:
        raise OverflowError(
            "sequences longer than 4294967295 elements are not supported"
        )
    return 0


cpdef unsigned int eval(object a, object b) except 0xffffffffffffffff:
    """
    Return the Levenshtein distance between two sequences.

    Parameters
    ----------
    a, b : sequence
        Strings, or sequences of hashable elements.

    Returns
    -------
    int
        Minimum number of insertions, deletions and substitutions.

    Raises
    ------
    TypeError
        If an element is unhashable.

    Examples
    --------
    >>> eval("kitten", "sitting")
    3
    >>> eval([-1], [-2])  # equal hashes, unequal elements
    1
    """
    cdef vector[int64_t] left
    cdef vector[int64_t] right
    _encode_pair(a, b, left, right)
    return edit_distance(
        left.data(), <unsigned int>left.size(),
        right.data(), <unsigned int>right.size(),
    )


cpdef bint eval_criterion(object a, object b, const unsigned int thr) except 0xffffffffffffffff:
    """
    Return whether the Levenshtein distance is at most ``thr``.

    Parameters
    ----------
    a, b : sequence
        Strings, or sequences of hashable elements.
    thr : int
        Non-negative threshold.

    Returns
    -------
    bool
        ``eval(a, b) <= thr``, computed with an early exit.

    Raises
    ------
    TypeError
        If an element is unhashable.
    OverflowError
        If ``thr`` is negative.

    Examples
    --------
    >>> eval_criterion("a", "a", 0)
    True
    >>> eval_criterion("abc", "abd", 0)
    False
    """
    cdef vector[int64_t] left
    cdef vector[int64_t] right
    _encode_pair(a, b, left, right)
    return edit_distance_criterion(
        left.data(), <unsigned int>left.size(),
        right.data(), <unsigned int>right.size(),
        thr,
    )


cpdef unsigned int eval_dp(object a, object b) except 0xffffffffffffffff:
    """
    Return the Levenshtein distance with the plain dynamic-programming kernel.

    Parameters
    ----------
    a, b : sequence
        Strings, or sequences of hashable elements.

    Returns
    -------
    int
        The same value as :func:`eval`; this kernel is the reference the
        bit-parallel one is tested against.

    Raises
    ------
    TypeError
        If an element is unhashable.
    """
    cdef vector[int64_t] left
    cdef vector[int64_t] right
    _encode_pair(a, b, left, right)
    return edit_distance_dp(left.data(), left.size(), right.data(), right.size())
