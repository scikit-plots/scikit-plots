# scikitplot/cexternals/_annoy/annoylib.pyx
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# distutils: language=c++
"""
Cython Implementation of Annoy Index Wrapper
=============================================

This module provides a Python interface to the Annoy C++ library with:
* Secure index initialization and lifecycle management
* Strict type checking and validation
* Proper error propagation from C++ to Python
* Memory-safe resource handling

Design Principles
-----------------
1. **Explicit Initialization**: All structural parameters (f, metric) must be
   set before index construction
2. **Lazy Construction**: C++ index created only when both f and metric are known
3. **State Consistency**: Index state transitions are deterministic and safe
4. **Error Handling**: C++ exceptions caught and converted to Python exceptions
5. **Memory Safety**: All C++ pointers properly managed with RAII patterns

Initialization Workflow
-----------------------
1. Python object allocation (__cinit__)
2. Parameter validation and storage (__init__)
3. C++ index construction (ensure_index)
4. Configuration application (seed, verbose, on_disk_path)
5. Ready for operations (add_item, build, etc.)

State Machine
-------------
States:
* UNINITIALIZED: ptr == NULL, f == 0, metric == UNKNOWN
* CONFIGURED:    ptr == NULL, f > 0,  metric != UNKNOWN
* CONSTRUCTED:   ptr != NULL (C++ index exists)
* BUILDING:      items added but not built
* BUILT:         build() completed, read-only
* LOADED:        loaded from disk

Transitions:
* UNINITIALIZED → CONFIGURED: set f and metric
* CONFIGURED → CONSTRUCTED: ensure_index()
* CONSTRUCTED → BUILDING: add_item()
* BUILDING → BUILT: build()
* BUILT → BUILDING: unbuild()
* Any → UNINITIALIZED: reset or dealloc

Architecture inspired by sklearn with proper separation of concerns:

* _HTMLDocumentationLinkMixin - Doc link generation
* ReprHTMLMixin - HTML representation
* BaseIndex - Core functionality
* Index - User-facing class

Supports multiple float types:
* float16 (half precision)
* float32 (single precision) - default
* float64 (double precision)
* float80 (extended precision, x87)
* float128 (quadruple precision)
"""

# ==============================================================================
# Cython Imports
# ==============================================================================

from libc.stdlib cimport malloc, free  # no-cython-lint
from libc.string cimport memcpy  # no-cython-lint
from libc.stdint cimport int8_t, int16_t, int32_t, int64_t  # no-cython-lint
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t  # no-cython-lint

from libcpp cimport bool as cpp_bool
from libcpp.string cimport string as cpp_string
from libcpp.vector cimport vector as cpp_vector

from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF, Py_XINCREF, Py_XDECREF  # no-cython-lint
from cpython.exc cimport PyErr_SetString, PyErr_Format, PyErr_NoMemory, PyErr_Occurred  # no-cython-lint

# Python imports
import warnings  # no-cython-lint
import json  # no-cython-lint
import pickle  # no-cython-lint
from typing import Any, Dict, Optional  # no-cython-lint
from typing import Optional, Union, List, Tuple, Sequence  # no-cython-lint
from threading import RLock
from typing_extensions import Self  # no-cython-lint

# Import declarations from .pxd file
from .annoylib cimport (
    # Random
    Kiss64Random,
    # Base interface
    AnnoyIndexInterface,
    # Metric types
    Angular,
    Euclidean,
    Manhattan,
    DotProduct,
    Hamming,  # no-cython-lint
    # Template class
    AnnoyIndex,
    HammingWrapper,
    # Build policy
    AnnoyIndexThreadedBuildPolicy,
    # dtype
    # float16_t,
)

__all__ = [
    "Index",
]

# ==============================================================================
# Float Type Support
# ==============================================================================

# Enum for float types
cpdef enum FloatType:
    """
    Supported floating-point types.

    Values
    ------
    FLOAT16 : 0
        Half precision (16-bit)
    FLOAT32 : 1
        Single precision (32-bit) - DEFAULT
    FLOAT64 : 2
        Double precision (64-bit)
    FLOAT80 : 3
        Extended precision (80-bit, x87)
    FLOAT128 : 4
        Quadruple precision (128-bit)
    """
    FLOAT16 = 0
    FLOAT32 = 1
    FLOAT64 = 2
    FLOAT80 = 3
    FLOAT128 = 4


cdef FloatType parse_dtype(str dtype_str):
    """
    Parse dtype string to FloatType enum.

    Parameters
    ----------
    dtype_str : str
        Data type string

    Returns
    -------
    FloatType
        Parsed type

    Raises
    ------
    ValueError
        If dtype not supported
    """
    dtype_lower = dtype_str.lower()

    if dtype_lower in ("float16", "half", "fp16"):
        return FLOAT16
    elif dtype_lower in ("float32", "single", "fp32", "float"):
        return FLOAT32
    elif dtype_lower in ("float64", "double", "fp64"):
        return FLOAT64
    elif dtype_lower in ("float80", "extended", "fp80"):
        return FLOAT80
    elif dtype_lower in ("float128", "quadruple", "fp128", "quad"):
        return FLOAT128
    else:
        raise ValueError(
            f"Unsupported dtype '{dtype_str}'. "
            f"Supported: float16, float32, float64, float80, float128"
        )


cdef str dtype_to_string(FloatType dtype):
    """Convert FloatType to canonical string."""
    if dtype == FLOAT16:
        return "float16"
    elif dtype == FLOAT32:
        return "float32"
    elif dtype == FLOAT64:
        return "float64"
    elif dtype == FLOAT80:
        return "float80"
    elif dtype == FLOAT128:
        return "float128"
    else:
        return "unknown"


cdef int dtype_size(FloatType dtype):
    """Return byte size for dtype."""
    if dtype == FLOAT16:
        return 2
    elif dtype == FLOAT32:
        return 4
    elif dtype == FLOAT64:
        return 8
    elif dtype == FLOAT80:
        return 10  # or 16 padded
    elif dtype == FLOAT128:
        return 16
    else:
        return 0

# ==============================================================================
# Metric Enumeration (Python-Side Type Safety)
# ==============================================================================

cdef enum MetricId:
    """
    Internal metric identifier enum.

    Values
    ------
    METRIC_UNKNOWN : 0
        Metric not yet configured (lazy mode)
    METRIC_ANGULAR : 1
        Cosine-like distance: 1 - cos(theta)
    METRIC_EUCLIDEAN : 2
        L2 distance: ||u - v||_2
    METRIC_MANHATTAN : 3
        L1 distance: ||u - v||_1
    METRIC_DOT : 4
        Negative dot product: -(u·v)
    METRIC_HAMMING : 5
        Bitwise Hamming distance

    Notes
    -----
    * METRIC_UNKNOWN is a sentinel for uninitialized state
    * Each ID maps to exactly one canonical metric name
    * Conversion functions ensure consistency
    """
    METRIC_UNKNOWN = 0
    METRIC_ANGULAR = 1
    METRIC_EUCLIDEAN = 2
    METRIC_MANHATTAN = 3
    METRIC_DOT = 4
    METRIC_HAMMING = 5


# ==============================================================================
# Metric Conversion Functions
# ==============================================================================

cdef const char* metric_to_cstr(MetricId metric_id) nogil:
    """
    Convert MetricId to canonical C string.

    Parameters
    ----------
    metric_id : MetricId
        Internal metric identifier

    Returns
    -------
    name : const char*
        Canonical metric name, or NULL for METRIC_UNKNOWN

    Notes
    -----
    * Thread-safe (nogil)
    * Returns static string literals (no allocation)
    * NULL indicates invalid/unknown metric
    """
    if metric_id == METRIC_ANGULAR:
        return "angular"
    elif metric_id == METRIC_EUCLIDEAN:
        return "euclidean"
    elif metric_id == METRIC_MANHATTAN:
        return "manhattan"
    elif metric_id == METRIC_DOT:
        return "dot"
    elif metric_id == METRIC_HAMMING:
        return "hamming"
    else:
        return NULL


cdef MetricId metric_from_string(const char* metric_str) except? METRIC_UNKNOWN:
    """
    Parse metric string to MetricId with alias support.

    Parameters
    ----------
    metric_str : const char*
        User-provided metric string (case-insensitive)

    Returns
    -------
    metric_id : MetricId
        Parsed metric identifier, or METRIC_UNKNOWN if unrecognized

    Supported Aliases
    -----------------
    * Angular: "angular", "cosine"
    * Euclidean: "euclidean", "l2", "lstsq"
    * Manhattan: "manhattan", "l1", "cityblock", "taxicab"
    * Dot: "dot", "@", ".", "dotproduct", "inner", "innerproduct"
    * Hamming: "hamming"

    Notes
    -----
    * Case-insensitive matching
    * Leading/trailing whitespace is trimmed
    * Returns METRIC_UNKNOWN for invalid input
    """
    if metric_str == NULL:
        return METRIC_UNKNOWN

    # Convert to Python string for easier manipulation
    cdef str metric_py = metric_str.decode("utf-8", "replace").strip().lower()

    # Canonical names
    if metric_py == "angular" or metric_py == "cosine":
        return METRIC_ANGULAR
    elif metric_py == "euclidean" or metric_py == "l2" or metric_py == "lstsq":
        return METRIC_EUCLIDEAN
    elif metric_py == "manhattan" or metric_py == "l1" or metric_py == "cityblock" or metric_py == "taxicab":
        return METRIC_MANHATTAN
    elif metric_py == "dot" or metric_py == "@" or metric_py == "." or metric_py == "dotproduct" or metric_py == "inner" or metric_py == "innerproduct":
        return METRIC_DOT
    elif metric_py == "hamming":
        return METRIC_HAMMING
    else:
        return METRIC_UNKNOWN


# ==============================================================================
# Seed Normalization
# ==============================================================================

cdef inline uint64_t annoy_default_seed() nogil:
    """
    Get Annoy's deterministic default seed.

    Returns
    -------
    seed : uint64_t
        Default seed value from Kiss64Random

    Notes
    -----
    * This is a static constant defined in Kiss64Random
    * Value is 1234567890987654321ULL
    """
    return Kiss64Random.get_default_seed()


cdef inline uint64_t normalize_seed(uint64_t seed) nogil:
    """
    Normalize user-provided seed to valid non-zero value.

    Parameters
    ----------
    seed : uint64_t
        User-provided seed (may be 0)

    Returns
    -------
    normalized : uint64_t
        Valid seed (>= 1)

    Notes
    -----
    * Seed 0 is mapped to default_seed
    * All other seeds are passed through unchanged
    * This prevents degenerate RNG states
    """
    return Kiss64Random.normalize_seed(seed)


# ==============================================================================
# Error Handling Wrapper
# ==============================================================================

cdef class ScopedError:
    """
    RAII-style error message holder for C++ char** error pattern.

    Attributes
    ----------
    err : char*
        Pointer to error string (allocated by C++ code)

    Lifecycle
    ---------
    1. C++ code allocates error message with strdup() on failure
    2. ScopedError stores the pointer
    3. On destruction, ScopedError frees the memory

    Usage
    -----
    cdef ScopedError error = ScopedError()
    if not index.load(filename, False, &error.err):
        if error.err:
            raise IOError(error.err.decode('utf-8', 'replace'))
        else:
            raise IOError("Unknown error")

    Notes
    -----
    * Automatically frees error message in __dealloc__
    * Safe to use even if err remains NULL (no error occurred)
    * Prevents memory leaks from forgotten free() calls
    """

    cdef char* err

    def __cinit__(self):
        """Initialize error pointer to NULL."""
        self.err = NULL

    def __dealloc__(self):
        """Free error message if allocated."""
        if self.err != NULL:
            free(self.err)
            self.err = NULL


# ==============================================================================
# Global State (Module-Level)
# ==============================================================================

# Global counter for unique repr IDs for repr_html (matches C implementation)
cdef unsigned long long g_annoy_repr_html_seq = 0

# ==============================================================================
# Mixin Classes (sklearn-style)
# ==============================================================================
import itertools

from scikitplot import __version__
from scikitplot.externals._packaging.version import parse as parse_version
from scikitplot import get_config


cdef class ReprHTMLMixin:
    """
    Mixin for consistent HTML representation.

    Based on sklearn.base.ReprHTMLMixin.

    Notes
    -----
    When inheriting, define attribute `_html_repr` which is a
    callable returning the HTML representation.
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_repr_html/base.py#L11

    The mixin handles configuration checking and MIME bundle
    generation for Jupyter notebooks.

    Examples
    --------
    >>> class MyIndex(ReprHTMLMixin):
    ...     def _html_repr(self):
    ...         return '<div>My HTML</div>'
    >>> index = MyIndex()
    >>> # In Jupyter:
    >>> index  # Displays HTML
    """

    @property
    def _repr_html_(self):
        """
        HTML representation (property).

        This property checks configuration and returns the
        appropriate HTML representation method.

        Returns
        -------
        callable or None
            HTML repr method if display='diagram', else raises

        Raises
        ------
        AttributeError
            If display configuration is not 'diagram'
        """
        if get_config()["display"] != "diagram":
            raise AttributeError(
                "_repr_html_ is only defined when the "
                "'display' configuration option is set to "
                "'diagram'"
            )
        return self._repr_html_inner

    def _repr_html_inner(self):
        """
        Inner HTML representation method.

        This is the actual method that generates HTML.
        Separated from property for proper hasattr() behavior.

        Returns
        -------
        html : str
            HTML representation
        """
        return self._html_repr()

    def _repr_mimebundle_(self, **kwargs):
        """
        MIME bundle for Jupyter kernels.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments (unused)

        Returns
        -------
        bundle : dict
            MIME bundle with text/plain and text/html
        """
        output = {"text/plain": repr(self)}

        # Add HTML representation
        try:
            if get_config()["display"] == "diagram":
                output["text/html"] = self._html_repr()
        except Exception:
            # Fall back to text if HTML generation fails
            pass

        return output

    def _html_repr(self):
        """
        Generate HTML representation.

        This method should be overridden by subclasses.

        Returns
        -------
        html : str
            HTML representation
        """
        # raise NotImplementedError(
        #     "Subclasses must implement _html_repr()"
        # )
        pass


cdef class _HTMLDocumentationLinkMixin(ReprHTMLMixin):
    """
    Mixin class for generating API documentation links.

    Based on sklearn.base._HTMLDocumentationLinkMixin.

    Attributes
    ----------
    _doc_link_module : str
        Root module (default: 'scikitplot')
    _doc_link_template : str
        URL template
    _doc_link_url_param_generator : callable or None
        Custom URL parameter generator

    Notes
    -----
    This mixin provides :meth:`_get_doc_link` which generates links
    to the API documentation for sklearn estimator HTML diagrams.
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_repr_html/base.py#L11

    Examples
    --------
    >>> class MyIndex(_HTMLDocumentationLinkMixin):
    ...     _doc_link_module = "mymodule"
    >>> index = MyIndex()
    >>> index._get_doc_link()
    'https://...'
    """

    cdef str _doc_link_module
    cdef object _doc_link_url_param_generator

    def __init__(self):
        """Initialize mixin attributes."""
        super(_HTMLDocumentationLinkMixin, self).__init__()

        self._doc_link_module = "scikitplot"
        self._doc_link_url_param_generator = None

    @property
    def _doc_link_template(self):
        """
        Documentation link template.

        Automatically determines version (dev vs release) and
        constructs appropriate URL template.

        Returns
        -------
        template : str
            URL template with placeholders
        """
        scikitplot_version = parse_version(__version__)
        if scikitplot_version.dev is None:
            version_url = f"{scikitplot_version.major}.{scikitplot_version.minor}"
        else:
            version_url = "dev"
        return getattr(
            self,
            "__doc_link_template",
            (
                f"https://scikit-plots.github.io/{version_url}/modules/generated/"
                "{estimator_module}.{estimator_name}.html"
            ),
        )

    @_doc_link_template.setter
    def _doc_link_template(self, value):
        setattr(self, "__doc_link_template", value)

    def _get_doc_link(self):
        """
        Generate link to API documentation.

        Returns
        -------
        url : str
            Documentation URL, or empty string if not applicable

        Examples
        --------
        >>> index = Index(f=128, metric='angular')
        >>> index._get_doc_link()
        'https://scikit-plots.github.io/dev/modules/generated/...'
        """
        # Check if module matches
        module_parts = self.__class__.__module__.split(".")
        if module_parts[0] != self._doc_link_module:
            return ""

        # Custom generator if provided
        if self._doc_link_url_param_generator is not None:
            return self._doc_link_template.format(
                **self._doc_link_url_param_generator()
            )

        # Default: use class name and module
        estimator_name = self.__class__.__name__

        # Construct the estimator's module name, up to the first private submodule.
        # This works because in scikit-learn all public estimators are exposed at
        # that level, even if they actually live in a private sub-module.
        # estimator_module = ".".join(
        #     part for part in module_parts
        #     if not part.startswith("_")
        # )
        estimator_module = ".".join(
            itertools.takewhile(
                lambda part: not part.startswith("_"),
                self.__class__.__module__.split("."),
            )
        )

        return self._doc_link_template.format(
            estimator_module=estimator_module,
            estimator_name=estimator_name
        )


# ==============================================================================
# Base Index Class
# ==============================================================================

# (MRO) Only one extension type base class allowed.
cdef class BaseIndex(_HTMLDocumentationLinkMixin):
    """
    Base class for all Index classes.

    Provides core infrastructure:
    * Parameter management (get_params, set_params)
    * HTML representation (via mixins)
    * Documentation links (via mixins)
    * Common utilities

    Notes
    -----
    This class should not be instantiated directly.
    Use Index or specialized subclasses.

    Examples
    --------
    >>> # Don't do this:
    >>> # base = BaseIndex()
    >>>
    >>> # Instead:
    >>> index = Index(f=128, metric='angular')
    """

    # Core parameters (common to all indices)
    cdef int f
    cdef str metric
    cdef int n_neighbors
    cdef int schema_version

    cdef str dtype
    cdef str index_dtype
    cdef str wrapper_dtype
    cdef str random_dtype

    def __init__(self):
        """
        C-level initialization.

        Sets up core attributes before Python __init__.
        """
        # Initialize mixins
        super(BaseIndex, self).__init__()
        # _HTMLDocumentationLinkMixin.__init__(self)

        # Core parameters
        self.f = 0
        self.metric = None
        self.n_neighbors = 5
        self.schema_version = 0

        self.dtype = "float"
        self.index_dtype = "int32"
        self.wrapper_dtype = "uint64"
        self.random_dtype = "uint64"

    def get_params(self, bint deep=True):
        """
        Get parameters (sklearn-style).

        Parameters
        ----------
        deep : bool, default=True
            If True, include nested params (for future use)

        Returns
        -------
        params : dict
            Parameter dictionary

        Examples
        --------
        >>> index = Index(f=128, metric='angular')
        >>> params = index.get_params()
        >>> print(params['f'])
        128
        """
        return {
            "f": self.f,
            "metric": self.metric,
            "n_neighbors": self.n_neighbors,
            "dtype": dtype_to_string(self.dtype),
            "index_dtype": self.index_dtype,
            "wrapper_dtype": self.wrapper_dtype,
            "random_dtype": self.random_dtype,
            "schema_version": self.schema_version,
        }

    def set_params(self, **params):
        """
        Set parameters (sklearn-style).

        Parameters
        ----------
        **params : dict
            Parameters to set

        Returns
        -------
        self : BaseIndex
            Returns self for method chaining

        Raises
        ------
        ValueError
            If parameter is invalid or immutable after construction

        Examples
        --------
        >>> index = Index(f=128, metric='angular')
        >>> index.set_params(n_neighbors=20)
        >>> print(index.get_params()['n_neighbors'])
        20
        """
        for key, value in params.items():
            if hasattr(self, key):
                # Note: Subclasses should override for construction checks
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")

        return self

    def _html_repr(self):
        """
        Generate HTML representation.

        Subclasses should override to provide custom HTML.

        Returns
        -------
        html : str
            HTML representation
        """
        # Default implementation
        return f"<div><strong>{self.__class__.__name__}</strong></div>"


# ==============================================================================
# Concrete Main Annoy Index Class
# ==============================================================================

cdef class Index(BaseIndex):
    """
    Annoy Approximate Nearest Neighbors Index.

    This is a Cython-powered Python wrapper around the Annoy C++ library.

    Parameters
    ----------
    f : int or None, default=None
        Embedding dimension. If 0 or None, dimension is inferred from first
        vector added. Must be positive for immediate index construction.
    metric : str or None, default=None
        Distance metric. Supported values:
        * "angular", "cosine" → cosine-like distance
        * "euclidean", "l2", "lstsq" → L2 distance
        * "manhattan", "l1", "cityblock", "taxicab" → L1 distance
        * "dot", "@", ".", "dotproduct", "inner", "innerproduct" → negative dot product
        * "hamming" → bitwise Hamming distance
        If None and f > 0, defaults to "angular" with FutureWarning.
    n_neighbors : int, default=5
        Default number of neighbors for queries (estimator parameter).
    on_disk_path : str or None, default=None
        Path for on-disk building. If provided, enables memory-efficient
        building for large indices.
    prefault : bool, default=False
        Whether to prefault pages when loading (may improve query latency).
    seed : int or None, default=None
        Random seed for tree construction. If None, uses Annoy's default.
        Value 0 is treated as "use default" and emits a UserWarning.
    verbose : int or None, default=None
        Verbosity level (clamped to [-2, 2]). Level >= 1 enables logging.
    schema_version : int, default=0
        Pickle schema version marker (does not affect on-disk format).
    dtype : str, default='float32'
        Data type: float16, float32, float64, float80, float128
    index_dtype : str, default='int32'
        Index type: int32, int64
    wrapper_dtype : str, default='uint64'
        Wrapper type (for Hamming): uint32, uint64
    random_dtype : str, default='uint64'
        Random seed type
    **kwargs
        Future extensibility

    Attributes
    ----------
    f : int
        Embedding dimension (0 means "unset / lazy").
    metric : str or None
        Canonical metric name, or None if not configured.
    ptr : AnnoyIndexInterface*
        Pointer to C++ index (NULL if not constructed).

    # State Indicators (Internal)
    _f_valid : bool
        True if f has been set (> 0)
    _metric_valid : bool
        True if metric has been configured
    _index_constructed : bool
        True if C++ index exists (ptr != NULL)

    Examples
    --------
    >>> index = Index(f=128, metric='angular', seed=42)
    >>> index.add_item(0, [0.1] * 128)
    >>> index.add_item(1, [0.2] * 128)
    >>> index.build(n_trees=10)
    >>> neighbors, distances = index.get_nns_by_item(0, n=5, include_distances=True)

    set dtype:

    >>> # Standard usage (float32)
    >>> index = Index(f=128, metric='angular', dtype='float32')
    >>>
    >>> # High precision (float64)
    >>> index = Index(f=128, metric='euclidean', dtype='float64')
    >>>
    >>> # Half precision (float16) - future
    >>> # index = Index(f=128, metric='angular', dtype='float16')
    """

    # =========================================================================
    # C-Level Attributes (Memory Layout)
    # =========================================================================

    # C++ pointer (to be added by concrete implementation)
    # cdef AnnoyIndexInterface[int64_t, double, uint64_t]* ptr_64_64
    cdef AnnoyIndexInterface[int32_t, float, uint64_t]* ptr
    # cdef AnnoyIndexInterface[int32_t, float16_t, uint64_t]* ptr

    # Core index state
    cdef int f  # dimension (0 = unset)
    cdef MetricId metric_id  # METRIC_UNKNOWN = unset

    # Estimator parameters (SLEP013-like)
    cdef size_t n_neighbors  # default neighbors for queries

    # On-disk configuration
    cdef cpp_bool on_disk_active  # True if currently backed by disk
    cdef cpp_string on_disk_path  # path for on_disk_build
    cdef cpp_bool on_disk_pending  # True if on-disk should activate on ensure_index

    # Persistence configuration
    cdef cpp_bool prefault  # prefault pages when loading
    cdef int schema_version  # pickle schema marker

    # Pending runtime configuration (applied on ensure_index)
    cdef uint64_t pending_seed  # seed to apply
    cdef int pending_verbose  # verbosity level
    cdef cpp_bool has_pending_seed  # (bint) True if user set seed
    cdef cpp_bool has_pending_verbose  # (bint) True if user set verbose

    # Python object references (for GC)
    cdef object feature_names_in_  # Optional[Sequence[str]]
    cdef object y  # Optional[Sequence]
    cdef object y_map  # Optional[dict]

    # Extended parameters (for future features)
    cdef str dtype  # Data type: float32, float64, float16
    cdef str index_dtype  # Index type: int32, int64
    cdef str wrapper_dtype  # Wrapper type: uint32, uint64 (for Hamming)
    cdef str random_dtype  # Random seed type: uint32, uint64

    # Threading support
    cdef object lock  # Initialization under __init__ python RLock for thread safety

    # =========================================================================
    # Lifecycle: Allocation
    # =========================================================================

    def __cinit__(self):
        """
        Allocate C-level attributes (called automatically by Python).

        This method is guaranteed to be called exactly once, before __init__.
        It must not raise exceptions or call Python code.

        Notes
        -----
        * Initializes all C-level attributes to safe defaults
        * Sets pointers to NULL
        * Does NOT allocate the C++ index (that happens in ensure_index)
        """
        # Core state: uninitialized
        self.ptr = NULL
        self.f = 0
        self.metric_id = METRIC_UNKNOWN

        # Estimator parameters
        self.n_neighbors = 5

        # On-disk state
        self.on_disk_active = False
        # C++ string constructor called automatically by Cython
        # self.on_disk_path is empty by default
        self.on_disk_pending = False

        # Persistence config
        self.prefault = False
        self.schema_version = 0

        # Pending configuration
        self.pending_seed = 0
        self.pending_verbose = 0
        self.has_pending_seed = False
        self.has_pending_verbose = False

        # Python objects: NULL until set
        self.feature_names_in_ = None
        self.y = None
        self.y_map = None

        # Extended type parameters (for future use)
        self.dtype = "float32"
        self.index_dtype = "int32"
        self.wrapper_dtype = "uint64"
        self.random_dtype = "uint64"

    # =========================================================================
    # Lifecycle: Initialization
    # =========================================================================

    def __init__(
        self,
        f: Optional[int] = None,
        metric: Optional[str] = None,
        *,
        n_neighbors: int = 5,
        on_disk_path: Optional[str] = None,
        prefault: bool = False,
        seed: Optional[int] = None,
        verbose: Optional[int] = None,
        schema_version: int = 0,
        # Extended parameters (for future use)
        dtype: str = "float32",
        index_dtype: str = "int32",
        wrapper_dtype: str = "uint64",
        random_dtype: str = "uint64",
        **kwargs  # Future extensibility
    ):
        """
        Initialize Annoy index with validated parameters.

        This method handles parameter validation and storage. The actual C++
        index is constructed lazily when both f and metric are known.

        Parameters
        ----------
        f : int or None
            Embedding dimension (>= 0). If 0 or None, dimension is inferred
            from first vector. Must be positive for eager construction.
        metric : str or None
            Distance metric. See class docstring for supported values.
        n_neighbors : int
            Default neighbors for queries (must be >= 1).
        on_disk_path : str or None
            Path for on-disk building. Enables memory-efficient mode.
        prefault : bool
            Prefault pages when loading (may improve query latency).
        seed : int or None
            Random seed (0 means "use default", emits UserWarning).
        verbose : int or None
            Verbosity level (clamped to [-2, 2]).
        schema_version : int
            Pickle schema marker (does not affect disk format).
        dtype : str, default="float32"
            Data type for embeddings. Currently only "float32" supported.
            Future: "float16", "float64", "int8", "uint8"
        index_dtype : str, default="int32"
            Index identifier type. Currently only "int32" supported.
            Future: "int64", "uint32"
        wrapper_dtype : str, default="uint64"
            Internal wrapper type (e.g., for Hamming packing).
            Future: "uint32", "uint8", "bool"
        random_dtype : str, default="uint64"
            Random seed type. Currently only "uint64" supported.
        **kwargs : dict
            Reserved for future extensions. Currently ignored with warning.

        Raises
        ------
        ValueError
            If parameters are invalid (negative f, unknown metric, etc.).
        TypeError
            If parameters have wrong types.

        Notes
        -----
        * Re-initialization is allowed (resets state)
        * Seed 0 is deterministic but emits warning when explicitly provided
        * Index construction is deferred until ensure_index() is called
        """
        # Initialize mixins
        super(Index, self).__init__()
        # BaseIndex.__init__(self)

        # Allow re-initialization (rare but valid in CPython)
        if self.ptr != NULL:
            self._destroy_index()

        # Clear Python references
        self.feature_names_in_ = None
        self.y = None
        self.y_map = None

        # Reset to clean state
        self.f = 0
        self.metric_id = METRIC_UNKNOWN
        self.n_neighbors = 5
        self.on_disk_active = False
        self.on_disk_path.clear()
        self.on_disk_pending = False
        self.prefault = False
        self.schema_version = 0
        self.pending_seed = 0
        self.pending_verbose = 0
        self.has_pending_seed = False
        self.has_pending_verbose = False

        # Store extended parameters types
        self.dtype = dtype
        self.index_dtype = index_dtype
        self.wrapper_dtype = wrapper_dtype
        self.random_dtype = random_dtype

        # Warn about unused kwargs (future extensibility)
        if kwargs:
            warnings.warn(
                f"Unknown parameters ignored: {list(kwargs.keys())}. "
                f"These may be supported in future versions.",
                FutureWarning,
                stacklevel=2
            )

        # ---------------------------------------------------------------------
        # Validate and apply f (dimension)
        # ---------------------------------------------------------------------
        if f is None:
            self.f = 0  # lazy mode
        else:
            if not isinstance(f, int):
                raise TypeError(f"`f` must be an integer, got {type(f).__name__}")
            if f < 0:
                raise ValueError(f"`f` must be non-negative, got {f}")
            self.f = f

        # ---------------------------------------------------------------------
        # Validate and apply metric
        # ---------------------------------------------------------------------
        cdef MetricId parsed_id
        if metric is not None:
            if not isinstance(metric, str):
                raise TypeError(f"`metric` must be a string, got {type(metric).__name__}")

            metric_cstr = metric.encode("utf-8")
            parsed_id = metric_from_string(metric_cstr)

            if parsed_id == METRIC_UNKNOWN:
                raise ValueError(
                    f"Invalid metric '{metric}'. "
                    f"Expected one of: angular, euclidean, manhattan, dot, hamming "
                    f"(or aliases: cosine, l2, l1, cityblock, dotproduct, etc.)"
                )

            self.metric = metric
            self.metric_id = parsed_id

        # Default metric for legacy compatibility (with warning)
        if self.f > 0 and self.metric_id == METRIC_UNKNOWN:
            warnings.warn(
                "The default metric will be removed in a future version. "
                "Please pass metric='angular' explicitly.",
                FutureWarning,
                stacklevel=2
            )
            self.metric = "angular"
            self.metric_id = METRIC_ANGULAR

        # ---------------------------------------------------------------------
        # Validate and apply n_neighbors
        # ---------------------------------------------------------------------
        if not isinstance(n_neighbors, int):
            raise TypeError(f"`n_neighbors` must be an integer, got {type(n_neighbors).__name__}")
        if n_neighbors < 1:
            raise ValueError(f"`n_neighbors` must be >= 1, got {n_neighbors}")
        self.n_neighbors = n_neighbors

        # ---------------------------------------------------------------------
        # Apply on_disk_path (stored; activation deferred)
        # ---------------------------------------------------------------------
        if on_disk_path is not None:
            if not isinstance(on_disk_path, str):
                raise TypeError(f"`on_disk_path` must be a string, got {type(on_disk_path).__name__}")
            self.on_disk_path = on_disk_path.encode("utf-8")
            self.on_disk_pending = True  # activate on ensure_index

        # ---------------------------------------------------------------------
        # Apply prefault
        # ---------------------------------------------------------------------
        self.prefault = bool(prefault)

        # ---------------------------------------------------------------------
        # Apply schema_version
        # ---------------------------------------------------------------------
        if not isinstance(schema_version, int):
            raise TypeError(f"`schema_version` must be an integer, got {type(schema_version).__name__}")
        self.schema_version = schema_version

        # ---------------------------------------------------------------------
        # Apply seed (stored; applied on ensure_index)
        # ---------------------------------------------------------------------
        cdef uint64_t seed_u64
        if seed is not None:
            if not isinstance(seed, int):
                raise TypeError(f"`seed` must be an integer, got {type(seed).__name__}")
            if seed < 0:
                raise ValueError(f"`seed` must be non-negative, got {seed}")

            # Check for overflow (Python int can be arbitrarily large)
            if seed > <uint64_t>(-1):
                raise ValueError(f"`seed` must fit in uint64_t range [0, 2**64 - 1], got {seed}")

            seed_u64 = <uint64_t>seed

            if seed_u64 == 0:
                warnings.warn(
                    "seed=0 uses Annoy's deterministic default seed",
                    UserWarning,
                    stacklevel=2
                )
                self.pending_seed = 0
                self.has_pending_seed = False  # treat as "no override"
            else:
                self.pending_seed = normalize_seed(seed_u64)
                self.has_pending_seed = True

        # ---------------------------------------------------------------------
        # Apply verbose (stored; applied on ensure_index)
        # ---------------------------------------------------------------------
        cdef int level
        if verbose is not None:
            if not isinstance(verbose, int):
                raise TypeError(f"`verbose` must be an integer, got {type(verbose).__name__}")

            # Clamp to [-2, 2]
            level = verbose
            if level < -2:
                level = -2
            if level > 2:
                level = 2

            self.pending_verbose = level
            self.has_pending_verbose = True

        # ---------------------------------------------------------------------
        # Eagerly construct index if both f and metric are known
        # ---------------------------------------------------------------------
        if self.f > 0 and self.metric_id != METRIC_UNKNOWN:
            self._ensure_index()

        # Threading
        self.lock = RLock()

    # =========================================================================
    # Lifecycle: Destruction
    # =========================================================================

    def __dealloc__(self):
        """
        Deallocate C++ resources (called automatically by Python).

        This method is guaranteed to be called exactly once when the Python
        object is garbage-collected. It must not raise exceptions.

        Notes
        -----
        * Unloads memory-mapped files
        * Deletes C++ index
        * Frees all C++ allocations
        * Safe to call even if __cinit__ partially failed
        """
        self._destroy_index()

    cdef void _destroy_index(self) nogil:
        """
        Internal: safely destroy C++ index (nogil for efficiency).

        Notes
        -----
        * Safe to call multiple times (idempotent)
        * Safe to call from __dealloc__ or __init__
        * Never raises exceptions (nogil context)
        """
        if self.ptr != NULL:
            self.ptr.unload()
            # Delete the C++ object
            del self.ptr
            self.ptr = NULL

        # Reset state flags
        self.on_disk_active = False
        self.on_disk_pending = False

    # =========================================================================
    # Index Construction (Core Logic)
    # =========================================================================

    cdef void _ensure_index(self) except *:
        """
        Ensure C++ index exists, constructing it if necessary.

        This is the central index construction logic. It:
        1. Validates that f and metric are set
        2. Constructs the appropriate C++ index type
        3. Applies pending configuration (seed, verbose, on_disk_path)
        4. Transitions state to CONSTRUCTED

        Raises
        ------
        RuntimeError
            If f or metric is not set, or if construction fails.
        MemoryError
            If C++ allocation fails.
        IOError
            If on_disk_build fails.

        Notes
        -----
        * Thread-safe: can be called multiple times (idempotent)
        * On failure, index remains in UNINITIALIZED/CONFIGURED state
        * Seed and verbose are applied before on_disk_build
        """
        # Already constructed: nothing to do
        if self.ptr != NULL:
            return

        # Validate preconditions
        if self.f <= 0:
            raise RuntimeError("Index dimension `f` is not set (must be > 0)")

        if self.metric_id == METRIC_UNKNOWN:
            raise RuntimeError("Index metric is not set")

        # ---------------------------------------------------------------------
        # Construct the appropriate C++ index type
        # ---------------------------------------------------------------------
        try:
            if self.metric_id == METRIC_ANGULAR:
                self.ptr = <AnnoyIndexInterface[int32_t, float, uint64_t]*>(
                    new AnnoyIndex[int32_t, float, Angular, Kiss64Random, AnnoyIndexThreadedBuildPolicy](self.f)
                )
            elif self.metric_id == METRIC_EUCLIDEAN:
                self.ptr = <AnnoyIndexInterface[int32_t, float, uint64_t]*>(
                    new AnnoyIndex[int32_t, float, Euclidean, Kiss64Random, AnnoyIndexThreadedBuildPolicy](self.f)
                )
            elif self.metric_id == METRIC_MANHATTAN:
                self.ptr = <AnnoyIndexInterface[int32_t, float, uint64_t]*>(
                    new AnnoyIndex[int32_t, float, Manhattan, Kiss64Random, AnnoyIndexThreadedBuildPolicy](self.f)
                )
            elif self.metric_id == METRIC_DOT:
                self.ptr = <AnnoyIndexInterface[int32_t, float, uint64_t]*>(
                    new AnnoyIndex[int32_t, float, DotProduct, Kiss64Random, AnnoyIndexThreadedBuildPolicy](self.f)
                )
            elif self.metric_id == METRIC_HAMMING:
                self.ptr = <AnnoyIndexInterface[int32_t, float, uint64_t]*>(
                    new HammingWrapper[int32_t, float, uint64_t, Kiss64Random, AnnoyIndexThreadedBuildPolicy](self.f)
                )
            else:
                raise RuntimeError(f"Internal error: unknown metric_id {self.metric_id}")

        except MemoryError:
            raise MemoryError("Failed to allocate Annoy index (out of memory)")
        except Exception as e:
            raise RuntimeError(f"Failed to create Annoy index: {e}")

        # ---------------------------------------------------------------------
        # Apply pending configuration
        # ---------------------------------------------------------------------

        # Apply seed (if explicitly set)
        if self.has_pending_seed:
            self.ptr.set_seed(self.pending_seed)

        # Apply verbose (if explicitly set)
        if self.has_pending_verbose:
            self.ptr.set_verbose(self.pending_verbose >= 1)

        # Apply on_disk_build (if path is configured and pending)
        cdef ScopedError error
        cdef cpp_bool success
        if self.on_disk_pending and not self.on_disk_active:
            if self.on_disk_path.empty():
                # Sanity check: should never happen
                self._destroy_index()
                raise RuntimeError(
                    "Internal error: on_disk_pending is True but on_disk_path is empty"
                )

            # Activate on-disk build mode
            error = ScopedError()
            success = self.ptr.on_disk_build(
                self.on_disk_path.c_str(),
                &error.err
            )

            if not success:
                # Roll back to safe state
                self._destroy_index()
                if error.err != NULL:
                    raise IOError(error.err.decode("utf-8", "replace"))
                else:
                    raise IOError("on_disk_build failed (unknown error)")

            self.on_disk_active = True
            self.on_disk_pending = False

    # =========================================================================
    # Properties: Core Index State
    # =========================================================================

    @property
    def f(self) -> int:
        """
        Embedding dimension.

        Returns
        -------
        f : int
            Number of dimensions (0 means "unset / lazy").

        Notes
        -----
        * Immutable after index construction
        * Setting to 0 after construction raises ValueError
        """
        return self.f

    @f.setter
    def f(self, value: int):
        """Set embedding dimension (only allowed before index construction)."""
        if self.ptr != NULL:
            raise ValueError(
                "Cannot modify `f` after index construction. "
                "Create a new index if you need a different dimension."
            )

        if not isinstance(value, int):
            raise TypeError(f"`f` must be an integer, got {type(value).__name__}")

        if value < 0:
            raise ValueError(f"`f` must be non-negative, got {value}")

        self.f = value

    @property
    def metric(self) -> Optional[str]:
        """
        Distance metric name.

        Returns
        -------
        metric : str or None
            Canonical metric name, or None if not configured.

        Possible Values
        ---------------
        * "angular" : cosine-like distance
        * "euclidean" : L2 distance
        * "manhattan" : L1 distance
        * "dot" : negative dot product
        * "hamming" : bitwise Hamming distance
        * None : not yet configured (lazy mode)

        Notes
        -----
        * Immutable after index construction
        * Returns canonical name even if alias was used in constructor
        """
        cdef const char* name_cstr = metric_to_cstr(self.metric_id)
        if name_cstr == NULL:
            return None
        return name_cstr.decode("utf-8")

    @metric.setter
    def metric(self, value: Optional[str]):
        """Set distance metric (only allowed before index construction)."""
        if self.ptr != NULL:
            raise ValueError(
                "Cannot modify `metric` after index construction. "
                "Create a new index if you need a different metric."
            )

        if value is None:
            self.metric_id = METRIC_UNKNOWN
            return

        if not isinstance(value, str):
            raise TypeError(f"`metric` must be a string, got {type(value).__name__}")

        cdef bytes metric_bytes = value.encode("utf-8")
        cdef MetricId parsed_id = metric_from_string(metric_bytes)

        if parsed_id == METRIC_UNKNOWN:
            raise ValueError(
                f"Invalid metric '{value}'. "
                f"Expected one of: angular, euclidean, manhattan, dot, hamming"
            )

        self.metric_id = parsed_id

    @property
    def n_neighbors(self) -> int:
        """Default number of neighbors for queries."""
        return self.n_neighbors

    @n_neighbors.setter
    def n_neighbors(self, value: int):
        """Set default number of neighbors."""
        if not isinstance(value, int):
            raise TypeError(f"`n_neighbors` must be an integer, got {type(value).__name__}")
        if value < 1:
            raise ValueError(f"`n_neighbors` must be >= 1, got {value}")
        self.n_neighbors = value

    # =========================================================================
    # Core Annoy Methods
    # =========================================================================

    def add_item(self, int item, vector) -> None:
        """
        Add a vector to the index.

        Parameters
        ----------
        item : int
            Non-negative item identifier
        vector : sequence
            Embedding vector of length f

        Raises
        ------
        RuntimeError
            If index is not constructed or already built
        ValueError
            If vector dimension doesn't match f
        IndexError
            If item is negative

        Notes
        -----
        * Must be called before build()
        * Item IDs need not be contiguous
        * After build(), call unbuild() to add more items
        """
        if self.ptr == NULL:
            self._ensure_index()

        if item < 0:
            raise IndexError("item id cannot be negative")

        # Convert Python sequence to C array
        cdef cpp_vector[float] vec
        vec.resize(self.f)

        cdef int i
        for i in range(self.f):
            vec[i] = float(vector[i])

        # Call C++ add_item
        cdef char* error = NULL
        cdef cpp_bool success

        # CRITICAL: Release GIL during expensive C++ operation
        # This allows other Python threads to run concurrently
        with nogil:
            success = self.ptr.add_item(item, vec.data(), &error)
        # GIL automatically reacquired here

        if not success:
            if error != NULL:
                err_msg = error.decode("utf-8", "replace")
                free(error)
                raise RuntimeError(f"add_item failed: {err_msg}")
            else:
                raise RuntimeError("add_item failed (unknown error)")

    def build(self, int n_trees=-1, int n_jobs=-1) -> None:
        """
        Build the search forest (thread-safe, releases GIL).

        Parameters
        ----------
        n_trees : int, default=-1
            Number of trees to build. If -1, auto-selects based on dimension.
            More trees = better accuracy but slower queries and more memory.
        n_jobs : int, default=-1
            Number of threads. If -1, uses all available cores.

        Raises
        ------
        RuntimeError
            If index is not constructed or no items added

        Notes
        -----
        * Index becomes read-only after build()
        * Auto n_trees formula: max(10, 2*f)
        * Call unbuild() to add more items
        * Releases GIL during C++ build operation
        * Allows concurrent Python threads to run
        * The C++ build itself is multi-threaded (n_jobs)

        Examples
        --------
        >>> # Multiple threads can build independently:
        >>> from concurrent.futures import ThreadPoolExecutor
        >>> def worker(index, i):
        ...     index.build(n_trees=10)
        >>> with ThreadPoolExecutor(max_workers=4) as executor:
        ...     futures = [executor.submit(worker, index, i) for i in range(4)]
        """
        if self.ptr == NULL:
            raise RuntimeError("Cannot build: index not constructed")

        cdef char* error = NULL
        cdef cpp_bool success

        # CRITICAL: Release GIL during expensive C++ operation
        # This allows other Python threads to run concurrently
        with nogil:
            success = self.ptr.build(n_trees, n_jobs, &error)
        # GIL automatically reacquired here

        if not success:
            if error != NULL:
                err_msg = error.decode("utf-8", "replace")
                free(error)
                raise RuntimeError(f"build failed: {err_msg}")
            else:
                raise RuntimeError("build failed (unknown error)")

    def unbuild(self) -> None:
        """
        Remove all trees to allow adding more items.

        Transitions index back to BUILDING state.

        Raises
        ------
        RuntimeError
            If index is not built
        """
        if self.ptr == NULL:
            raise RuntimeError("Cannot unbuild: index not constructed")

        cdef char* error = NULL
        cdef cpp_bool success = self.ptr.unbuild(&error)

        if not success:
            if error != NULL:
                err_msg = error.decode("utf-8", "replace")
                free(error)
                raise RuntimeError(f"unbuild failed: {err_msg}")
            else:
                raise RuntimeError("unbuild failed (unknown error)")

    def get_nns_by_item(self, int item, int n, int search_k=-1, bint include_distances=False):
        """
        Find nearest neighbors (thread-safe, releases GIL).

        Parameters
        ----------
        item : int
            Query item ID (must be < n_items)
        n : int
            Number of neighbors to return
        search_k : int, default=-1
            Search effort. If -1, uses n_trees * n.
            Higher values = better accuracy but slower.
        include_distances : bool, default=False
            If True, return (neighbors, distances) tuple

        Returns
        -------
        neighbors : list[int]
            Item IDs of nearest neighbors
        distances : list[float], optional
            Distances to neighbors (only if include_distances=True)

        Raises
        ------
        RuntimeError
            If index not built
        IndexError
            If item >= n_items

        Notes
        -----
        * Releases GIL during query (true parallelism)
        * Multiple threads can query simultaneously
        * Linear speedup with thread count

        Examples
        --------
        >>> # Parallel queries from multiple threads:
        >>> from concurrent.futures import ThreadPoolExecutor
        >>> def query_worker(index, item_id):
        ...     return index.get_nns_by_item(item_id, n=10)
        >>> with ThreadPoolExecutor(max_workers=8) as executor:
        ...     results = list(executor.map(
        ...         lambda i: query_worker(index, i),
        ...         range(1000)
        ...     ))
        >>> # True parallelism - all 8 threads run concurrently!
        """
        if self.ptr == NULL:
            raise RuntimeError("Cannot query: index not constructed")

        if item < 0:
            raise IndexError("item id cannot be negative")

        cdef cpp_vector[int32_t] result
        cdef cpp_vector[float] distances
        # cdef char* error = NULL

        # CRITICAL: Release GIL during query
        # Multiple Python threads can now query concurrently!
        if include_distances:
            with nogil:
                self.ptr.get_nns_by_item(item, n, search_k, &result, &distances)
        else:
            with nogil:
                self.ptr.get_nns_by_item(item, n, search_k, &result, NULL)
        # GIL reacquired

        # Convert results to Python lists (requires GIL for Python object creation)
        py_result = [result[i] for i in range(result.size())]

        if include_distances:
            # cdef Py_ssize_t n = distances.size()
            py_distances = [distances[i] for i in range(distances.size())]
            return (py_result, py_distances)
        else:
            return py_result

    def get_nns_by_vector(self, vector, int n, int search_k=-1, bint include_distances=False):
        """
        Query by vector (thread-safe, releases GIL).

        Parameters
        ----------
        vector : sequence
            Query vector of length f
        n : int
            Number of neighbors to return
        search_k : int, default=-1
            Search effort. If -1, uses n_trees * n.
        include_distances : bool, default=False
            If True, return (neighbors, distances) tuple

        Returns
        -------
        neighbors : list[int]
            Item IDs of nearest neighbors
        distances : list[float], optional
            Distances to neighbors

        Raises
        ------
        RuntimeError
            If index not built
        ValueError
            If vector dimension doesn't match f
        """
        if self.ptr == NULL:
            raise RuntimeError("Cannot query: index not constructed")

        # Convert Python sequence to C array vector (requires GIL)
        cdef cpp_vector[float] vec
        vec.resize(self.f)

        cdef int i
        for i in range(self.f):
            vec[i] = float(vector[i])

        cdef cpp_vector[int32_t] result
        cdef cpp_vector[float] distances
        # cdef char* error = NULL

        # CRITICAL: Release GIL during query
        if include_distances:
            with nogil:
                self.ptr.get_nns_by_vector(vec.data(), n, search_k, &result, &distances)
        else:
            with nogil:
                self.ptr.get_nns_by_vector(vec.data(), n, search_k, &result, NULL)
        # GIL reacquired

        # Convert to Python lists
        py_result = [result[i] for i in range(result.size())]

        if include_distances:
            # cdef Py_ssize_t n = distances.size()
            py_distances = [distances[i] for i in range(distances.size())]
            return (py_result, py_distances)
        else:
            return py_result

    def get_item(self, int item):
        """
        Retrieve a stored embedding vector.

        Parameters
        ----------
        item : int
            Item ID (must be < n_items)

        Returns
        -------
        vector : list[float]
            Embedding vector of length f

        Raises
        ------
        RuntimeError
            If index not constructed
        IndexError
            If item is negative or >= n_items
        """
        if self.ptr == NULL:
            raise RuntimeError("Cannot get_item: index not constructed")

        if item < 0:
            raise IndexError("item id cannot be negative")

        # Allocate output buffer
        cdef cpp_vector[float] vec
        vec.resize(self.f)

        # Call C++ get_item
        self.ptr.get_item(item, vec.data())

        # Convert to Python list
        return [vec[i] for i in range(self.f)]

    def get_distance(self, int i, int j):
        """
        Compute distance between two stored items.

        Parameters
        ----------
        i, j : int
            Item IDs (must be < n_items)

        Returns
        -------
        distance : float
            Distance according to index metric

        Raises
        ------
        RuntimeError
            If index not constructed
        IndexError
            If i or j is negative or >= n_items

        Notes
        -----
        * Does not require built index
        * For Hamming metric, distance is clipped to [0, f]
        """
        if self.ptr == NULL:
            raise RuntimeError("Cannot get_distance: index not constructed")

        if i < 0 or j < 0:
            raise IndexError("item ids cannot be negative")

        return self.ptr.get_distance(i, j)

    def get_n_items(self) -> int:
        """
        Return number of items in the index.

        Returns
        -------
        n_items : int
            Number of items added (may be sparse)
        """
        if self.ptr == NULL:
            return 0
        return self.ptr.get_n_items()

    def get_n_trees(self) -> int:
        """
        Return number of trees in the index.

        Returns
        -------
        n_trees : int
            Number of trees (0 if not built)
        """
        if self.ptr == NULL:
            return 0
        return self.ptr.get_n_trees()

    def save(self, filename, bint prefault=False) -> None:
        """
        Save index to disk file.

        Parameters
        ----------
        filename : str
            Output file path
        prefault : bool, default=False
            Whether to prefault pages during save

        Raises
        ------
        RuntimeError
            If index not built
        IOError
            If file cannot be written
        """
        if self.ptr == NULL:
            raise RuntimeError("Cannot save: index not constructed")

        cdef bytes filename_bytes = filename.encode("utf-8")
        cdef char* error = NULL
        cdef cpp_bool success = self.ptr.save(filename_bytes, prefault, &error)

        if not success:
            if error != NULL:
                err_msg = error.decode("utf-8", "replace")
                free(error)
                raise IOError(f"save failed: {err_msg}")
            else:
                raise IOError("save failed (unknown error)")

    def load(self, filename, bint prefault=False) -> None:
        """
        Load index from disk file.

        Parameters
        ----------
        filename : str
            Input file path
        prefault : bool, default=False
            Whether to prefault pages into memory

        Raises
        ------
        RuntimeError
            If dimensions don't match
        IOError
            If file cannot be read

        Notes
        -----
        * Dimension f and metric must match the saved index
        * prefault=True may improve query latency at cost of load time
        """
        if self.ptr == NULL:
            self._ensure_index()

        cdef bytes filename_bytes = filename.encode("utf-8")
        cdef char* error = NULL
        cdef cpp_bool success = self.ptr.load(filename_bytes, prefault, &error)

        if not success:
            if error != NULL:
                err_msg = error.decode("utf-8", "replace")
                free(error)
                raise IOError(f"load failed: {err_msg}")
            else:
                raise IOError("load failed (unknown error)")

    def unload(self) -> None:
        """
        Unmap memory-mapped files and free memory.

        Transitions index to EMPTY state.
        Safe to call multiple times.
        """
        if self.ptr != NULL:
            self.ptr.unload()

    def set_seed(self, int seed) -> None:
        """
        Set random seed for index construction.

        Parameters
        ----------
        seed : int
            Random seed (0 uses default_seed)

        Notes
        -----
        * Must be called before build()
        * Seed is normalized: 0 -> default_seed
        * Affects tree construction randomness
        """
        if self.ptr == NULL:
            # Store for later application
            self.pending_seed = seed
            self.has_pending_seed = True
        else:
            self.ptr.set_seed(seed)
        return self

    def set_verbose(self, bint v) -> None:
        """
        Enable/disable verbose logging.

        Parameters
        ----------
        v : bool
            True to enable verbose output
        """
        if self.ptr == NULL:
            # Store for later application
            self.pending_verbose = 1 if v else 0
            self.has_pending_verbose = True
        else:
            self.ptr.set_verbose(v)
        return self

    # =========================================================================
    # Context Manager Protocol
    # =========================================================================

    def __enter__(self):
        """
        Context manager entry.

        Acquires lock for thread-safe operations.

        Returns
        -------
        self : Index
            The index instance

        Examples
        --------
        >>> with Index(f=10, metric='angular') as index:
        ...     index.add_item(0, [0.1] * 10)
        ...     index.build()
        """
        self.lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit.

        Releases lock after operations complete.

        Parameters
        ----------
        exc_type : type or None
            Exception type if raised
        exc_val : Exception or None
            Exception instance if raised
        exc_tb : traceback or None
            Traceback if exception raised

        Returns
        -------
        False : bool
            Always returns False (doesn't suppress exceptions)
        """
        self.lock.release()
        return False

    # =========================================================================
    # Enhanced String Representations
    # =========================================================================

    def __repr__(self) -> str:
        """
        Return detailed string representation with memory address.

        Returns
        -------
        repr : str
            Detailed representation with object ID

        Examples
        --------
        >>> index = Index(f=10, metric='angular')
        >>> print(repr(index))
        Index(f=10, metric='angular', ...) at 0x7F8B3C...
        """
        return f"{self} at 0x{id(self):X}"

    def __str__(self) -> str:
        """
        Return user-friendly string representation.

        Returns
        -------
        str : str
            Class name with key parameters

        Examples
        --------
        >>> index = Index(f=10, metric='angular')
        >>> print(str(index))
        Index
        """
        return f"{self.__class__.__name__}"

    # =========================================================================
    # Parameter Management (sklearn-style)
    # =========================================================================

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters (sklearn-style).

        Parameters
        ----------
        deep : bool, default=True
            If True, include nested parameters (reserved for future use)

        Returns
        -------
        params : dict
            Parameter dictionary with all configuration

        Examples
        --------
        >>> index = Index(f=128, metric='angular', seed=42)
        >>> params = index.get_params()
        >>> print(params['f'])
        128
        >>> print(params['metric'])
        'angular'
        """
        params = {
            # Core parameters
            "f": self.f,
            "metric": self.metric,
            "n_neighbors": self.n_neighbors,

            # Configuration
            "seed": self.pending_seed if self.has_pending_seed else None,
            "verbose": self.pending_verbose if self.has_pending_verbose else None,

            # Persistence
            "on_disk_path": self.on_disk_path.decode("utf-8") if not self.on_disk_path.empty() else None,
            "prefault": self.prefault,
            "schema_version": self.schema_version,

            # Extended parameters
            "dtype": self.dtype,
            "index_dtype": self.index_dtype,
            "wrapper_dtype": self.wrapper_dtype,
            "random_dtype": self.random_dtype,
        }

        return params

    def set_params(self, **params) -> Self:
        """
        Set parameters (sklearn-style).

        Parameters
        ----------
        **params : dict
            Parameters to update

        Returns
        -------
        self : Index
            Returns self for method chaining

        Raises
        ------
        ValueError
            If trying to set immutable parameters after construction

        Examples
        --------
        >>> index = Index(f=128, metric='angular')
        >>> index.set_params(n_neighbors=10, seed=42)
        >>> index.build()

        Notes
        -----
        * Cannot modify f or metric after index construction
        * Can always modify n_neighbors, seed, verbose
        """
        # Immutable parameters (only before construction)
        if "f" in params:
            if self.ptr != NULL:
                raise ValueError(
                    "Cannot modify 'f' after index construction. "
                    "Create a new index with desired dimension."
                )
            self.f = params["f"]

        if "metric" in params:
            if self.ptr != NULL:
                raise ValueError(
                    "Cannot modify 'metric' after index construction. "
                    "Create a new index with desired metric."
                )
            self.metric = params["metric"]

        # Always mutable parameters
        if "n_neighbors" in params:
            self.n_neighbors = params["n_neighbors"]

        if "seed" in params:
            seed_val = params["seed"]
            if seed_val is not None:
                self.pending_seed = seed_val
                self.has_pending_seed = True
                if self.ptr != NULL:
                    self.ptr.set_seed(seed_val)

        if "verbose" in params:
            verbose_val = params["verbose"]
            if verbose_val is not None:
                self.pending_verbose = verbose_val
                self.has_pending_verbose = True
                if self.ptr != NULL:
                    self.ptr.set_verbose(verbose_val >= 1)

        # Persistence parameters
        if "prefault" in params:
            self.prefault = params["prefault"]

        if "schema_version" in params:
            self.schema_version = params["schema_version"]

        # Extended parameters
        if "dtype" in params:
            self.dtype = params["dtype"]

        if "index_dtype" in params:
            self.index_dtype = params["index_dtype"]

        if "wrapper_dtype" in params:
            self.wrapper_dtype = params["wrapper_dtype"]

        if "random_dtype" in params:
            self.random_dtype = params["random_dtype"]

        return self

    # =========================================================================
    # Serialization Support (Pickle Protocol)
    # =========================================================================

    def __getstate__(self) -> Dict[str, Any]:
        """
        Return state for pickling.

        Returns
        -------
        state : dict
            Complete state for pickle

        Examples
        --------
        >>> index = Index(f=128, metric='angular', seed=42)
        >>> index.add_item(0, [0.1] * 128)
        >>> index.build()
        >>> import pickle
        >>> pickled = pickle.dumps(index)
        """
        return self.get_state()

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restore state from pickle.

        Parameters
        ----------
        state : dict
            State dictionary from __getstate__

        Examples
        --------
        >>> import pickle
        >>> restored = pickle.loads(pickled)
        """
        self.set_state(state)

    def __reduce__(self):
        """
        Custom pickle protocol.

        Returns
        -------
        tuple
            (constructor, args, state) for pickle
        """
        # Constructor call: Index()
        return (
            self.__class__,
            (),  # No args (use default constructor)
            self.__getstate__(),  # State to restore
        )

    def __reduce_ex__(self, protocol: int):
        """
        Protocol-specific pickle.

        Parameters
        ----------
        protocol : int
            Pickle protocol version

        Returns
        -------
        tuple
            Same as __reduce__()
        """
        return self.__reduce__()

    def get_state(self) -> Dict[str, Any]:
        """
        Get complete state dictionary.

        Returns
        -------
        state : dict
            Complete index state including:
            * Parameters (f, metric, etc.)
            * Index data (if built)
            * Configuration

        Examples
        --------
        >>> index = Index(f=128, metric='angular', seed=42)
        >>> index.add_item(0, [0.1] * 128)
        >>> index.build()
        >>> state = index.get_state()
        >>> print('f' in state)
        True
        >>> print('metric' in state)
        True
        """
        state = {
            # Version for compatibility checking
            "__version__": "1.0",

            # Core parameters
            "params": self.get_params(deep=False),

            # Index state
            "constructed": self.ptr != NULL,
            "n_items": self.ptr.get_n_items() if self.ptr != NULL else 0,
            "n_trees": self.ptr.get_n_trees() if self.ptr != NULL else 0,
        }

        # Serialize index data if built
        cdef char* error = NULL
        cdef cpp_vector[uint8_t] serialized_data
        if self.ptr != NULL and self.ptr.get_n_trees() > 0:
            serialized_data = self.ptr.serialize(&error)

            if error != NULL:
                err_msg = error.decode("utf-8", "replace")
                free(error)
                warnings.warn(
                    f"Failed to serialize index data: {err_msg}",
                    RuntimeWarning
                )
                state["index_data"] = None
            else:
                # Convert C++ vector to Python bytes
                state["index_data"] = bytes(serialized_data)
        else:
            state["index_data"] = None

        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore state from dictionary.

        Parameters
        ----------
        state : dict
            State dictionary from get_state()

        Examples
        --------
        >>> index1 = Index(f=128, metric='angular', seed=42)
        >>> index1.add_item(0, [0.1] * 128)
        >>> index1.build()
        >>> state = index1.get_state()
        >>>
        >>> index2 = Index()
        >>> index2.set_state(state)
        >>> # index2 now has same data as index1
        """
        if not isinstance(state, dict):
            raise TypeError("State must be a dictionary")

        # Check version compatibility
        if "__version__" not in state:
            warnings.warn(
                "State has no version marker, compatibility not guaranteed",
                RuntimeWarning
            )

        # Restore parameters
        if "params" in state:
            params = state["params"]

            # Core parameters
            self.f = params.get("f", 0)
            self.metric = params.get("metric", "")
            self.metric_id = metric_from_string(
                params.get("metric", "").encode("utf-8")
            ) if params.get("metric") else METRIC_UNKNOWN
            self.n_neighbors = params.get("n_neighbors", 5)

            # Configuration
            seed = params.get("seed")
            if seed is not None:
                self.pending_seed = seed
                self.has_pending_seed = True

            verbose = params.get("verbose")
            if verbose is not None:
                self.pending_verbose = verbose
                self.has_pending_verbose = True

            # Persistence
            on_disk = params.get("on_disk_path")
            if on_disk:
                self.on_disk_path = on_disk.encode("utf-8")

            self.prefault = params.get("prefault", False)
            self.schema_version = params.get("schema_version", 0)

            # Extended parameters
            self.dtype = params.get("dtype", "float32")
            self.index_dtype = params.get("index_dtype", "int32")
            self.wrapper_dtype = params.get("wrapper_dtype", "uint64")
            self.random_dtype = params.get("random_dtype", "uint64")

        # Restore index data if present
        cdef bytes data_bytes
        cdef cpp_vector[uint8_t] data_vec
        cdef const unsigned char[:] view
        cdef char* error = NULL
        cdef cpp_bool success
        if "index_data" in state and state["index_data"] is not None:

            if not isinstance(state["index_data"], (bytes, bytearray)):
                raise TypeError("index_data must be bytes")

            # Ensure index is constructed
            if self.ptr == NULL:
                self._ensure_index()

            # Deserialize
            # Use typed memoryview and explicit pointer extraction.
            data_bytes = state["index_data"]

            if len(data_bytes) == 0:
                raise ValueError("index_data is empty")

            # Zero-copy memoryview over Python bytes
            view = data_bytes

            # if your serialized format has header magic, validate it before calling into C++. Fail fast > corrupt index.
            if view.shape[0] < 16:
                raise ValueError("index_data is too small to be valid")

            # This avoids accidental reallocation edge cases in some STL implementations.
            data_vec.clear()
            data_vec.reserve(view.shape[0])

            # Assign raw memory into C++ vector
            # data_vec.assign(data_bytes, data_bytes + len(data_bytes))  # Treating Python bytes as C pointer range.
            # data_vec.assign(&view[0], &view[0] + view.shape[0])
            data_vec.insert(data_vec.end(), &view[0], &view[0] + view.shape[0])

            success = self.ptr.deserialize(&data_vec, False, &error)
            if not success:
                if error != NULL:
                    err_msg = error.decode("utf-8", "replace")
                    free(error)
                    raise RuntimeError(f"Failed to deserialize index: {err_msg}")
                else:
                    raise RuntimeError("Failed to deserialize index (unknown error)")
        return self

    def serialize(self) -> Dict[str, Any]:
        """
        Serialize to JSON-compatible dictionary.

        Returns
        -------
        data : dict
            JSON-serializable state

        Examples
        --------
        >>> import json
        >>> index = Index(f=128, metric='angular', seed=42)
        >>> index.add_item(0, [0.1] * 128)
        >>> index.build()
        >>> data = index.serialize()
        >>> json_str = json.dumps(data, default=str)  # handle bytes
        """
        state = self.get_state()

        # Add metadata for JSON compatibility
        state["__class__"] = self.__class__.__name__
        state["__module__"] = self.__class__.__module__

        # Convert bytes to base64 for JSON
        if state.get("index_data") is not None:
            import base64
            state["index_data"] = base64.b64encode(
                state["index_data"]
            ).decode("ascii")

        return state

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> Self:
        """
        Deserialize from dictionary.

        Parameters
        ----------
        data : dict
            Serialized state from serialize()

        Returns
        -------
        index : Index
            Restored index instance

        Raises
        ------
        TypeError
            If data is not a dict
        ValueError
            If data format is invalid

        Examples
        --------
        >>> import json
        >>> index = Index(f=128, metric='angular', seed=42)
        >>> json_str = json.dumps(index.serialize(), default=str)
        >>> data = json.loads(json_str)
        >>> restored = Index.deserialize(data)
        """
        if not isinstance(data, dict):
            raise TypeError("Serialized state must be a dictionary")

        # Decode base64 if present
        if "index_data" in data and isinstance(data["index_data"], str):
            import base64
            data["index_data"] = base64.b64decode(data["index_data"])

        # Create new instance
        instance = cls()

        # Restore state
        instance.set_state(data)

        return instance

    def to_dict(self) -> Dict[str, Any]:
        """
        Alias for serialize().

        Returns
        -------
        dict
            Serialized state
        """
        return self.serialize()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Alias for deserialize().

        Parameters
        ----------
        data : dict
            Serialized state

        Returns
        -------
        Index
            Restored instance
        """
        return cls.deserialize(data)

    # =========================================================================
    # Future-Proof Helper Methods
    # =========================================================================

    def clone(self, **override_params) -> Self:
        """
        Create a copy of the index with optional parameter overrides.

        Parameters
        ----------
        **override_params : dict
            Parameters to override in the clone

        Returns
        -------
        index : Index
            New index with same parameters (but no data)

        Examples
        --------
        >>> index1 = Index(f=128, metric='angular', seed=42)
        >>> index2 = index1.clone(seed=123)  # Same f and metric, different seed
        """
        params = self.get_params()
        params.update(override_params)

        # Don't copy state, just parameters
        params_clean = {
            k: v for k, v in params.items()
            if k in ["f", "metric", "n_neighbors", "seed", "verbose",
                     "on_disk_path", "prefault", "schema_version",
                     "dtype", "index_dtype", "wrapper_dtype", "random_dtype"]
        }

        return self.__class__(**params_clean)

    def _validate_dtype_compatibility(self) -> bool:
        """
        Validate that current dtype settings are compatible.

        Returns
        -------
        compatible : bool
            True if configuration is valid

        Notes
        -----
        Currently only validates that values are set.
        Future versions will check actual compatibility.
        """
        # Future: validate dtype combinations
        # For now, just check they're set
        return (
            self.dtype is not None and
            self.index_dtype is not None and
            self.wrapper_dtype is not None and
            self.random_dtype is not None
        )

    def is_built(self) -> bool:
        """
        Check if index has been built.

        Returns
        -------
        built : bool
            True if build() has been called
        """
        if self.ptr == NULL:
            return False
        return self.ptr.get_n_trees() > 0

    def is_empty(self) -> bool:
        """
        Check if index has no items.

        Returns
        -------
        empty : bool
            True if no items added
        """
        if self.ptr == NULL:
            return True
        return self.ptr.get_n_items() == 0

    # =========================================================================
    # Sequence Protocol (Exact Implementation from annoymodule.cc)
    # =========================================================================

    def __len__(self) -> int:
        """
        Return number of items in the index.

        Implements sequence protocol for Pythonic access.
        Matches py_an_len from annoymodule.cc.

        Returns
        -------
        length : int
            Number of items (0 if not constructed)

        Examples
        --------
        >>> index = Index(f=10, metric='angular')
        >>> for i in range(100):
        ...     index.add_item(i, [random.random() for _ in range(10)])
        >>> len(index)
        100

        >>> if len(index) > 50:
        ...     print("Index has enough data")
        """
        if self.ptr == NULL:
            return 0

        cdef int32_t n = self.ptr.get_n_items()
        if n < 0:
            n = 0  # Defensive (matches C implementation)
        return n

    def __getitem__(self, key):
        """
        Retrieve item(s) by ID (sequence protocol).

        Supports:
        * Integer indexing: index[5]
        * Slice notation: index[0:10]
        * Negative indexing: index[-1]

        Parameters
        ----------
        key : int or slice
            Item ID or slice

        Returns
        -------
        vector : list[float] or list[list[float]]
            Single vector or list of vectors

        Raises
        ------
        IndexError
            If key is out of range
        TypeError
            If key is not int or slice
        RuntimeError
            If index not constructed

        Examples
        --------
        >>> # Single item
        >>> vec = index[0]
        >>>
        >>> # Slice
        >>> vecs = index[0:10]
        >>>
        >>> # Negative indexing
        >>> last = index[-1]
        >>>
        >>> # Iteration
        >>> for vec in index[:100]:
        ...     process(vec)
        """
        if self.ptr == NULL:
            raise RuntimeError("Index not constructed")

        cdef int32_t n_items = self.ptr.get_n_items()

        # Integer indexing
        if isinstance(key, int):
            # Support negative indexing
            if key < 0:
                key = n_items + key

            if key < 0 or key >= n_items:
                raise IndexError(f"Index {key} out of range [0, {n_items})")

            return self.get_item(key)

        # Slice
        elif isinstance(key, slice):
            start, stop, step = key.indices(n_items)

            if step != 1:
                indices = range(start, stop, step)
            else:
                indices = range(start, stop)

            return [self.get_item(i) for i in indices]

        else:
            raise TypeError(
                f"Index key must be int or slice, not {type(key).__name__}"
            )

    def __contains__(self, int item) -> bool:
        """
        Check if item ID exists in index.

        Parameters
        ----------
        item : int
            Item ID to check

        Returns
        -------
        exists : bool
            True if item exists

        Examples
        --------
        >>> if 42 in index:
        ...     print("Item 42 exists")
        """
        if self.ptr == NULL:
            return False

        if item < 0:
            return False

        return item < self.ptr.get_n_items()

    def __iter__(self):
        """
        Iterate over all vectors in the index.

        Yields
        ------
        vector : list[float]
            Each stored vector in order

        Examples
        --------
        >>> for vec in index:
        ...     print(vec)

        >>> # With enumerate
        >>> for i, vec in enumerate(index):
        ...     print(f"Item {i}: {vec}")
        """
        if self.ptr == NULL:
            return

        cdef int32_t n_items = self.ptr.get_n_items()
        cdef int32_t i

        for i in range(n_items):
            yield self.get_item(i)

    # =========================================================================
    # Rich Representations (Exact Implementation from annoymodule.cc)
    # =========================================================================

    def repr_info(
        self,
        bint include_n_items=True,
        bint include_n_trees=True,
        include_memory=None
    ) -> str:
        """
        Rich dictionary-like string representation.

        Parameters
        ----------
        include_n_items : bool, default=True
            Include item count
        include_n_trees : bool, default=True
            Include tree count
        include_memory : bool or None, default=None
            Include memory usage estimate
            If None, includes only if index is built

        Returns
        -------
        repr_str : str
            Dictionary-style representation

        Examples
        --------
        >>> print(index.repr_info())
        Annoy(**{'f': 128, 'metric': 'angular', 'n_items': 1000, 'n_trees': 10})
        """
        info = {
            "f": self.f,
            "metric": self.metric,
            "n_neighbors": self.n_neighbors,
        }

        if self.ptr != NULL:
            if include_n_items:
                info["n_items"] = self.ptr.get_n_items()

            if include_n_trees:
                info["n_trees"] = self.ptr.get_n_trees()

            # Memory usage (expensive, only if requested)
            if include_memory is None:
                include_memory = (self.ptr.get_n_trees() > 0)

            if include_memory and self.ptr.get_n_trees() > 0:
                # Estimate: f * n_items * 4 bytes + tree overhead
                n_items = self.ptr.get_n_items()
                n_trees = self.ptr.get_n_trees()
                memory_bytes = self.f * n_items * 4 + n_trees * 1024
                info["memory_mb"] = round(memory_bytes / (1024 * 1024), 2)

        return f"Annoy(**{info!r})"

    def _repr_html_(self) -> str:
        """
        Rich HTML representation for Jupyter notebooks.

        Creates an interactive, expandable widget similar to sklearn estimators.
        Matches the exact implementation from annoymodule.cc.

        Returns
        -------
        html : str
            HTML string for notebook display

        Notes
        -----
        * Jupyter automatically calls this for rich display
        * Creates collapsible sections for parameters
        * Includes copy buttons for easy parameter extraction
        * Links to documentation (dev and stable)

        Examples
        --------
        >>> # In Jupyter:
        >>> index = Index(f=128, metric='angular')
        >>> index  # Displays rich HTML widget automatically
        """
        # Generate unique ID (matches C implementation)
        global g_annoy_repr_html_seq
        g_annoy_repr_html_seq += 1

        # cdef unsigned long long seq = g_annoy_repr_html_seq
        cdef size_t salt = <size_t>&g_annoy_repr_html_seq
        repr_id = f"annoy-repr-{id(self):x}-{salt:x}-{g_annoy_repr_html_seq}"

        # Try to get scikitplot version for docs link
        stable_version = None
        try:
            import sys
            if "scikitplot" in sys.modules:
                import scikitplot
                if hasattr(scikitplot, "__version__"):
                    ver = scikitplot.__version__
                    # Parse major.minor
                    parts = ver.split(".")
                    if len(parts) >= 2:
                        try:
                            major = int(parts[0])
                            minor = int(parts[1])
                            stable_version = f"{major}.{minor}"
                        except ValueError:
                            pass
        except Exception:
            pass

        # Build HTML (exact structure from C code)
        html_parts = []

        # CSS (embedded fallback from C code)
        css = f"""
<style>
#{repr_id} .annoy-box{{border:1px solid #d0d7de;border-radius:6px;display:inline-block;min-width:280px;}}
#{repr_id} details{{margin:0;padding:0;}}
#{repr_id} summary{{cursor:pointer;list-style:none;display:flex;align-items:center;gap:8px;padding:8px 10px;font:12px/1.35 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;}}
#{repr_id} summary::-webkit-details-marker{{display:none;}}
#{repr_id} .annoy-title{{font-weight:600;}}
#{repr_id} .annoy-links{{margin-left:auto;display:flex;align-items:center;gap:8px;}}
#{repr_id} .annoy-links a{{color:#0969da;text-decoration:none;}}
#{repr_id} .annoy-links a:hover{{text-decoration:underline; background-color: lightgreen;}}
#{repr_id} .annoy-sep{{color:#57606a;}}
#{repr_id} .annoy-subtitle{{font-weight:600;}}
#{repr_id} .annoy-arrow::before{{content:'\\25B6';display:inline-block;width:14px;}}
#{repr_id} details[open] > summary .annoy-arrow::before{{content:'\\25BC';}}
#{repr_id} .annoy-body{{padding:0 10px 10px 10px;}}
#{repr_id} .annoy-table{{border-collapse:collapse;width:100%;font:12px/1.35 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;}}
#{repr_id} .annoy-table th,.annoy-table td{{border-top:1px solid #eaeef2;padding:6px 6px;text-align:left;vertical-align:top;}}
#{repr_id} .annoy-table th{{font-weight:600;}}
#{repr_id} .annoy-td-btn{{width:72px;}}
#{repr_id} .annoy-copy{{font:11px/1 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;border:1px solid #d0d7de;border-radius:6px;padding:3px 6px;background:#f6f8fa;cursor:pointer;}}
#{repr_id} .annoy-copy:active{{transform:translateY(1px);}}
#{repr_id} .annoy-key{{white-space:nowrap;}}
#{repr_id} .annoy-value{{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,'Liberation Mono','Courier New',monospace;}}
</style>
"""

        html_parts.append(f'<div id="{repr_id}" class="annoy-repr">')
        html_parts.append(css)
        html_parts.append('<div class="annoy-box">')
        html_parts.append('<details class="annoy-main">')  # open
        html_parts.append("<summary>")
        html_parts.append('<span class="annoy-arrow"></span>')
        html_parts.append('<span class="annoy-title">Annoy</span>')

        # Documentation links
        html_parts.append('<span class="annoy-links">')
        # html_parts.append('<a href="https://scikit-plots.github.io/dev/modules/generated/scikitplot.cexternals._annoy.Annoy.html" target="_blank" rel="noopener noreferrer" title="Annoy docs (dev)">dev</a>')
        # if stable_version:
        #     html_parts.append('<span class="annoy-sep">|</span>')
        #     html_parts.append(f'<a href="https://scikit-plots.github.io/{stable_version}/modules/generated/scikitplot.cexternals._annoy.Annoy.html" target="_blank" rel="noopener noreferrer" title="Annoy docs (installed)">{stable_version}</a>')

        # ❓ Get doc link ⍰
        doc_link = self._get_doc_link()
        html_parts.append(f'<a href="{doc_link}" target="_blank" rel="noopener noreferrer" title="Annoy docs (installed)">⍰</a>')
        html_parts.append("</span>")

        html_parts.append("</summary>")
        html_parts.append('<div class="annoy-body">')

        # Parameters table
        html_parts.append('<details class="annoy-sub">')
        html_parts.append('<summary><span class="annoy-arrow"></span><span class="annoy-subtitle">Parameters</span></summary>')
        html_parts.append('<table class="annoy-table">')
        html_parts.append("<thead><tr><th></th><th>Parameter</th><th>Value</th></tr></thead>")
        html_parts.append("<tbody>")

        # Helper function for HTML escaping
        def html_escape(s):
            if s is None:
                return "None"
            s = str(s)
            s = s.replace("&", "&amp;")
            s = s.replace("<", "&lt;")
            s = s.replace(">", "&gt;")
            s = s.replace('"', "&quot;")
            s = s.replace("'", "&#x27;")
            return s

        # Add parameter rows (stable order matching C code)
        params = self.get_params()

        # for key in ['f', 'metric', 'n_neighbors', 'on_disk_path', 'prefault',
        #             'seed', 'verbose', 'schema_version']:
        for key, value in params.items():
            if key in params:
                value = params[key]
                value_repr = repr(value)
                html_parts.append("<tr>")
                html_parts.append('<td class="annoy-td-btn">')
                html_parts.append('<button type="button" class="annoy-copy" title="Copy value" aria-label="Copy value">🗗 Copy</button>')
                html_parts.append("</td>")
                html_parts.append(f'<td class="annoy-key">{html_escape(key)}</td>')
                html_parts.append(f'<td class="annoy-value">{html_escape(value_repr)}</td>')
                html_parts.append("</tr>")

        html_parts.append("</tbody>")
        html_parts.append("</table>")
        html_parts.append("</details>")  # parameters
        html_parts.append("</div>")      # body
        html_parts.append("</details>")  # main
        html_parts.append("</div>")      # box

        # JavaScript (exact from C code)
        js = f"""
<script>
(function(){{
  var root=document.getElementById('{repr_id}');
  if(!root)return;
  var btns=root.querySelectorAll('button.annoy-copy');
  for(var i=0;i<btns.length;i++){{
    btns[i].addEventListener('click',function(e){{
      e.preventDefault();
      var tr=this.closest('tr'); if(!tr) return;
      var val=tr.querySelector('.annoy-value'); if(!val) return;
      var txt=val.textContent || ''; if(!txt) return;

      function done(btn){{
        var old=btn.textContent;
        btn.textContent='✔︎ Copied';
        setTimeout(function(){{btn.textContent=old;}},800);
      }}

      if(navigator.clipboard && navigator.clipboard.writeText){{
        navigator.clipboard.writeText(txt).then(done.bind(null,this),function(){{done(this);}}.bind(this));
      }} else {{
        var ta=document.createElement('textarea');
        ta.value=txt;
        ta.style.position='fixed';
        ta.style.left='-9999px';
        document.body.appendChild(ta);
        ta.select();
        try{{document.execCommand('copy');}}catch(_e){{}}
        document.body.removeChild(ta);
        done(this);
      }}
    }});
  }}
}})();
</script>
"""

        html_parts.append(js)
        html_parts.append("</div>")  # outer container

        return "".join(html_parts)

    _html_repr = _repr_html_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
