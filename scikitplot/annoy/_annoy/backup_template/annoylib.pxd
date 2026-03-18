# cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, binding=True, embedsignature=True
# distutils: language = c++
# distutils: extra_compile_args = -std=c++14 -O3 -march=native -DNDEBUG -pthread -DANNOYLIB_MULTITHREADED_BUILD
# distutils: extra_link_args = -std=c++14
# cython: warn.unused=False
#
# cython annoylib.pyx --cplus
# g++ -std=c++14 -O3 -fPIC -DANNOYLIB_MULTITHREADED_BUILD -pthread \
#     -c annoylib.cpp -o annoylib.o
# g++ -shared annoylib.o -lpthread -o annoylib.so

"""
Cython Declaration File for Annoy C++ Library.

This file declares the C++ interfaces from annoylib.h for use in Cython.

Design Principles:

1. **Explicit Type Declarations**: All C++ types explicitly declared
2. **Safe Memory Management**: Proper pointer handling with nogil where safe
3. **Error Propagation**: char** error pattern for C++ exceptions
4. **Platform Compatibility**: Windows, Linux, macOS support

Type System:

* S (index type): int32_t (item/node identifiers)
* T (data type): float (embedding values)
* Distance: float (distance values, potentially clipped for Hamming)
* Random: uint64_t (seed type)

Notes
-----
* All methods that can throw C++ exceptions use char** error parameter
* nogil blocks are used only for pure computational methods
* GIL is held during Python object manipulation

🧠 What .pxd Is Allowed To Contain - C-level declaration headers
Allowed:
    - cdef enum
    - cdef struct
    - cdef cppclass
    - cdef extern
    - cdef inline functions that are pure C-level
Not allowed:
    - Returning str
    - Raising ValueError
    - Creating Python objects
    - Runtime Python logic

Code Type                    Where It Goes     Why
Python dict {}               .pyx              Runtime Python objects
Function raising exception   .pyx              Python runtime feature
Pure C cdef inline           .pxd              No Python objects/exceptions
cdef extern                  .pxd              C/C++ declarations
cdef enum/struct             .pxd              Type declarations

References:
    - annoylib.h: C++ template implementation
    - https://en.cppreference.com/w/cpp/types/integer.html
    - https://cython-guidelines.readthedocs.io/en/latest/articles/lie.html
    - Cython fused types: https://cython.readthedocs.io/en/latest/src/userguide/fusedtypes.html
    - https://cythoncython.readthedocs.io/en/latest/src/userguide/external_C_code.html#c-api-declarations
    - https://cythoncython.readthedocs.io/en/latest/src/userguide/external_C_code.html#public-declarations
    - https://cythoncython.readthedocs.io/en/latest/src/userguide/sharing_declarations.html#sharing-declarations
    - https://cythoncython.readthedocs.io/en/latest/src/userguide/external_C_code.html#struct-union-enum-styles
    - https://cythoncython.readthedocs.io/en/latest/src/userguide/language_basics.html?highlight=cpdef%20enum#type-casting
    - https://cythoncython.readthedocs.io/en/latest/src/userguide/external_C_code.html#releasing-the-gil
    - https://cythoncython.readthedocs.io/en/latest/src/userguide/external_C_code.html#acquiring-the-gil
    - https://cythoncython.readthedocs.io/en/latest/src/userguide/external_C_code.html#declaring-a-function-as-callable-without-the-gil
    - https://cython.readthedocs.io/en/latest/src/userguide/external_C_code.html#resolving-naming-conflicts-c-name-specifications
    - https://cython.readthedocs.io/en/latest/src/userguide/external_C_code.html#including-verbatim-c-code
    - https://docs.cython.org/en/latest/src/userguide/wrapping_CPlusPlus.html#templates
    - https://docs.cython.org/en/latest/src/userguide/extension_types.html#attribute-name-matching-and-aliasing
"""

from libc.stdint cimport int32_t, uint32_t, uint64_t, uint8_t
from libcpp cimport bool as cpp_bool
from libcpp.string cimport string as cpp_string
from libcpp.vector cimport vector as cpp_vector
from libcpp.unordered_map cimport unordered_map

# ==============================================================================
# KISS Random Number Generator Interface
# ==============================================================================

cdef extern from "../../cexternals/_annoy/src/kissrandom.h" namespace "Annoy" nogil:

    # 64-bit KISS random generator (used by default)
    cdef cppclass Kiss64Random nogil:
        # Type aliases
        ctypedef uint64_t result_type

        # Constants
        @staticmethod
        uint64_t default_seed

        # Constructors
        Kiss64Random() except +
        Kiss64Random(uint64_t seed) except +

        # Core methods
        void reset(uint64_t seed) except +
        void reset_default() except +
        void set_seed(uint64_t seed) except +
        uint64_t kiss() except +
        int flip() except +
        size_t index(size_t n) except +

        # Static helpers
        @staticmethod
        uint64_t get_default_seed() nogil

        @staticmethod
        uint64_t normalize_seed(uint64_t seed) nogil

# ==============================================================================
# Annoy Index Interface (Base Class)
# ==============================================================================

cdef extern from "../../cexternals/_annoy/src/annoylib.h" namespace "Annoy" nogil:

    # Base interface for all Annoy index types
    cdef cppclass AnnoyIndexInterface[S, T, Random] nogil:
        """
        Abstract base interface for Annoy indices.

        Lifecycle
        ---------
        1. Construction: AnnoyIndex(f) where f > 0
        2. Building: add_item() repeatedly, then build()
        3. Querying: get_nns_by_item(), get_nns_by_vector()
        4. Persistence: save(), load(), serialize(), deserialize()
        5. Destruction: unload(), then delete

        State Transitions
        -----------------
        * EMPTY → BUILDING: first add_item()
        * BUILDING → BUILT: build()
        * BUILT → BUILDING: unbuild()
        * BUILT → LOADED: load()
        * Any → EMPTY: unload()
        """

        # ==================== Core Building ====================

        cpp_bool add_item(S item, const T* embedding, char** error) except +

        cpp_bool build(int n_trees, int n_threads, char** error) except +

        cpp_bool unbuild(char** error) except +

        # ==================== Querying ====================

        void get_nns_by_item(
            S item,
            size_t n,
            int search_k,
            cpp_vector[S]* result,
            cpp_vector[T]* distances
        ) except +

        void get_nns_by_vector(
            const T* embedding,
            size_t n,
            int search_k,
            cpp_vector[S]* result,
            cpp_vector[T]* distances
        ) except +

        T get_distance(S i, S j) except +

        void get_item(S item, T* embedding) except +

        # ==================== Metadata ====================

        S get_n_items() nogil

        S get_n_trees() nogil

        int get_f() nogil

        # ==================== Persistence (Disk) ====================

        cpp_bool save(const char* filename, cpp_bool prefault, char** error) except +

        cpp_bool load(const char* filename, cpp_bool prefault, char** error) except +

        cpp_bool on_disk_build(const char* filename, char** error) except +

        void unload() except +

        # ==================== Persistence (Memory) ====================

        cpp_vector[uint8_t] serialize(char** error) except +

        cpp_bool deserialize(cpp_vector[uint8_t]* bytes, cpp_bool prefault, char** error) except +

        # ==================== Configuration ====================

        void set_seed(Random seed) except +

        void set_verbose(cpp_bool v) except +

# ==============================================================================
# Concrete Annoy Index Types (Metric-Specific)
# ==============================================================================

cdef extern from "../../cexternals/_annoy/src/annoylib.h" namespace "Annoy" nogil:

    # Forward declarations of metric types (defined in annoylib.h)
    cdef cppclass Angular nogil:
        pass

    cdef cppclass Euclidean nogil:
        pass

    cdef cppclass Manhattan nogil:
        pass

    cdef cppclass DotProduct nogil:
        pass

    cdef cppclass Hamming nogil:
        pass

    # Build policy types
    cdef cppclass AnnoyIndexSingleThreadedBuildPolicy nogil:
        pass

    cdef cppclass AnnoyIndexThreadedBuildPolicy nogil:
        pass

    # Generic AnnoyIndex template
    cdef cppclass AnnoyIndex[S, T, Distance, Random, ThreadedBuildPolicy](AnnoyIndexInterface[S, T, Random]) nogil:
        """
        Generic templated Annoy index.

        Template Parameters
        -------------------
        S : int32_t
            Index type
        T : float
            Data type
        Distance : {Angular, Euclidean, Manhattan, DotProduct}
            Metric type
        Random : Kiss64Random
            RNG type
        ThreadedBuildPolicy : {Single, Multi}ThreadedBuildPolicy
            Threading strategy
        """

        AnnoyIndex() except +
        AnnoyIndex(int f) except +

    # Hamming wrapper (special case: packs float embeddings into binary)
    cdef cppclass HammingWrapper[S, T, InternalT, Random, ThreadedBuildPolicy](AnnoyIndexInterface[S, T, Random]) nogil:
        """
        Hamming metric wrapper with float-to-binary packing.

        Embedding Interpretation
        ------------------------
        * Input: float array of length f
        * Each float is thresholded: val >= 0.5 → 1, else → 0
        * Packed into uint32_t or uint64_t words for efficient XOR
        * Distance is Hamming distance (popcount of XOR)
        * Distance is clipped to [0, f]

        Memory Layout
        -------------
        * f_external: user-facing dimension (float array length)
        * f_internal: packed dimension (ceil(f_external / bits_per_word))
        * Storage: f_internal * sizeof(InternalT) bytes per vector
        """

        HammingWrapper() except +
        HammingWrapper(int f) except +

    cdef cppclass float16_t:
        float16_t() except +
        float16_t(float) except +
        float16_t& operator=(float) except +
        # DO NOT declare operator float() here

    # float safe_numeric_cast(const float16_t&);
    # float float16_to_float(const float16_t&);
    # float16_t float_to_float16(float);

    # Concrete typedefs (these must match the actual C++ code structure)
    # ctypedef AnnoyIndex[int32_t, float16_t, Angular, Kiss64Random, AnnoyIndexThreadedBuildPolicy] \
    #     AnnoyAngularIndex "AnnoyIndex<int32_t, float16_t, Annoy::Angular, Annoy::Kiss64Random, Annoy::AnnoyIndexThreadedBuildPolicy>"
    # ctypedef AnnoyIndex[int32_t, float16_t, Euclidean, Kiss64Random, AnnoyIndexThreadedBuildPolicy] \
    #     AnnoyEuclideanIndex "AnnoyIndex<int32_t, float16_t, Annoy::Euclidean, Annoy::Kiss64Random, Annoy::AnnoyIndexThreadedBuildPolicy>"
    # ctypedef AnnoyIndex[int32_t, float16_t, Manhattan, Kiss64Random, AnnoyIndexThreadedBuildPolicy] \
    #     AnnoyManhattanIndex "AnnoyIndex<int32_t, float16_t, Annoy::Manhattan, Annoy::Kiss64Random, Annoy::AnnoyIndexThreadedBuildPolicy>"
    # ctypedef AnnoyIndex[int32_t, float16_t, DotProduct, Kiss64Random, AnnoyIndexThreadedBuildPolicy] \
    #     AnnoyDotIndex "AnnoyIndex<int32_t, float16_t, Annoy::DotProduct, Annoy::Kiss64Random, Annoy::AnnoyIndexThreadedBuildPolicy>"
    # ctypedef HammingWrapper[int32_t, float16_t, uint64_t, Kiss64Random, AnnoyIndexThreadedBuildPolicy] \
    #     AnnoyHammingIndex "HammingWrapper<int32_t, float16_t, uint64_t, Annoy::Kiss64Random, Annoy::AnnoyIndexThreadedBuildPolicy>"

    # Concrete typedefs (these must match the actual C++ code structure)
    ctypedef AnnoyIndex[int32_t, float, Angular, Kiss64Random, AnnoyIndexThreadedBuildPolicy] \
        AnnoyAngularIndex "AnnoyIndex<int32_t, float, Annoy::Angular, Annoy::Kiss64Random, Annoy::AnnoyIndexThreadedBuildPolicy>"
    ctypedef AnnoyIndex[int32_t, float, Euclidean, Kiss64Random, AnnoyIndexThreadedBuildPolicy] \
        AnnoyEuclideanIndex "AnnoyIndex<int32_t, float, Annoy::Euclidean, Annoy::Kiss64Random, Annoy::AnnoyIndexThreadedBuildPolicy>"
    ctypedef AnnoyIndex[int32_t, float, Manhattan, Kiss64Random, AnnoyIndexThreadedBuildPolicy] \
        AnnoyManhattanIndex "AnnoyIndex<int32_t, float, Annoy::Manhattan, Annoy::Kiss64Random, Annoy::AnnoyIndexThreadedBuildPolicy>"
    ctypedef AnnoyIndex[int32_t, float, DotProduct, Kiss64Random, AnnoyIndexThreadedBuildPolicy] \
        AnnoyDotIndex "AnnoyIndex<int32_t, float, Annoy::DotProduct, Annoy::Kiss64Random, Annoy::AnnoyIndexThreadedBuildPolicy>"
    ctypedef HammingWrapper[int32_t, float, uint64_t, Kiss64Random, AnnoyIndexThreadedBuildPolicy] \
        AnnoyHammingIndex "HammingWrapper<int32_t, float, uint64_t, Annoy::Kiss64Random, Annoy::AnnoyIndexThreadedBuildPolicy>"

# ==============================================================================
# Helper Functions
# ==============================================================================

cdef extern from "../../cexternals/_annoy/src/annoylib.h" namespace "Annoy" nogil:
    # Metric string normalization (not exposed directly in annoylib.h,
    # but implemented in Python extension; we declare it here for completeness)
    pass

# ==============================================================================
# Memory Management Helpers
# ==============================================================================

cdef extern from "<cstdlib>" namespace "std" nogil:

    void free(void* ptr)

cdef extern from "<cstring>" namespace "std" nogil:

    char* strdup(const char* s)
