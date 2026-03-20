
# include "annoylib.pxi"

# ==============================================================================
# Constants Configuration
# ==============================================================================

DEF ANNOY_VERSION_MAJOR = 1
DEF ANNOY_VERSION_MINOR = 17
DEF ANNOY_VERSION_PATCH = 3

# Default parameters (from annoymodule.cc)
DEF DEFAULT_N_TREES = 10
DEF DEFAULT_SEARCH_K = -1
DEF DEFAULT_N_THREADS = -1
DEF DEFAULT_N_NEIGHBORS = 5
DEF DEFAULT_PREFAULT = False
DEF DEFAULT_SCHEMA_VERSION = 1

# Memory and size limits
DEF MAX_DIMENSION = 1000000

# Index type item-ID upper bounds (non-negative IDs only; all validated via int64_t bridge)
# Signed small integer types: upper bound = max positive value of the type
DEF MAX_ITEMS_8 = 126  # int8_t:   2^7  - 2  (max item id; n_items = id+1 must fit in int8_t)
DEF MAX_ITEMS_16 = 32766  # int16_t:  2^15 - 2
DEF MAX_ITEMS_32 = 2147483646  # int32_t:  2^31 - 2
DEF MAX_ITEMS_64 = 9223372036854775806  # int64_t:  2^63 - 2
# Unsigned integer types: upper bound = type max (all fit in int64_t bridge except uint64_t)
DEF MAX_ITEMS_U8 = 254  # uint8_t:  2^8  - 2
DEF MAX_ITEMS_U16 = 65534  # uint16_t: 2^16 - 2
DEF MAX_ITEMS_U32 = 4294967294  # uint32_t: 2^32 - 2
# uint64_t: bridge limit = INT64_MAX (values 2^63..2^64-1 cannot be passed via int64_t _w bridge)
DEF MAX_ITEMS_U64 = 18446744073709551614  # uint64_t: 2^64 - 2
DEF MAX_TREES = 10000
DEF MAX_SEARCH_K = 1000000
DEF MAX_THREADS = 256

# Node and memory sizes (bytes)
DEF NODE_HEADER_SIZE = 16
DEF VECTOR_ALIGNMENT = 32  # AVX alignment
DEF PAGE_SIZE = 4096

# File format constants
DEF ANNOY_MAGIC = 0x414E4E59  # "ANNY" in hex
DEF ANNOY_SCHEMA_VERSION_PORTABLE = 1
DEF ANNOY_SCHEMA_VERSION_CANONICAL = 2

# Seed constants
DEF ANNOY_DEFAULT_SEED = 1234567890987654321ULL  # Kiss64Random::default_seed

# Custom metric limits
DEF MAX_CUSTOM_METRIC_NAME_LEN = 256

# Defaults
DEF DEFAULT_METRIC = "angular"
DEF DEFAULT_INDEX_DTYPE = "int32"
DEF DEFAULT_DATA_DTYPE = "float32"
