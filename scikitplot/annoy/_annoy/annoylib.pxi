
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
DEF MAX_ITEMS_32 = 2147483647  # 2^31 - 1 = 2**31 -1
DEF MAX_ITEMS_64 = 9223372036854775807  # 2^63 - 1 = 2**63 -1
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
DEF DEFAULT_DATA_DTYPE = "float"
