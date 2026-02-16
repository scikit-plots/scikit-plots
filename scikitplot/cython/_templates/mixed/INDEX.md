# Template index

Each template is a standalone `.pyx` file. Use `compile_template(name)` to compile.

| Template | Focus | NumPy required |
|---|---|---|
| `t01_square_int` | Typed int function; minimal example. | no |
| `t02_fib_cpdef` | cpdef for dual C/Python calling; iterative Fibonacci. | no |
| `t03_cdef_class_counter` | cdef class with typed attribute and methods. | no |
| `t04_memoryview_sum` | Typed memoryviews without NumPy dependency. | no |
| `t05_numpy_ndarray_sum` | NumPy typed ndarray with cimport numpy. | yes |
| `t06_directives_boundscheck` | Compiler directives to disable boundscheck/wraparound in a block. | yes |
| `t07_libc_math` | Use libc math functions; typed loops. | no |
| `t08_struct_point` | cdef struct and distance calculation. | no |
| `t09_enum_state` | cdef enum; map states to ints. | no |
| `t10_safe_div_except` | cpdef with exception contract; deterministic error handling. | no |
| `t11_fused_types_dot` | Fused types generic dot product for float/double. | no |
| `t12_inline_clamp` | cdef inline helper; clamp values. | no |
| `t13_bytes_xor` | Operate on bytes via memoryview; XOR with key. | no |
| `t14_string_reverse` | String manipulation; reverse deterministically. | no |
| `t15_lcg_rng` | Deterministic PRNG in cdef class; reproducible. | no |
| `t16_insertion_sort` | In-place insertion sort on memoryview (small arrays). | no |
| `t17_kahan_sum` | Kahan summation for numerical stability. | no |
| `t18_histogram_int` | Histogram counting for int32 array (NumPy). | yes |
| `t19_popcount` | Portable popcount for 32-bit ints. | no |
| `t20_matmul_small` | Naive matrix multiply using memoryviews (2D). | yes |
