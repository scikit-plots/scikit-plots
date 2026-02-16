# cython: language_level=3
"""
Fused type add demo.
"""
ctypedef fused number_t:
    int
    double

cpdef number_t add(number_t a, number_t b):
    return a + b
