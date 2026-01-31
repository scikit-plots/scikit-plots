# cython: language_level=3

"""Medium Cython template: fused-type add."""

ctypedef fused number_t:
    int
    long
    double


def add(number_t a, number_t b):
    return a + b
