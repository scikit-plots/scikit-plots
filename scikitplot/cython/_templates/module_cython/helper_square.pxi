# This file is included by a `.pyx` template via: include "helper_square.pxi"
# It demonstrates splitting code across multiple files.

cdef inline int _square_int(int n) nogil:
    return n * n
