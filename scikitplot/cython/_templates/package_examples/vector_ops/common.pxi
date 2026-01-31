# Shared helper for package examples.
cdef inline Py_ssize_t _require_same_len(double[:] a, double[:] b):
    if a.shape[0] != b.shape[0]:
        raise ValueError("length mismatch")
    return a.shape[0]
