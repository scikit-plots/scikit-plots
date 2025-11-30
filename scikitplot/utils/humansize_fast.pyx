cdef list SUFFIXES = ["B", "KB", "MB", "GB", "TB", "PB"]

def humansize_fast(double n):
    """
    Extremely fast scalar conversion of bytes → human size.
    """
    cdef int i = 0
    cdef double x = n
    cdef bint negative = x < 0

    if negative:
        x = -x

    while x >= 1024.0 and i < len(SUFFIXES) - 1:
        x /= 1024.0
        i += 1

    return (("-" if negative else "") +
            ("%.2f" % x).rstrip("0").rstrip(".") +
            " " + SUFFIXES[i])
