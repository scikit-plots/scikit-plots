# cython: language_level=3
#
# scikitplot/cexternals/_editdistance/bycython.pxd

cpdef unsigned int eval(object a, object b) except 0xffffffffffffffff
cpdef bint eval_criterion(object a, object b, const unsigned int thr) except 0xffffffffffffffff
