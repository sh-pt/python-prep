# ewm_fast.pyx
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
from libc.math cimport isnan, NAN
from libc.stdlib cimport calloc, free

ctypedef cnp.float64_t float64_t
ctypedef cnp.int64_t   int64_t

cpdef ewm_kernel(
        cnp.ndarray[float64_t, ndim=1, mode="c"] x,
        cnp.ndarray[float64_t, ndim=1, mode="c"] w_arr,
        cnp.ndarray[int64_t, ndim=1, mode="c"] g_codes,
        double alpha,
        long n_groups,
        cnp.ndarray[float64_t, ndim=1] out
):
    cdef Py_ssize_t n = x.shape[0]

    cdef Py_ssize_t i
    cdef double xi, wi, s_prev, w_prev, s_now, w_now
    cdef long gi

    # Allocate state arrays using C (GIL-FREE)
    cdef float64_t *s_acc = NULL
    cdef float64_t *w_acc = NULL

    s_acc = <float64_t *> calloc(n_groups, sizeof(float64_t))
    w_acc = <float64_t *> calloc(n_groups, sizeof(float64_t))

    try:
        with nogil:
            for i in range(n):
                xi = x[i]
                wi = w_arr[i]
                gi = g_codes[i]

                # We are now accessing C pointers
                s_prev = s_acc[gi]
                w_prev = w_acc[gi]

                if isnan(xi) or wi == 0.0:
                    s_now = s_prev * (1.0 - alpha)
                    w_now = w_prev * (1.0 - alpha)
                else:
                    s_now = alpha * wi * xi + (1.0 - alpha) * s_prev
                    w_now = alpha * wi + (1.0 - alpha) * w_prev

                s_acc[gi] = s_now
                w_acc[gi] = w_now

                if w_now != 0.0:
                    out[i] = s_now / w_now
                else:
                    out[i] = NAN
    finally:
        free(s_acc)
        free(w_acc)