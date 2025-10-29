# stream_core.pyx
# cython: language_level=3
from cython cimport boundscheck, wraparound

@boundscheck(False)
@wraparound(False)
cpdef update_windows(
    list records,
    list header,
    list acc,
    list cnt,
    list windows,
    double time
):

    cdef Py_ssize_t nw = len(windows)
    cdef Py_ssize_t i, idx, nrec = len(records)
    cdef double cutoff
    cdef list rec

    for i in range(nw):
        cutoff = time - windows[i]
        idx = header[i]
        while idx < nrec:
            rec = records[idx]
            if rec[0] >= cutoff:
                break
            acc[i] -= rec[1]
            cnt[i] -= 1
            idx += 1
        header[i] = idx
