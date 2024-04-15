import cython
from cython.view cimport array as cvarray
from cython.cimports.libc.stdlib cimport malloc, calloc, free

cimport numpy as np
import numpy as np
# import scipy.signal as sig
import time

cdef extern from "math.h":
    double cos(double x)
cdef extern from "math.h":
    double exp(double x)
cdef extern from "math.h":
    double fabs(double x)
cdef extern from "math.h":
    double atan(double x)
cdef extern from "math.h":
    double M_PI

@cython.boundscheck(False)
@cython.wraparound(False)
def pll_signal(int ltimes, 
                np.ndarray[float, ndim=1, mode="c"] x not None, 
                np.ndarray[float, ndim=1, mode="c"] lock not None,
                np.ndarray[float, ndim=1, mode="c"] freq not None,
                np.ndarray[float, ndim=1, mode="c"] vco not None,
                double wo, double G0, double G1, double G2, double Fs, double agcgain, 
                np.ndarray[double, ndim=1, mode="c"] a_lock not None, np.ndarray[double, ndim=1, mode="c"] b_lock not None):
    
    cdef int n #tiempo
    cdef double* a_lock_p = <double*> a_lock.data
    cdef double[::1] a_lock_c = <double[:2]> a_lock_p
    cdef double* b_lock_p = <double*> b_lock.data
    cdef double[::1] b_lock_c = <double[:2]> b_lock_p
    cdef float* x_p = <float*>x.data
    cdef float[::1] x_c = <float[:ltimes]>x_p
    cdef float* lock_p = <float*>lock.data
    cdef float[::1] lock_c = <float[:ltimes]>lock_p
    cdef float* freq_p = <float*>freq.data
    cdef float[::1] freq_c = <float[:ltimes]>freq_p
    cdef float* vco_p = <float*>vco.data
    cdef float[::1] vco_c = <float[:ltimes]>vco_p
    cdef float* s_p = <float*>calloc(8, sizeof(float))
    cdef float[::1] s = <float[:8]>s_p
    cdef float au = 0.0
    cdef float vcoi = 0.0
    cdef float vco90 = 0.0
    cdef float tmp0 = 0.0
    cdef float tmp1 = 0.0
    cdef float tmp2 = 0.0
    # t5 = time.time()
    for n in range(ltimes):
            au = s[3] + wo*(n-1)
            vcoi = cos(au)
            vco90 = cos(au - M_PI/2.)
            #paso2, ud:
            au = x_c[n]*s[4]
            tmp0 = vcoi * au
            tmp1 = vco90 * au
            tmp2 = s[3]
            #paso4, theta0:
            s[3] = G0*s[2] + s[3]
            #paso3, uf: 
            s[2] = (G1+G2)*tmp0 -G1*s[0]+ s[2]
            #paso5, lockin:
            s[5] = b_lock_c[0] * tmp1 + b_lock_c[1] * s[1] - a_lock_c[1] * s[5]
            #paso6, agc:
            s[4] = agcgain/exp(fabs(3.*atan(0.7*fabs(s[5]))))
            #paso 7, guardar ud:
            s[0] = tmp0
            s[1] = tmp1
            #5) Cálculo de la frecuencia
            s[6] = (s[3] - tmp2 + wo)*(Fs/2./M_PI)
            s[7] = vcoi

            lock_c[n] = <float>s[5]
            freq_c[n] = <float>s[6]
            vco_c[n] = <float>s[7]
            # t6 = time.time()

    free(s_p)
    # print("tiempo de procesamiento: ", t6-t5)

    return
