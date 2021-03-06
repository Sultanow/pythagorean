# GPU-optimized Python Program
# generating 6-tuple (s,t,u,t+u,t+u-s,t-s) of squares

from math import sqrt
import pandas as pd
import numpy as np

import numba
from numba import njit
import time
import sys

@njit('void(uint64)', cache = True, parallel = True, locals = dict(
    t = numba.int64, s = numba.int64, u = numba.int64))
def generateData(limit: np.uint64):
    for t in np.arange(17835, limit+1):
        for s in np.arange(13572, t):
            for u in np.arange(121220, limit+1):
                if t != u:
                    tt = np.uint64(t*t)
                    uu = np.uint64(u*u)
                    t_u = np.uint64(tt+uu)
                    sqr = np.uint64(sqrt(t_u) + 0.5)
                    if sqr*sqr == t_u:
                        ss = np.uint64(s*s)
                        t_u_s = np.uint64(t_u-ss)
                        sqr = np.uint64(sqrt(t_u_s) + 0.5)
                        if sqr*sqr == t_u_s:
                            t_s = np.uint64(tt-ss)
                            sqr = np.uint64(sqrt(t_s) + 0.5)
                            if sqr*sqr == t_s:
                                print([s, t, u, ss, tt, uu, t_u, t_u_s, t_s])
                                

def main() -> int:
    limit = np.uint64(500000)

    start = time.time()
    generateData(limit)
    end = time.time()
    print("Time elapsed: {0}".format(end - start))
    
    return 0

if __name__ == '__main__':
    sys.exit(main())