# GPU-optimized Python Program
# generating 6-tuple (s,t,u,t+u,t+u-s,t-s) of squares

from math import sqrt
import pandas as pd
import numpy as np

from numba import jit
import time
import sys

@jit('void(uint64)')
def generateData(limit: np.uint64):
    rows = np.uint32(0)
    for t in np.arange(1, limit+1, dtype=np.uint64):
        for s in np.arange(1, t, dtype=np.uint64):
            for u in np.arange(1, limit+1, dtype=np.uint64):
                if t != u:
                    tt = np.uint64(t*t)
                    uu = np.uint64(u*u)
                    t_u = np.uint64(tt+uu)
                    sqr = np.uint64(sqrt(t_u))
                    if sqr*sqr == t_u:
                        ss = np.uint64(s*s)
                        t_u_s = np.uint64(t_u-ss)
                        sqr = np.uint64(sqrt(t_u_s))
                        if sqr*sqr == t_u_s:
                            t_s = np.uint64(tt-ss)
                            sqr = np.uint64(sqrt(t_s))
                            if sqr*sqr == t_s:
                                str_row = str(s)+","+str(t)+","+str(u)+","+str(ss)+","+str(tt)+","+str(uu)+","+str(t_u)+","+str(t_u_s)+","+str(t_s)
                                print(str_row)
                                

def main() -> int:
    limit = 500000

    start = time.time()
    generateData(limit)
    end = time.time()
    print("Time elapsed: {0}".format(end - start))
    
    return 0

if __name__ == '__main__':
    sys.exit(main())