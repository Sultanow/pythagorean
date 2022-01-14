# GPU-optimized Python Program
# generating 6-tuple (s,t,u,t+u,t+u-s,t-s) of squares

from math import sqrt
import pandas as pd
import numpy as np

from numba import jit
import time
import sys

@jit('void(uint64, uint64[:])')
def generateData(limit: np.uint64, triplet: np.ndarray) -> np.ndarray:
    s0 = triplet[0]
    t0 = triplet[1]
    u0 = triplet[2]
    for t in np.arange(t0, limit+1, dtype=np.uint64):
        for s in np.arange(s0, t, dtype=np.uint64):
            for u in np.arange(u0, limit+1, dtype=np.uint64):
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
                                return np.array([s, t, u, ss, tt, uu, t_u, t_u_s, t_s], dtype=np.uint64)

def main() -> int:
    f = open('pythagorean_stu.txt', "a")
    f.write("s, t, u, ss, tt, uu, t_u, t_u_s, t_s")
    f.write('\n')
    f.close()

    limit = 500000
    s0 = np.uint64(153)
    t0 = np.uint64(185)
    u0 = np.uint64(672)
    while not (t0 == limit and s0 == limit-1 and u0 == limit):
        print([s0, t0, u0])
        arr = generateData(limit, np.array([s0, t0, u0], dtype=np.uint64))
        
        s = arr[0]
        t = arr[1]
        u = arr[2]
        ss = arr[3]
        tt = arr[4]
        uu = arr[5]
        t_u = arr[6]
        t_u_s = arr[7]
        t_s = arr[8]
        str_row = str(s)+","+str(t)+","+str(u)+","+str(ss)+","+str(tt)+","+str(uu)+","+str(t_u)+","+str(t_u_s)+","+str(t_s)
        f = open('pythagorean_stu.txt', "a")
        f.write(str_row)
        f.write('\n')
        f.close()
        
        s0 = np.uint64(s+1)
        t0 = np.uint64(t+1)
        u0 = np.uint64(u+1)
        
    return 0

if __name__ == '__main__':
    sys.exit(main())