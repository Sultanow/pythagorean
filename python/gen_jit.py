# GPU-optimized Python version of script from
# https://math.stackexchange.com/questions/3278660/solutions-to-a-system-of-three-equations-with-pythagorean-triples

from math import sqrt
import pandas as pd
import numpy as np

from numba import jit
import time
import sys

@jit('void(uint64, uint64[:])')
def generateData(limit: np.uint64, triples: np.ndarray) -> np.uint32: # -> list[list[int]]
    A=np.zeros(limit, dtype=np.uint64)
    B=np.zeros(limit, dtype=np.uint64)
    rows = np.uint32(0)
    for w in np.arange(1, limit+1, dtype=np.uint64):
        count = np.uint32(0)
        for a in np.arange(1, w+1, dtype=np.uint64):
            sqr = np.uint64(sqrt(w*w-a*a))
            if sqr*sqr == w*w-a*a:
                count+=1
                A[count]=a
                B[count]=np.uint64(sqrt(w*w-a*a))
        if count>1:
            for i in np.arange(1, count+1, dtype=np.uint64):
                for j in np.arange(1, count+1, dtype=np.uint64):
                    if i!=j:
                        x=np.uint64(A[i])
                        t=np.uint64(B[i])
                        s=np.uint64(A[j])
                        z=np.uint64(B[j])
                        if z > x:
                            y = np.uint64(sqrt(z*z-x*x))
                            if y*y == z*z-x*x:
                                #x, z, w are the important vars
                                arr = np.array([x, y, z, s, t, w], dtype=np.uint64)
                                #old_size = triples.shape
                                #rows = np.uint64(old_size[0])
                                #cols = np.uint64(old_size[1])
                                #triples.resize((rows + 1, cols), refcheck=False)
                                triples[rows] = arr
                                rows+=1
        if w % 10000 == 0:
            print(w)
    return rows

def main() -> int:
    limit = 6000000
    triples = np.empty((limit,6), dtype=np.uint64)

    start = time.time()
    rows = generateData(limit, triples)
    end = time.time()
    print("Time elapsed: {0}".format(end - start))
    print(rows)
    df = pd.DataFrame(triples[0:rows], columns = ['x', 'y', 'z', 's', 't', 'w'])
    df.to_csv('pythagorean_' + str(limit) + '.csv', index=False)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())