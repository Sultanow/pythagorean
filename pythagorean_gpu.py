# GPU-optimized Python version of script from
# https://math.stackexchange.com/questions/3278660/solutions-to-a-system-of-three-equations-with-pythagorean-triples

from math import sqrt
import pandas as pd
import numpy as np

from numba import jit
import time
import sys

@jit('void(int32, int32[:])')
def generateData(limit: np.int32, triples: np.ndarray) -> np.int32: # -> list[list[int]]
    A=np.zeros(limit, dtype=np.int32)
    B=np.zeros(limit, dtype=np.int32)
    rows = 0
    for w in np.arange(1, limit+1, dtype=np.int32):
        count = np.int32(0)
        for a in np.arange(1, w+1, dtype=np.int32):
            sqr = np.int32(sqrt(w*w-a*a))
            if sqr*sqr == w*w-a*a:
                count+=1
                A[count]=a
                B[count]=np.int32(sqrt(w*w-a*a))
        if count>1:
            for i in np.arange(1, count+1, dtype=np.int32):
                for j in np.arange(1, count+1, dtype=np.int32):
                    if i!=j:
                        x=np.int32(A[i])
                        t=np.int32(B[i])
                        s=np.int32(A[j])
                        z=np.int32(B[j])
                        if z > x:
                            y = np.int32(sqrt(z*z-x*x))
                            if y*y == z*z-x*x:
                                #x, z, w are the important vars
                                arr = np.array([x, y, z, s, t, w], dtype=np.int32)
                                #old_size = triples.shape
                                #rows = np.int32(old_size[0])
                                #cols = np.int32(old_size[1])
                                #triples.resize((rows + 1, cols), refcheck=False)
                                triples[rows] = arr
                                rows+=1
        if w % 10000 == 0:
            print(w)
    return rows

def main() -> int:
    limit = 2000000
    triples = np.empty((limit,6), dtype=np.int32)

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