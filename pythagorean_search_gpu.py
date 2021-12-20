# GPU-optimized Python version of script from
# https://math.stackexchange.com/questions/3278660/solutions-to-a-system-of-three-equations-with-pythagorean-triples

from math import sqrt
import pandas as pd
import numpy as np

from numba import jit
import sys

# old signature was
# validateW4Triple(w:np.uint64, triple:'np.ndarray[3,np.uint64]') -> np.int8:

@jit('void(uint64[:])')
def validateW4Triple(triples:np.ndarray):
    rows = np.uint32(triples.shape[0])
    for row in np.arange(0, rows, dtype=np.uint64):
        x = np.uint64(triples[row][0])
        y = np.uint64(triples[row][1])
        z = np.uint64(triples[row][2])
        for w in np.arange(1, x, dtype=np.uint64):
            sqr = np.uint64(x*x-w*w)
            sqt = np.uint64(sqrt(sqr))
            if sqt*sqt != sqr:
                continue
            sqr = np.uint64(y*y-w*w)
            sqt = np.uint64(sqrt(sqr))
            if sqt*sqt != sqr:
                continue
            sqr = np.uint64(z*z-w*w)
            sqt = np.uint64(sqrt(sqr))
            if sqt*sqt != sqr:
                continue
            print([w, x, y, z])

        if row % 1000 == 0:
            print(row)

def main() -> int:
    df = pd.read_csv('pythagorean_2000000.csv')
    
    triples = df[['x', 'z', 'w']].to_numpy()
    validateW4Triple(triples)   

    return 0

if __name__ == '__main__':
    sys.exit(main())