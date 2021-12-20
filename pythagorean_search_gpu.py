# GPU-optimized Python version of script from
# https://math.stackexchange.com/questions/3278660/solutions-to-a-system-of-three-equations-with-pythagorean-triples

from math import sqrt
import pandas as pd
import numpy as np

from numba import jit
import sys

@jit('void(uint64, uint64[:])')
def validateW4Triple(w:np.uint64, triple:'np.ndarray[3,np.uint64]') -> np.int8:
    x = np.uint64(triple[0])
    y = np.uint64(triple[1])
    z = np.uint64(triple[2])
    sqr = np.uint64(x*x-w*w)
    sqt = np.uint64(sqrt(sqr))
    if sqt*sqt != sqr:
        return np.int8(1)
    sqr = np.uint64(y*y-w*w)
    sqt = np.uint64(sqrt(sqr))
    if sqt*sqt != sqr:
        return np.int8(1)
    sqr = np.uint64(z*z-w*w)
    sqt = np.uint64(sqrt(sqr))
    if sqt*sqt != sqr:
        return np.int8(1)
    print([w, x, y, z])
    return np.int8(0)

def main() -> int:
    df = pd.read_csv('pythagorean_2000000.csv')
    
    triples = df[['x', 'z', 'w']].values
    rows = triples.shape[0]
    for row in range(0, rows):
        triple = triples[row]
        x = triple[0]
        for w in range(1, x):
            validateW4Triple(w, triple)

    return 0

if __name__ == '__main__':
    sys.exit(main())