import pandas as pd
import sys
from z3 import Ints, solve

# for z3 consider:
# https://stackoverflow.com/questions/61170977/z3-solver-installed-but-i-cant-import-anything
# https://github.com/Z3Prover/z3/wiki/Using-Z3Py-on-Windows
def main() -> int:
    df = pd.read_csv('pythagorean_stu.txt', header=None)
    
    tuples = df.to_numpy()

    x, y, z, w = Ints('x y z w')
    for row in tuples:
        s=int(row[3])
        t=int(row[4])
        u=int(row[5])
        solve(x*x-w*w==s, y*y-w*w==t, z*z-y*y==u, w!=0)

    return 0

if __name__ == '__main__':
    sys.exit(main())