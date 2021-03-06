# Highly optimized script by Arty
# on Stack Overflow: https://stackoverflow.com/questions/70824573/optimizing-an-algorithm-with-3-inner-loops-for-searching-6-tuples-of-squares-usi
import numpy as np, numba, time, sys, math

def create_filters():
    Ks = [2 * 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19,    23 * 29 * 31 * 37 * 41]
    filts = []
    for i, K in enumerate(Ks):
        a = np.arange(K, dtype = np.uint64)
        a *= a
        a %= K
        filts.append((K, np.zeros((K,), dtype = np.uint8)))
        filts[-1][1][a] = 1
        print(f'filter {i} ratio', round(len(np.flatnonzero(filts[-1][1])) / K, 4))
    return filts

@numba.njit('void(u8, u4, u1[:], u4, u1[:])', cache = True, parallel = True,
    locals = dict(t = numba.int64, s = numba.int64, u = numba.int64))
def generateData(limit, k0, f0, k1, f1):
    def is_square(x):
        if not (f0[x % k0] and f1[x % k1]):
            return False
        root = np.uint64(math.sqrt(x) + 0.5)
        return root * root == x
    
    for t in np.arange(17835, limit + 1):
        if t % 1000 == 0:
            print('t', t)
        for s in np.arange(13572, t):
            tt = t * t
            ss = s * s
            t_s = tt - ss
            if not is_square(t_s):
                continue
            for u in np.arange(121220, limit + 1):
                if t == u:
                    continue
                uu = u * u
                t_u = tt + uu
                if not is_square(t_u):
                    continue
                t_u_s = t_u - ss
                if not is_square(t_u_s):
                    continue
                with numba.objmode():
                    print([s, t, u, ss, tt, uu, t_u, t_u_s, t_s], flush = True)

def main() -> int:
    limit = np.uint64(200000)
    
    filt = create_filters()
    
    start = time.time()
    generateData(limit, filt[0][0], filt[0][1], filt[1][0], filt[1][1])
    end = time.time()
    
    print("Time elapsed: {0}".format(end - start))
    return 0

if __name__ == '__main__':
    sys.exit(main())