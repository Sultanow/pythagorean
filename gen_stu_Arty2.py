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
        #print(f'filter {i} ratio', round(len(np.flatnonzero(filts[-1][1])) / K, 4))
    return filts

@numba.njit('void(i8, u4, u1[:], u4, u1[:])', cache = False, parallel = True,
    locals = dict(t = numba.int64, s = numba.int64, u = numba.int64, i = numba.int64))
def generateData(limit, k0, f0, k1, f1):
    def is_square(x):
        if not (f0[x % k0] and f1[x % k1]):
            return False
        root = np.uint64(math.sqrt(x) + 0.5)
        return root * root == x
        
    tsl = []
    
    for t in range(2, limit + 1):
        #if t % 5000 == 0:
        #    print('t', t)
        for s in np.arange(2, t):
            tt = t * t
            ss = s * s
            t_s = tt - ss
            if not is_square(t_s):
                continue
            tsl.append([t, s, tt, ss, t_s])
    
    tl = sorted({e[2]: e[0] for e in tsl}.items())
    #tl = {}
    #for e in tsl:
    #    tl[e[2]] = e[0]
    #tl = sorted(tl.items())
    
    tul = []
    
    for u in range(2, limit + 1):
        #if u % 5000 == 0:
        #    print('u', u)
        uu = u * u
        for tt, t in tl:
            t_u = tt + uu
            if not is_square(t_u):
                continue
            tul.append((t, u, uu, t_u))
    
    tul = sorted(tul, key = lambda e: e[:2])
    
    tul_idx = {}
    for i, e in enumerate(tul):
        if e[0] not in tul_idx:
            tul_idx[e[0]] = i
    
    for t, s, tt, ss, t_s in tsl:
        if t not in tul_idx:
            continue
        for i in range(tul_idx[t], 1 << 60):
            if i >= len(tul) or tul[i][0] != t:
                break
            u, uu, t_u = tul[i][1:]
            t_u_s = t_u - ss
            if not is_square(t_u_s):
                continue
            with numba.objmode():
                print([s, t, u, ss, tt, uu, t_u, t_u_s, t_s], flush = True)

def main() -> int:
    limit = np.uint64(400_000)

    filt = create_filters()
    
    start = time.time()
    generateData(limit, filt[0][0], filt[0][1], filt[1][0], filt[1][1])
    end = time.time()
    
    print("Time elapsed: {0}".format(round(end - start, 3)))
    return 0

if __name__ == '__main__':
    sys.exit(main())