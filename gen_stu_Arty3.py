numba = None
import numba

import time, timeit, os, math, numpy as np

if numba is None:
    class numba:
        uint32, uint64 = [None] * 2
        def njit(*pargs, **nargs):
            return lambda f: f
        def prange(*pargs):
            return range(*pargs)
        class objmode:
            def __enter__(self):
                return self
            def __exit__(self, ext, exv, tb):
                pass

@numba.njit(cache = True, parallel = True)
def create_filters():
    Ks = [np.uint32(e) for e in [2 * 2 * 3 * 3 * 5 * 7 * 11 * 13,    17 * 19 * 23 * 29 * 31 * 37]]
    filts = []
    for i, K in enumerate(Ks):
        filt = np.zeros((K,), dtype = np.uint8)
        block = 1 << 25
        nblocks = (K + block - 1) // block
        for j0 in numba.prange(nblocks):
            j = j0 * block
            a = np.arange(j, min(j + block, K)).astype(np.uint64)
            a *= a; a %= K
            filt[a] = 1
        idxs = np.flatnonzero(filt).astype(np.uint32)
        filts.append((K, filt, idxs))
        print(f'filter {i} ratio', round(len(filts[-1][2]) / K, 4))
    return filts

@numba.njit('u2[:, :, :](u4, u4[:])', cache = True, parallel = True, locals = dict(
    t = numba.uint32, s = numba.uint32, i = numba.uint32, j = numba.uint32))
def filter_chain(K, ix):
    assert len(ix) < (1 << 16)
    ix_rev = np.full((K,), len(ix), dtype = np.uint16)
    for i, e in enumerate(ix):
        ix_rev[e] = i
    r = np.zeros((len(ix), K, 2), dtype = np.uint16)
    
    print('filter chain pre-computing...')
    
    for i in numba.prange(K):
        if i % 5000 == 0 or i + 1 >= K:
            with numba.objmode():
                print(f'{i}/{K}, ', end = '', flush = True)
        for j, x in enumerate(ix):
            t, s = i, x
            while True:
                s += 2 * t + 1; s %= K
                t += 1
                if ix_rev[s] < len(ix):
                    assert t - i < (1 << 16)
                    assert t - i < K
                    r[j, i, 0] = ix_rev[s]
                    r[j, i, 1] = np.uint16(t - i)
                    break
    
    print()
    
    return r

def filter_chain_create_load(K, ix):
    fname = f'filter_chain.{K}'
    if not os.path.exists(fname):
        r = filter_chain(K, ix)
        with open(fname, 'wb') as f:
            f.write(r.tobytes())
    with open(fname, 'rb') as f:
        return np.copy(np.frombuffer(f.read(), dtype = np.uint16).reshape(len(ix), K, 2))

@numba.njit('u4[:](i8, i8, i8, u4, u4, u1[:], u4[:], u2[:, :, :])', cache = True,
    locals = dict(j = numba.uint32, tK = numba.uint32, rpos = numba.uint64))
def gen_squares_candidates_A(cnt, lim, off, t, K, f, fi, fc):
    mark = np.zeros((K,), dtype = np.uint8)
    while True:
        start_s = (off + np.int64(t) ** 2) % K
        tK = t % K
        if mark[tK]:
            return np.zeros((0,), dtype = np.uint32)
        mark[tK] = 1
        if f[start_s]:
            break
        t += 1
    j = np.searchsorted(fi, start_s)
    assert fi[j] == start_s
    r = np.zeros((cnt,), dtype = np.uint32)
    r[0] = t
    rpos = 1
    tK = t % K
    while True:
        j, dt = fc[j, tK]
        t += dt
        tK += dt
        if tK >= K:
            tK -= K
        if t >= lim:
            return r[:rpos]
        if rpos >= len(r):
            r = np.concatenate((r, np.zeros((len(r),), dtype = np.uint32)))
        assert rpos < len(r)
        r[rpos] = t
        rpos += 1

@numba.njit('u4[:](i8, i8, i8, u4, u4, u1[:], u4[:], u2[:, :, :])', cache = True,
    locals = dict(rpos = numba.uint64, tK = numba.uint32))
def gen_squares_candidates_A_slow(cnt, lim, off, t, K, f, fi, fc):
    mark = np.zeros((K,), dtype = np.uint8)
    r = np.zeros((cnt,), dtype = np.uint32)
    rpos = 0
    found = False
    while True:
        tK = t % K
        if not found:
            if mark[tK]:
                return np.zeros((0,), dtype = np.uint32)
            mark[tK] = 1
        if f[(off + np.int64(t) ** 2) % K]:
            if t >= lim:
                return r[:rpos]
            if rpos >= len(r):
                r = np.concatenate((r, np.zeros((len(r),), dtype = np.uint32)))
            assert rpos < len(r)
            r[rpos] = t
            rpos += 1
            found = True
        t += 1

@numba.njit('u4[:](i8, i8, i8, u4, u4, u1[:], u4[:], u2[:, :, :], u4, u1[:])', cache = True,
    locals = dict(rpos = numba.uint64))
def gen_squares(cnt, lim, off, t, K, f, fi, fc, k1, f1):
    def is_square(x):
        assert x >= 0
        if not f1[x % k1]:
            return False
        root = np.uint64(math.sqrt(x) + 0.5)
        return root * root == x
    rA = gen_squares_candidates_A(cnt, lim, off, t, K, f, fi, fc)
    r = np.zeros((len(rA),), dtype = np.uint32)
    rpos = 0
    for t in rA:
        if not is_square(off + np.int64(t) ** 2):
            continue
        if rpos >= len(r):
            r = np.concatenate((r, np.zeros((len(r),), dtype = np.uint32)))
        r[rpos] = t
        rpos += 1
    return r[:rpos]

@numba.njit('void(i8, u4, u1[:], u4[:], u2[:, :, :], u4, u1[:])', cache = True, parallel = True,
    locals = dict(t = numba.int64, s = numba.int64, u = numba.int64, i = numba.int64))
def generateData(limit, k0, f0, fi0, fc0, k1, f1):
    def is_square(x):
        if not (f0[x % k0] and f1[x % k1]):
            return False
        root = np.uint64(math.sqrt(x) + 0.5)
        return root * root == x
    
    cnt_limit = pow(limit, 0.66)
    
    tsl0 = [[(np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0))] for i in range(limit + 1)]
    
    for s in numba.prange(2, limit + 1):
        if s % 20_000 == 0 or s + 1 >= limit + 1:
            print(f's {str(s).rjust(7)}/{limit}')
        ss = s * s
        t_sqrs = gen_squares(cnt_limit, limit + 1, -np.int64(ss), s + 1, k0, f0, fi0, fc0, k1, f1)
        for t in t_sqrs:
            tt = t * t
            t_s = tt - ss
            if 0 and not is_square(t_s):
                print('non-square t_s', t_s, 't', t, 's', s)
                assert False
            tsl0[s].append((np.int64(t), np.int64(s), np.int64(tt), np.int64(ss), np.int64(t_s)))
    
    tsl = []
    for e in tsl0:
        tsl.extend(e[1:])
    
    tl = {}
    for e in tsl:
        tl[e[2]] = e[0]
    tl = sorted(tl.items())
    
    len_tl = len(tl)
    tul0 = [[(np.int64(0), np.int64(0), np.int64(0), np.int64(0))] for i in range(len_tl)]
    
    for i in numba.prange(len_tl):
        tt, t = tl[i]
        if i % 20_000 == 0 or i + 1 >= len(tl):
            len_tl = len_tl
            print(f't {str(i).rjust(7)}/{len_tl}')
        u_sqrs = gen_squares(cnt_limit, limit + 1, np.int64(tt), 2, k0, f0, fi0, fc0, k1, f1)
        for u in u_sqrs:
            uu = u * u
            t_u = tt + uu
            if 0 and not is_square(t_u):
                print('non-square t_u', t_u, 't', t, 'u', u)
                assert False
            tul0[i].append((np.int64(t), np.int64(u), np.int64(uu), np.int64(t_u)))
    
    tul = []
    for e in tul0:
        tul.extend(e[1:])
    
    tul = sorted(tul)
    
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

def test():        
    filts = create_filters()
    fc0 = filter_chain_create_load(filts[0][0], filts[0][2])
    
    not_found = [f(1 << 22, 1 << 19, 1234, np.uint32(5678), np.uint32(filts[0][0]), *filts[0][1:3], fc0)
        for f in [
            gen_squares_candidates_A,
            gen_squares_candidates_A_slow,
        ]]
    assert len(not_found[0]) == 0
    assert len(not_found[1]) == 0

    print([timeit.timeit(lambda: f(1 << 22, 1 << 28, 123, np.uint32(456), np.uint32(filts[0][0]), *filts[0][1:3], fc0), number = 1)
        for f in [
            gen_squares_candidates_A,
            gen_squares_candidates_A_slow,
        ]])
    
    for i, (a, b) in enumerate(zip(*[f(1 << 22, 1 << 28, 123, np.uint32(456), np.uint32(filts[0][0]), *filts[0][1:3], fc0)
            for f in [
                gen_squares_candidates_A,
                gen_squares_candidates_A_slow,
            ]])):
        if i % 100_000 == 0:
            print(a, end = ' ', flush = True)
        assert a == b

def main():
    #test(); return
    
    limit = np.uint64(1000_000)
    
    filts = create_filters()
    fc0 = filter_chain_create_load(filts[0][0], filts[0][2])
    
    start = time.time()
    generateData(limit, filts[0][0], filts[0][1], filts[0][2], fc0, filts[1][0], filts[1][1])
    end = time.time()
    
    print("Time elapsed: {0}".format(round(end - start, 3)))

if __name__ == '__main__':
    main()