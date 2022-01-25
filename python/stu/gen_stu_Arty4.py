numba = None
import numba

import multiprocessing, time, timeit, os, math, numpy as np

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
    Ks = [np.uint32(e) for e in [2 * 2 * 2 * 3 * 3 * 5 * 7 * 11 * 13,    17 * 19 * 23 * 29 * 31 * 37]]
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

@numba.njit('u4[:](i8, i8, i8, u4, u4, u1[:], u4[:], u2[:, :, :])',
    cache = True, #inline = 'always',
    locals = dict(j = numba.uint32, tK = numba.uint32, rpos = numba.uint64))
def gen_squares_candidates_A(cnt, lim, off, t, K, f, fi, fc):
    mark = np.zeros((K,), dtype = np.uint8)
    while True:
        start_s = (off + np.int64(t) ** 2) % K
        tK = np.uint32(t % K)
        if mark[tK]:
            return np.zeros((0,), dtype = np.uint32)
        mark[tK] = 1
        if f[start_s]:
            break
        t += 1
    j = np.searchsorted(fi, start_s)
    assert fi[j] == start_s
    r = np.zeros((np.int64(cnt),), dtype = np.uint32)
    r[0] = t
    rpos = 1
    tK = np.uint32(t % K)
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

@numba.njit('u4[:](i8, i8, i8, u4, u4, u1[:], u4[:], u2[:, :, :])',
    cache = True, #inline = 'always',
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

@numba.njit('u4[:](i8, i8, i8, u4, u4, u1[:], u4[:], u2[:, :, :], u4, u1[:])',
    cache = True, #inline = 'always',
    locals = dict(rpos = numba.uint64))
def gen_squares(cnt, lim, off, t, K, f, fi, fc, k1, f1):
    def is_square(x):
        assert x >= 0
        if not f1[x % k1]:
            return False
        root = np.uint64(math.sqrt(np.float64(x)) + 0.5)
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

@numba.njit('u8[:, :, :](i8, i8, i8, u4, u1[:], u4[:], u2[:, :, :], u4, u1[:])', cache = True, parallel = True,
    locals = dict(t = numba.int64, s = numba.int64, u = numba.int64, i = numba.int64))
def generateData(cpu_count, L, limit, k0, f0, fi0, fc0, k1, f1):
    def is_square(x):
        if not (f0[x % k0] and f1[x % k1]):
            return False
        root = np.uint64(math.sqrt(np.float64(x)) + 0.5)
        return root * root == x
    
    cnt_limit = pow(limit, 0.66)
    start = 2
    
    A = np.zeros((limit - start, 2, 1), dtype = np.uint64)
    A[:, 0, 0] = np.arange(start, limit)
    A[:, 1, 0] = A[:, 0, 0] ** 2
    
    nblocks = cpu_count * 64
    
    while True:
        NAs = [np.zeros((1 << 10, 2, A.shape[-1] + 1), dtype = np.uint64) for i in range(nblocks)]
        na_poss = [np.int64(0) for i in range(nblocks)]
        block = (A.shape[0] + nblocks - 1) // nblocks
        
        for iMblock in range(0, nblocks, cpu_count):
            for iblock in numba.prange(iMblock, min(iMblock + cpu_count, nblocks)):
                for ie in range(iblock * block, min(A.shape[0], (iblock + 1) * block)):
                    e = A[ie]
                    #if ie & ((1 << 13) - 1) == 0:
                    #    print(f'l {A.shape[-1] + 1}/{L}', f'i {str(ie >> 10).rjust(5)}/{A.shape[0] >> 10} K')
                    na_pos_start = na_poss[iblock]
                    for e2 in gen_squares(cnt_limit, limit, e[1, -1], start, k0, f0, fi0, fc0, k1, f1):
                        v = np.uint64(e[1, -1] + np.uint64(e2) ** 2)
                        for e3 in e[1, :-1]:
                            if not is_square(v - e3):
                                break
                        else:
                            if na_poss[iblock] >= NAs[iblock].shape[0]:
                                NAs[iblock] = np.concatenate((NAs[iblock], np.zeros((NAs[iblock].shape[0] // 2,) +
                                    NAs[iblock].shape[1:], dtype = np.uint64)), axis = 0)
                            NAs[iblock][na_poss[iblock], 0, -1] = np.uint64(e2)
                            NAs[iblock][na_poss[iblock], 1, -1] = v
                            na_poss[iblock] += 1
                    for j in range(na_pos_start, na_poss[iblock]):
                        NAs[iblock][j, :, : NAs[iblock].shape[-1] - 1] = e
            print(f'l {A.shape[-1] + 1}/{L}', f'i {str(min(A.shape[0], block * min(nblocks, iMblock + cpu_count)) >> 10).rjust(5)}/{A.shape[0] >> 10} K')
        print()
        
        A = NAs[0][:na_poss[0]]
        for i in range(1, nblocks):
            A = np.concatenate((A, NAs[i][:na_poss[i]]), axis = 0)
        if A.shape[-1] >= L:
            break

    print(A.shape[0], 'solutions')
    return A

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
    
    limit = 3_000_000
    L = 3
    
    filts = create_filters()
    fc0 = filter_chain_create_load(filts[0][0], filts[0][2])
    
    start = time.time()
    res = generateData(multiprocessing.cpu_count(), L, limit,
        filts[0][0], filts[0][1], filts[0][2], fc0, filts[1][0], filts[1][1])
    end = time.time()
    
    fname = f'solutions.{L}.{limit}'
    with open(fname, 'w', encoding = 'utf-8') as f:
        for e in res:
            f.write(str([int(e2) for e2 in e[0]] + [int(e2) for e2 in e[1]]) + '\n')
    print(f"Written file '{fname}'")
    
    print("Time elapsed: {0}".format(round(end - start, 3)), 'sec')
    
if __name__ == '__main__':
    main()
