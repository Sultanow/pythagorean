numba = None
import numba

import json, multiprocessing, time, timeit, os, math, numpy as np

if numba is None:
    class NumbaInt:
        def __getitem__(self, key):
            return None
    class numba:
        uint8, uint16, uint32, int64, uint64 = [NumbaInt() for i in range(5)]
        def njit(*pargs, **nargs):
            return lambda f: f
        def prange(*pargs):
            return range(*pargs)
        class types:
            class Tuple:
                def __init__(self, *nargs, **pargs):
                    pass
                def __call__(self, *nargs, **pargs):
                    pass
        class objmode:
            def __init__(self, *pargs, **nargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, ext, exv, tb):
                pass

@numba.njit(cache = True, parallel = True)
def create_filters():
    Ks = [np.uint32(e) for e in [2 * 2 * 2 * 3 * 3 * 5 * 7 * 11 * 13,    17 * 19 * 23 * 29 * 31 * 37]]
    filts = []
    for i in range(len(Ks)):
        K = Ks[i]
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

@numba.njit(
    #'void(i8, i8, u4, u1[:], u4[:], u2[:, :, :], u4, u1[:])',
    numba.types.Tuple([numba.uint64[:], numba.uint32[:]])(
        numba.int64, numba.int64, numba.uint32, numba.uint8[:],
        numba.uint32[:], numba.uint16[:, :, :], numba.uint32, numba.uint8[:]),
    cache = True, parallel = True,
    locals = dict(x = numba.uint64, Atpos = numba.uint64, Btpos = numba.uint64, bpos = numba.uint64))
def create_table(limit, cpu_count, k0, f0, fi0, fc0, k1, f1):
    print('Computing tables...')
    
    def gen_squares_candidates_A(cnt, lim, off, t, K, f, fi, fc):
        mark = np.zeros((np.int64(K),), dtype = np.uint8)
        while True:
            start_s = np.int64((np.int64(off) + np.int64(t) ** 2) % K)
            tK = np.uint32(np.int64(t) % np.int64(K))
            if mark[tK]:
                return np.zeros((0,), dtype = np.uint32)
            mark[tK] = 1
            if f[start_s]:
                break
            t += 1
        j = np.int64(np.searchsorted(fi, start_s))
        assert fi[j] == start_s
        r = np.zeros((np.int64(cnt),), dtype = np.uint32)
        r[0] = t
        rpos = np.int64(1)
        tK = np.uint32(np.int64(t) % np.int64(K))
        while True:
            j, dt = fc[j, tK]
            t += dt
            tK += dt
            if tK >= np.uint32(K):
                tK -= np.uint32(K)
            if t >= lim:
                return r[:rpos]
            if np.int64(rpos) >= np.int64(r.shape[0]):
                r = np.concatenate((r, np.zeros_like(r)), axis = 0)
            assert rpos < len(r)
            r[rpos] = t
            rpos += 1
    
    def gen_squares(cnt, lim, off, t, K, f, fi, fc, k1, f1):
        def is_square(x):
            assert x >= 0
            if not f1[np.int64(x) % np.uint32(k1)]:
                return False
            root = np.uint64(math.sqrt(np.float64(x)) + 0.5)
            return root * root == x
        rA = gen_squares_candidates_A(cnt, lim, off, t, K, f, fi, fc)
        r = np.zeros_like(rA)
        rpos = np.int64(0)
        for t in rA:
            if not is_square(np.int64(off) + np.int64(t) ** 2):
                continue
            assert np.int64(rpos) < np.int64(r.shape[0])
            r[rpos] = t
            rpos += 1
        return r[:rpos]
    
    with numba.objmode(gtb = 'f8'):
        gtb = time.time()
    
    search_start = 2
    cnt_limit = max(1 << 4, round(pow(limit, 0.66)))
    
    nblocks2 = cpu_count * 8
    nblocks = nblocks2 * 64
    block = (limit + nblocks - 1) // nblocks
    
    At = np.zeros((limit + 1,), dtype = np.uint64)
    Bt = np.zeros((0,), dtype = np.uint32)
    Atpos, Btpos = search_start + 1, 0
    
    with numba.objmode(tb = 'f8'):
        tb = time.time()
    for iMblock in range(0, nblocks, nblocks2):
        cur_blocks = min(nblocks, iMblock + nblocks2) - iMblock
        As = np.zeros((cur_blocks, block), dtype = np.uint64)
        As_size = np.zeros((cur_blocks,), dtype = np.uint64)
        Bs = np.zeros((cur_blocks, 1 << 16,), dtype = np.uint32)
        Bs_size = np.zeros((cur_blocks,), dtype = np.uint64)
        for iblock in numba.prange(cur_blocks):
            iblock0 = iMblock + iblock
            begin, end = max(search_start, iblock0 * block), min(limit, (iblock0 + 1) * block)
            begin = min(begin, end)
            #a = np.zeros((block,), dtype = np.uint64)
            #b = np.zeros((1 << 10,), dtype = np.uint32)
            bpos = 0
            for ix, x in enumerate(range(begin, end)):
                s = gen_squares(cnt_limit, limit, x ** 2, search_start, k0, f0, fi0, fc0, k1, f1)
                assert not (np.int64(bpos) + np.int64(s.shape[0]) > np.int64(Bs[iblock].shape[0]))
                #while np.int64(bpos) + np.int64(s.shape[0]) > np.int64(b.shape[0]):
                #    b = np.concatenate((b, np.zeros_like(b)), axis = 0)
                bpos_end = bpos + s.shape[0]
                Bs[iblock, bpos : bpos_end] = s
                As[iblock, ix] = bpos_end
                bpos = bpos_end
            As_size[iblock] = end - begin
            Bs_size[iblock] = bpos
        for iblock, (cA, cB) in enumerate(zip(As, Bs)):
            cA = cA[:As_size[iblock]]
            cB = cB[:Bs_size[iblock]]
            assert Atpos + cA.shape[0] <= At.shape[0]
            prevA = At[Atpos - 1]
            for e in cA:
                At[Atpos] = prevA + e
                Atpos += 1
            #while np.int64(Btpos) + np.int64(cB.shape[0]) > np.int64(Bt.shape[0]):
                #Bt = np.concatenate((Bt, np.zeros_like(Bt)), axis = 0)
                #Bt = np.concatenate((Bt, np.zeros(Bt.shape, dtype = np.uint32)), axis = 0)
            #assert np.int64(Btpos) + np.int64(cB.shape[0]) <= np.int64(Bt.shape[0])
            #assert cB.shape[0] > 0
            #Bt[Btpos : Btpos + cB.shape[0]] = cB
            Bt = np.concatenate((Bt, cB))
            #Btpos += cB.shape[0]
            #assert At[Atpos - 1] == Btpos
            assert At[Atpos - 1] == Bt.shape[0]
        with numba.objmode(tim = 'f8'):
            tim = max(0.001, round(time.time() - tb, 3))
        print(f'{str(min(limit, (iMblock + cur_blocks) * block) >> 10).rjust(len(str(limit >> 10)))}/{limit >> 10} K, ELA',
            round(tim / 60.0, 1), 'min, ETA', round((nblocks - (iMblock + cur_blocks)) * (tim / (iMblock + cur_blocks)) / 60.0, 1), 'min')
    
    assert Atpos == At.shape[0]
    
    with numba.objmode(gtb = 'f8'):
        gtb = time.time() - gtb
    
    print(f'Tables sizes: A {Atpos}, B {Bt.shape[0]}')
    print('Time elapsed computing tables:', round(gtb / 60.0, 1), 'min')
    
    return At, Bt
    
def table_create_load(limit, *pargs):
    fnameA = f'right_triangles_table.A.{limit}'
    fnameB = f'right_triangles_table.B.{limit}'
    if not os.path.exists(fnameA) or not os.path.exists(fnameB):
        A, B = create_table(limit, *pargs)
        with open(fnameA, 'wb') as f:
            f.write(A.tobytes())
        with open(fnameB, 'wb') as f:
            f.write(B.tobytes())
        del A, B
    with open(fnameA, 'rb') as f:
        A = np.copy(np.frombuffer(f.read(), dtype = np.uint64))
        assert A.shape[0] == limit + 1, (fnameA, A.shape[0], limit + 1)
    with open(fnameB, 'rb') as f:
        B = np.copy(np.frombuffer(f.read(), dtype = np.uint32))
        assert A[-1] == B.shape[0], (fnameB, A[-1], B.shape[0])
    print(f'Table A size {A.shape[0]}, B size {B.shape[0]}')
    return A, B

def find_solutions(tA, tB, stu):
    def is_square(x):
        root = np.uint64(math.sqrt(np.float64(x)) + 0.5)
        return bool(root * root == x), int(root)
    
    assert tA[-1] == tB.shape[0]
    
    fname = f'stu_solutions.{tA.shape[0] - 1}'
    with open(fname, 'w', encoding = 'utf-8') as fout:
        for s, t, u in stu:
            s, t, u = map(int, (s, t, u))
            r = {'stu': [s, t, u]}
            if s + 1 >= tA.shape[0]:
                r['err'] = f's {s} exceeds table A len {tA.shape[0]}'
            elif t + 1 >= tA.shape[0]:
                r['err'] = f't {t} exceeds table A len {tA.shape[0]}'
            else:
                r['res'] = []
                bs = tB[tA[s] : tA[s + 1]]
                ts = tB[tA[t] : tA[t + 1]]
                for w in np.intersect1d(bs, ts):
                    w = int(w)
                    x2 = s ** 2 + w ** 2
                    y2 = t ** 2 + w ** 2
                    x_isq, x = is_square(x2)
                    assert x_isq, (s, t, u, w, x2)
                    y_isq, y = is_square(y2)
                    assert y_isq, (s, t, u, w, x2, y2)
                    z2 = u ** 2 + y2
                    z_isq, z = is_square(z2)
                    r['res'].append({
                        'w': w,
                        'x': x,
                        'y': y,
                        'z2': z2,
                        'z2_is_square': z_isq,
                        'z': z if z_isq else math.sqrt(z2),
                    })
            fout.write(json.dumps(r, ensure_ascii = False) + '\n')
    
    print(f'STU solutions written to {fname}')

def solve(limit):
    import requests
    
    filts = create_filters()
    fc0 = filter_chain_create_load(filts[0][0], filts[0][2])
    
    tA, tB = table_create_load(limit, multiprocessing.cpu_count(),
        filts[0][0], filts[0][1], filts[0][2], fc0, filts[1][0], filts[1][1])
    
    # https://github.com/Sultanow/pythagorean/blob/main/data/pythagorean_stu_Arty_.txt?raw=true
    ifname = 'pythagorean_stu_Arty_.txt'
    iurl = f'https://github.com/Sultanow/pythagorean/blob/main/data/{ifname}?raw=true'
    if not os.path.exists(ifname):
        print(f'Downloading: {iurl}')
        data = requests.get(iurl).content
        with open(ifname, 'wb') as f:
            f.write(data)
    stu = []
    with open(ifname, 'r', encoding = 'utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            if 'elapsed' in line:
                continue
            s, t, u, *_ = eval(f'[{line}]')
            stu.append([s, t, u])
    print(f'Read {len(stu)} s/t/u tuples')
    find_solutions(tA, tB, stu)
    
def main():
    limit = 100_000
    solve(limit)

if __name__ == '__main__':
    main()