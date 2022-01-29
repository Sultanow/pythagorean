# Run it using:
# python search_almost_Arty8.py "C:\Users\esultano\Downloads\cpp_solutions.3.536870910" > almost.txt

import sys
with open(sys.argv[1], 'r', encoding = 'utf-8') as f:
    m = {}
    a = []
    for line in f:
        if not line.strip():
            continue
        ns = list(map(int, line.split(',')))
        a.append(ns)
        key = tuple(ns[1:])
        if key not in m:
            m[key] = []
        m[key].append(ns)
    for ns in a:
        for e2 in m.get(tuple(ns[:-1]), []):
            print(', '.join(str(e) for e in (e2 + ns[-1:])))