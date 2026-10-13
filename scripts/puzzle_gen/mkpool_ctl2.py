import json, random, sys
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'src')
sys.path.insert(0, '.scratch')
import puzzles as P
import puzzle_defs as D
from gen_stall import attacks, REC

L = P.lib()
rng = random.Random(11)
R, O = [], []
for sp, m in L.items():
    f = m.split(',')
    mv = [x.split(';')[0] for x in f[22:26]]
    if set(mv) & REC and f[10] != 'LIFEORB' and attacks(mv):
        R.append((sp, mv))
    if f[10] == 'LIFEORB' and len(attacks(mv)) >= 2:
        O.append((sp, mv))
big = []
for rs, rm in R:
    rr = [m for m in rm if m in REC][0]
    ratk = attacks(rm)[:2]
    for os_, om in O:
        if os_ == rs:
            continue
        oa = attacks(om)[:3]
        try:
            st = D.stall_state(rs, 0.5, [rr.lower()] + [m.lower() for m in ratk], os_, [m.lower() for m in oa], bench=False)
            h = D.hits(st)
        except Exception:
            continue
        mx, mn = max(v[1] for v in h.values()), max(v[0] for v in h.values())
        if 0.6 <= mx <= 0.9 and mn >= 0.5:
            big.append((rs, [rr.lower()] + [m.lower() for m in ratk], rng.choice([0.5, 0.7, 0.9]), os_, [m.lower() for m in oa], 'LIFEORB', 'bighit'))
rng.shuffle(big)
print(len(big), 'bighit candidates', file=sys.stderr)
pool = []
for c in big[180:330]:
    st = D.stall_state(c[0], c[2], c[1], c[3], c[4])
    pool.append({'id': f'R{len(pool):03d}', 'theme': 'stall', 'src': 'ctl:bighit2', 'state': st, 'r': c[0], 'hp': c[2],
                 'rm': list(c[1]), 'o': c[3], 'om': list(c[4]), 'item': 'LIFEORB'})
json.dump(pool, open('.scratch/pool_stall2.json', 'w'))
print(len(pool))
