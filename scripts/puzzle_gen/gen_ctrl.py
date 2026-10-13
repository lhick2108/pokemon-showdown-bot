import json, random, sys
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'src')
import puzzles as P
import puzzle_defs as D
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, '.scratch')
from gen_stall import attacks, REC


def job(c):
    sys.path.insert(0, 'src')
    r, rm, hp, o, om, item, kind = c
    ours = [P.mon(r, hp=hp, moves=rm, item='LEFTOVERS'), P.mon('TERRAKION', hp=0.1)]
    theirs = [P.mon(o, moves=om, item=item)]
    st = P.state(P.side(ours), P.side(theirs))
    return c, st, P.rollout_table(st, 60)


if __name__ == '__main__':
    L = P.lib()
    rng = random.Random(11)
    cands = []
    # (1) no-LO variants of the decisive base matchups + random ones
    bases = json.load(open('.scratch/stall2.json'))
    for d in bases:
        r, rm, hp, o, om, obk = d['c']
        if obk == 'none':
            for item in ('LEFTOVERS', 'CHOICESPECS'):
                cands.append((r, rm, hp, o, om, item, 'nolo'))
    # (2) big hitters with LO
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
                st = D.stall_state(rs, 0.5, [rr.lower()] + [m.lower() for m in ratk], os_, [m.lower() for m in oa], bench=())
                h = D.hits(st)
            except Exception:
                continue
            mx, mn = max(v[1] for v in h.values()), max(v[0] for v in h.values())
            if 0.6 <= mx <= 0.9 and mn >= 0.5:
                big.append((rs, [rr.lower()] + [m.lower() for m in ratk], rng.choice([0.5, 0.7, 0.9]), os_, [m.lower() for m in oa], 'LIFEORB', 'bighit'))
    rng.shuffle(big)
    cands = big[int(sys.argv[2]):int(sys.argv[2]) + int(sys.argv[1])]
    print(len(cands), file=sys.stderr)
    out = []
    with ProcessPoolExecutor(5) as pool:
        for c, st, tb in pool.map(job, cands, chunksize=1):
            V = {a: max(v.values()) for a, v in tb.items()}
            rec = [a for a in V if a in {m.lower() for m in P.RECOVERY}]
            vr = max(V[a] for a in rec) if rec else None
            vo = max(V[a] for a in V if a not in rec)
            out.append({'c': c, 'state': st, 'tb': tb})
            print(f"{(vo - vr):+.2f} rec={vr:.2f} oth={vo:.2f}", c[0], c[2], c[3], c[5], c[6], {a: round(v, 2) for a, v in V.items()}, flush=True)
    json.dump(out, open('.scratch/ctrl_b.json', 'w'))
