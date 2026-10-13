import json, sys, itertools
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'src')
import puzzles as P
from concurrent.futures import ProcessPoolExecutor

BASES = [
    ('TOXAPEX', ['recover', 'liquidation'], 'LUMINEON', ['alluringvoice', 'hydropump', 'icebeam']),
    ('DUDUNSPARCE', ['roost', 'earthquake', 'headbutt'], 'SPECTRIER', ['shadowball', 'darkpulse', 'drainingkiss']),
    ('MILOTIC', ['recover', 'icebeam', 'scald'], 'GRENINJA', ['hydropump', 'darkpulse', 'icebeam']),
    ('EMPOLEON', ['roost', 'surf', 'knockoff'], 'LUMINEON', ['alluringvoice', 'hydropump', 'icebeam']),
    ('NOCTOWL', ['roost', 'hypervoice', 'hurricane'], 'SPECTRIER', ['shadowball', 'darkpulse', 'drainingkiss']),
    ('SINISTCHA', ['strengthsap', 'shadowball', 'matchagotcha'], 'ELECTRODEHISUI', ['thunderbolt', 'gigadrain']),
    ('VESPIQUEN', ['roost', 'toxicspikes', 'hurricane'], 'DUGTRIOALOLA', ['earthquake', 'suckerpunch', 'ironhead']),
    ('SWANNA', ['roost', 'hydropump', 'bravebird'], 'DUGTRIOALOLA', ['earthquake', 'suckerpunch', 'ironhead']),
]
OB = {
    'none': [],
    'two': [('SPECTRIER', ['shadowball', 'darkpulse']), ('LUMINEON', ['hydropump', 'icebeam'])],
}


def job(c):
    sys.path.insert(0, 'src')
    r, rm, hp, o, om, obk = c
    ours = [P.mon(r, hp=hp, moves=rm, item='LEFTOVERS'), P.mon('TERRAKION', hp=0.1)]
    theirs = [P.mon(o, moves=om, item='LIFEORB')] + [P.mon(s, moves=m, item='LIFEORB') for s, m in OB[obk]]
    st = P.state(P.side(ours), P.side(theirs))
    return c, st, P.rollout_table(st, 60)


if __name__ == '__main__':
    cands = []
    for (r, rm, o, om), hp, obk in itertools.product(BASES, (0.35, 0.45, 0.55), ('none', 'two')):
        if o in [s for s, _ in OB[obk]]:
            continue
        cands.append((r, rm, hp, o, om, obk))
    print(len(cands), file=sys.stderr)
    out = []
    with ProcessPoolExecutor(5) as pool:
        for c, st, tb in pool.map(job, cands, chunksize=1):
            V = {a: max(v.values()) for a, v in tb.items()}
            rec = [a for a in V if a in {m.lower() for m in P.RECOVERY}]
            vr = max(V[a] for a in rec)
            vo = max(V[a] for a in V if a not in rec)
            out.append({'c': c, 'state': st, 'tb': tb})
            print(f"{vr - vo:+.2f} rec={vr:.2f} oth={vo:.2f}", c, {a: round(v, 2) for a, v in V.items()}, flush=True)
    json.dump(out, open('.scratch/stall2.json', 'w'))
