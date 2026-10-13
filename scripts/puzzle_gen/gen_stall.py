"""Generate and rollout-evaluate stall candidates; dump to .scratch/stall_cands_all.json."""
import json, random, sys
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'src')
import puzzles as P
import puzzle_defs as D
from concurrent.futures import ProcessPoolExecutor

SETUP = {'SWORDSDANCE', 'NASTYPLOT', 'PROTECT', 'DRAGONDANCE', 'CALMMIND', 'QUIVERDANCE', 'BULKUP', 'ENCORE',
         'THUNDERWAVE', 'WILLOWISP', 'SPORE', 'SLEEPPOWDER', 'TAUNT', 'TOXIC', 'STEALTHROCK', 'SUNNYDAY',
         'TRICKROOM', 'SPIKES', 'DEFOG', 'ROAR', 'WHIRLWIND', 'AGILITY', 'CURSE', 'IRONDEFENSE', 'HAZE',
         'UTURN', 'VOLTSWITCH', 'FLIPTURN', 'PARTINGSHOT', 'COURTCHANGE', 'KNOCKOFF', 'GLARE', 'LEECHSEED',
         'NORETREAT', 'WORKUP', 'SHIFTGEAR', 'TERABLAST', 'REVIVALBLESSING', 'RAPIDSPIN'}
REC = set(P.RECOVERY) - {'WISH', 'REST'}


def attacks(mv):
    return [m for m in mv if m not in SETUP and m not in REC and m != 'NONE']


def job(c):
    sys.path.insert(0, 'src')
    r, rm, hp, o, om, item, bench = c
    ours = [P.mon(r, hp=hp, moves=[m.lower() for m in rm], item='LEFTOVERS')] + [P.mon(bench, hp=0.1)]
    theirs = [P.mon(o, moves=[m.lower() for m in om], item=item)]
    st = P.state(P.side(ours), P.side(theirs))
    tb = P.rollout_table(st, 60)
    return c, st, tb


if __name__ == '__main__':
    L = P.lib()
    R, O = [], []
    for sp, m in L.items():
        f = m.split(',')
        mv = [x.split(';')[0] for x in f[22:26]]
        if set(mv) & REC and f[10] != 'LIFEORB' and len(attacks(mv)) >= 1:
            R.append((sp, mv))
        if f[10] == 'LIFEORB' and len(attacks(mv)) >= 2:
            O.append((sp, mv))
    rng = random.Random(7)
    cands = []
    for rs, rm in R:
        rr = [m for m in rm if m in REC][0]
        ratk = attacks(rm)[:2]
        for os_, om in O:
            oa = attacks(om)[:3]
            if os_ == rs:
                continue
            c = (rs, [rr] + ratk, 0.5, os_, oa, 'LIFEORB', 'TERRAKION')
            try:
                st = D.stall_state(rs, 0.5, [m.lower() for m in c[1]], os_, [m.lower() for m in oa], bench=())
                h = D.hits(st)
            except Exception:
                continue
            mx = max(v[1] for v in h.values())
            mn = max(v[0] for v in h.values())
            if 0.30 <= mx <= 0.46 and mn >= 0.2:
                for hp in (min(0.6, round(mx + 0.05, 2)), 0.6):
                    cands.append((rs, c[1], hp, os_, oa, 'LIFEORB', 'TERRAKION'))
    print(len(cands), 'candidates', file=sys.stderr)
    rng.shuffle(cands)
    cands = cands[:int(sys.argv[1])]
    out = []
    with ProcessPoolExecutor(5) as pool:
        for c, st, tb in pool.map(job, cands, chunksize=1):
            out.append({'c': c, 'state': st, 'tb': tb})
            rec = [a for a in tb if a in {m.lower() for m in REC}]
            print(c[0], c[3], c[2], {a: tuple(v.values()) for a, v in tb.items()}, flush=True)
    json.dump(out, open('.scratch/stall_cands_all.json', 'w'))
