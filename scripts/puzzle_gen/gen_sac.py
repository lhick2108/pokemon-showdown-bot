"""Generate sacrifice candidates (A low and doomed, B a counter) and rollout-evaluate them."""
import json, random, sys
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'src')
import puzzles as P
from concurrent.futures import ProcessPoolExecutor
from poke_engine import State, calculate_damage

BAD = {'REVIVALBLESSING', 'PROTECT', 'SUBSTITUTE', 'REST', 'SLEEPTALK', 'BATONPASS', 'TRICKROOM', 'SUNNYDAY',
       'RAINDANCE', 'SANDSTORM', 'SNOWSCAPE', 'FINALGAMBIT', 'MEMENTO', 'EXPLOSION', 'SELFDESTRUCT',
       'HEALINGWISH', 'LUNARDANCE', 'WISH', 'DESTINYBOND', 'COUNTER', 'MIRRORCOAT', 'ENDURE', 'FAKEOUT',
       'SPLASH', 'UTURN', 'VOLTSWITCH', 'FLIPTURN', 'PARTINGSHOT', 'TELEPORT', 'ROAR', 'WHIRLWIND', 'DRAGONTAIL',
       'CIRCLETHROW', 'HAZE', 'COURTCHANGE', 'SHEDTAIL', 'CHILLYRECEPTION', 'ENCORE', 'TAUNT', 'TERABLAST',
       'SYNTHESIS', 'RECOVER', 'ROOST', 'SLACKOFF', 'SOFTBOILED', 'MOONLIGHT', 'MORNINGSUN', 'STRENGTHSAP',
       'MILKDRINK', 'SHOREUP', 'HEALORDER', 'HEALBELL', 'AROMATHERAPY', 'DEFOG', 'RAPIDSPIN', 'MAGICCOAT'}


def fld(m):
    return m.split(',')


def dmgs(state_str, m1, m2, first):
    s = State.from_string(state_str)
    a, b = calculate_damage(s, m1, m2, first)
    me = s.side_one.pokemon[int(s.side_one.active_index)]
    foe = s.side_two.pokemon[int(s.side_two.active_index)]
    return (a[0] / foe.maxhp, a[1] / foe.maxhp), (b[0] / me.maxhp, b[1] / me.maxhp), me, foe


def moves_of(sp):
    return [x.split(';')[0] for x in fld(P.lib()[sp])[22:26] if not x.startswith('NONE')]


def clean(sp):
    return [m.lower() for m in moves_of(sp) if m not in KAM] or ['tackle']


KAM = {'EXPLOSION', 'SELFDESTRUCT', 'MISTYEXPLOSION', 'FINALGAMBIT', 'MEMENTO', 'HEALINGWISH', 'LUNARDANCE', 'DESTINYBOND', 'PERISHSONG'}


def build(a, b, o, extra, hp_a, a_moves, b_moves, o_moves, c=None, e_hp=0.6):
    ours = [P.mon(a, hp=hp_a, moves=[m.lower() for m in a_moves]),
            P.mon(b, moves=[m.lower() for m in b_moves if m not in KAM] or ['tackle'])]
    if c:
        ours.append(P.mon(c, moves=clean(c)))
    theirs = [P.mon(o, moves=[m.lower() for m in o_moves])] + [P.mon(e, hp=e_hp, moves=clean(e)) for e in extra]
    return P.state(P.side(ours), P.side(theirs))


def job(c):
    sys.path.insert(0, 'src')
    st = c['state']
    P.FORCED_PREF[:] = [c['b'].lower(), c['c'].lower()]
    tb = P.rollout_table(st, 30)
    c['tb'] = tb
    return c


if __name__ == '__main__':
    L = P.lib()
    rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    species = [s for s in L if len(moves_of(s)) == 4]
    cands = []
    tries = 0
    while len(cands) < int(sys.argv[1]) and tries < 200000:
        tries += 1
        a, b, o, e1, c3 = rng.sample(species, 5)
        e2 = None
        fa, fo = fld(L[a]), fld(L[o])
        if int(fo[17]) <= int(fa[17]) + 5:
            continue
        am = [m for m in moves_of(a)]
        om = [m for m in moves_of(o) if m not in BAD]
        bm = moves_of(b)
        if len(om) < 2:
            continue
        am_att = [m for m in am if m not in BAD]
        if not am_att:
            continue
        hp_a = rng.choice([0.12, 0.16, 0.2])
        try:
            st = build(a, b, o, [e1], hp_a, am, bm, om, c3, rng.choice([0.4, 0.7]))
            # O's damage to A (B at back): use A's first move
            best_o = 0.0
            mino = 0.0
            for m in om:
                _, d, me, foe = dmgs(st, am[0].lower(), m.lower(), False)
                if d[0] > mino:
                    mino = d[0]
                best_o = max(best_o, d[1])
            if mino < hp_a + 0.01:  # must be a sure OHKO on A with some move
                continue
            # A's best chip on O
            chip = 0.0
            for m in am_att:
                x, _, _, _ = dmgs(st, m.lower(), om[0].lower(), False)
                chip = max(chip, x[1])
            # B switched in: B's attack on O, O's attack on B
            st_b = build(b, a, o, [e1], 1.0, bm, am, om)
            bdmg, bkill = 0.0, 0.0
            odmgb = 0.0
            for m in [x for x in bm if x not in BAD]:
                x, _, _, _ = dmgs(st_b, m.lower(), om[0].lower(), True)
                bdmg = max(bdmg, x[0])
            for m in om:
                _, y, _, _ = dmgs(st_b, bm[0].lower(), m.lower(), False)
                odmgb = max(odmgb, y[1])
        except Exception as ex:
            continue
        if bdmg < 0.55 or not (0.12 <= odmgb <= 0.6):
            continue
        cands.append({'a': a, 'b': b, 'o': o, 'extra': [e1], 'c': c3, 'e_hp': st.split(',')[6] if False else None, 'hp_a': hp_a, 'am': am, 'bm': bm, 'om': om,
                      'state': st, 'chip': round(chip, 2), 'odmgb': round(odmgb, 2), 'bdmg': round(bdmg, 2)})
    print(len(cands), 'candidates after', tries, 'tries', file=sys.stderr)
    cands = cands[:int(sys.argv[1])]
    out = []
    with ProcessPoolExecutor(5) as pool:
        for c in pool.map(job, cands, chunksize=1):
            out.append(c)
            print(c['a'], c['hp_a'], c['b'], c['o'], c['chip'], c['odmgb'],
                  {k: tuple(round(x, 2) for x in v.values()) for k, v in c['tb'].items()}, flush=True)
    json.dump(out, open('.scratch/sac_cands_%s.json' % (sys.argv[2] if len(sys.argv) > 2 else '3'), 'w'))
