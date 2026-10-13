import json, random, sys
sys.path.insert(0, 'scripts')
import puzzles as P

rng = random.Random(21)
REC = {m.lower() for m in P.RECOVERY}

# sacrifice pool
rows = []
for f in ('.scratch/sac_cands_5.json', '.scratch/sac_cands_6.json'):
    for c in json.load(open(f)):
        V = {a: max(v.values()) for a, v in c['tb'].items()}
        sw = {a: v for a, v in V.items() if a in (c['b'].lower(), c['c'].lower())}
        st = {a: v for a, v in V.items() if a not in sw}
        if sw and st:
            rows.append((max(st.values()) - max(sw.values()), max(st.values()), max(sw.values()), c))
pos = [r for r in rows if r[0] >= 0.2 and r[1] >= 0.7]
ctl = [r for r in rows if r[0] <= -0.4 and r[2] >= 0.85]
oth = [r for r in rows if abs(r[0]) < 0.2]
rng.shuffle(pos); rng.shuffle(ctl); rng.shuffle(oth)
pool = []
for kind, grp, k in (('gpos', pos, 90), ('gctl', ctl, 60), ('goth', oth, 40)):
    for g, vs, vw, c in grp[:k]:
        pool.append({'id': f'P{len(pool):03d}', 'theme': 'sacrifice', 'src': kind, 'state': c['state'],
                     'a': c['a'], 'b': c['b'], 'c': c['c'], 'o': c['o'], 'extra': c['extra'], 'hp_a': c['hp_a'],
                     'am': c['am'], 'om': c['om'], 'pref': [c['b'].lower(), c['c'].lower()]})
json.dump(pool, open('.scratch/pool_sac.json', 'w'))
print('sac pool', len(pool))

# stall control pool and extra stall positives
pool2 = []
for f in ('.scratch/ctrl_a.json', '.scratch/ctrl_b.json'):
    for d in json.load(open(f)):
        c = d['c']
        V = {a: max(v.values()) for a, v in d['tb'].items()}
        rec = [a for a in V if a in REC]
        if not rec:
            continue
        if max(v for a, v in V.items() if a not in rec) - max(V[a] for a in rec) >= 0.0 or c[6] == 'bighit':
            pool2.append({'id': f'Q{len(pool2):03d}', 'theme': 'stall', 'src': 'ctl:' + c[6], 'state': d['state'],
                          'r': c[0], 'hp': c[2], 'rm': list(c[1]), 'o': c[3], 'om': list(c[4]), 'item': c[5]})
for d in json.load(open('.scratch/stall2.json')):
    c = d['c']
    pool2.append({'id': f'Q{len(pool2):03d}', 'theme': 'stall', 'src': 'pos', 'state': d['state'],
                  'r': c[0], 'hp': c[2], 'rm': list(c[1]), 'o': c[3], 'om': list(c[4]), 'item': 'LIFEORB', 'obk': c[5]})
json.dump(pool2, open('.scratch/pool_stall.json', 'w'))
print('stall pool', len(pool2))
