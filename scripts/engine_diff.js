// Node driver for scripts/engine_diff.py: plays seeded gen9randombattle games in the real Showdown
// simulator (in-process, no server) and prints one JSON line per TURN with the FULL true state
// before and after the turn, the chosen actions and the turn's protocol log.
//
//   node scripts/engine_diff.js <showdown-dir> <games> <seed0> [maxTurns]
//   node scripts/engine_diff.js <showdown-dir> scenario <scenarios.json>
//
// Random mode: one game = Battle seeded [seed0 + i, 2, 3, 4]; both sides pick random legal actions (a
// seeded PRNG separate from the battle PRNG): ~12% switch when allowed, otherwise a uniformly random
// enabled move, Terastallize with 15% probability while available. A "turn" record spans everything
// between two "move" requests, i.e. the move pair plus any forced replacement / pivot switches
// (`followups`).
//
// Scenario mode: hand-built teams (gen9customgame) and scripted commands, used by `engine_diff.py
// suspects` to probe single mechanics. Each scenario: {name, p1: [set..], p2: [set..], script: [[cmd1, cmd2], ..],
// seeds: [..], weather?, faint?: [n1, n2]}. Every scripted turn is emitted with `scenario` and `step`.
'use strict';
const dir = process.argv[2];
const {Battle} = require(dir + '/dist/sim/battle');
const {PRNG} = require(dir + '/dist/sim/prng');
const {Teams} = require(dir + '/dist/sim/teams');

function simpleState(o) {
	if (!o) return null;
	const out = {};
	for (const k of Object.keys(o)) {
		const v = o[k];
		if (v === null || ['number', 'string', 'boolean'].includes(typeof v)) out[k] = v;
		else if (v && typeof v === 'object' && typeof v.id === 'string' && k !== 'target' && k !== 'source') out[k] = v.id;
	}
	return out;
}

// Stable identity of a team member: the set species (randbats has one per species). Not
// `baseSpecies`, which permanent forme changes (Disguise, Tera Shift, Tera) rewrite mid-battle.
function uidOf(p) {
	return ((p.set && p.set.species) || p.baseSpecies.name).toLowerCase().replace(/[^a-z0-9]+/g, "");
}

function dumpMon(p) {
	if (p.status !== 'slp') p.restFlag = false;
	const volatiles = {};
	for (const id of Object.keys(p.volatiles)) volatiles[id] = simpleState(p.volatiles[id]);
	const ss = p.statusState || {};
	return {
		uid: uidOf(p), sid: p.species.id, name: p.name, level: p.level,
		hp: p.hp, maxhp: p.maxhp, fainted: p.fainted, active: p.isActive,
		types: p.getTypes(false, true), baseTypes: p.baseSpecies.types.slice(),
		ability: p.ability, baseAbility: p.baseAbility, item: p.item, lastItem: p.lastItem,
		stats: p.storedStats, status: p.status,
		statusState: {time: ss.time, startTime: ss.startTime, stage: ss.stage, src: p.status === 'slp' && p.restFlag ? 'rest' : null},
		moves: p.moveSlots.map((m) => ({id: m.id, pp: m.pp, maxpp: m.maxpp, disabled: !!m.disabled})),
		teraType: p.teraType, terastallized: p.terastallized || '', boosts: Object.assign({}, p.boosts),
		volatiles, weightkg: p.weighthg / 10, timesAttacked: p.timesAttacked || 0,
		lastMove: p.lastMove ? p.lastMove.id : null, activeTurns: p.activeTurns, trapped: p.trapped || false,
		transformed: !!p.volatiles['transform'],
		// Protean / Libero (gen 9): once per switch-in, tracked in the ability state, not a volatile
		abilityOnce: !!((p.abilityState || {}).protean || (p.abilityState || {}).libero),
	};
}

function dumpSide(b, side) {
	const sc = {};
	for (const id of Object.keys(side.sideConditions)) sc[id] = simpleState(side.sideConditions[id]);
	const slot = {};
	const slotc = side.slotConditions[0] || {};
	for (const id of Object.keys(slotc)) slot[id] = simpleState(slotc[id]);
	return {id: side.id, mons: side.pokemon.map(dumpMon), sideConditions: sc, slotConditions: slot, totalFainted: side.totalFainted};
}

function dump(b) {
	const f = b.field;
	const pseudo = {};
	for (const id of Object.keys(f.pseudoWeather)) pseudo[id] = simpleState(f.pseudoWeather[id]);
	return {
		turn: b.turn, sides: b.sides.map((s) => dumpSide(b, s)),
		weather: f.weather, weatherDuration: f.weatherState ? f.weatherState.duration : 0,
		terrain: f.terrain, terrainDuration: f.terrainState ? f.terrainState.duration : 0,
		pseudoWeather: pseudo,
	};
}

function requestInfo(side) {
	const r = side.activeRequest || {};
	const info = {kind: r.wait ? 'wait' : r.forceSwitch ? 'switch' : r.teamPreview ? 'preview' : r.active ? 'move' : 'none'};
	if (info.kind === 'move') {
		const a = r.active[0];
		info.moves = a.moves.map((m) => ({id: m.id, disabled: !!m.disabled, pp: m.pp, maxpp: m.maxpp}));
		info.canTera = !!a.canTerastallize;
		info.trapped = !!(a.trapped || a.maybeTrapped);
	}
	info.switches = side.pokemon.map((p, i) => (!p.isActive && !p.fainted ? i : -1)).filter((i) => i >= 0);
	return info;
}

function pickAction(side, info, rng) {
	const canSwitch = info.switches.length > 0 && !info.trapped;
	if (canSwitch && rng.random() < 0.12) {
		const i = info.switches[rng.random(info.switches.length)];
		return {cmd: `switch ${i + 1}`, a: {type: 'switch', uid: uidOf(side.pokemon[i])}};
	}
	const enabled = info.moves.map((m, i) => (m.disabled ? -1 : i)).filter((i) => i >= 0);
	if (!enabled.length) {
		if (canSwitch) {
			const i = info.switches[rng.random(info.switches.length)];
			return {cmd: `switch ${i + 1}`, a: {type: 'switch', uid: uidOf(side.pokemon[i])}};
		}
		return {cmd: 'move 1', a: {type: 'move', id: info.moves[0].id, tera: false}};
	}
	const i = enabled[rng.random(enabled.length)];
	const tera = info.canTera && rng.random() < 0.15;
	return {cmd: `move ${i + 1}${tera ? ' terastallize' : ''}`, a: {type: 'move', id: info.moves[i].id, tera}};
}

function scriptedAction(side, info, cmd) {
	const m = /^(move|switch) (\d+)( terastallize)?$/.exec(cmd);
	if (!m) throw new Error('bad scripted command ' + cmd);
	const i = +m[2] - 1;
	if (m[1] === 'switch') return {cmd, a: {type: 'switch', uid: uidOf(side.pokemon[i])}};
	return {cmd, a: {type: 'move', id: info.moves[i].id, tera: !!m[3]}};
}

// Remember which sleepers fell asleep through Rest (statusState does not keep the source effect).
function markRest(b, fromIdx) {
	for (let i = fromIdx; i < b.log.length; i++) {
		const m = /^\|-status\|(p[12])[ab]: [^|]*\|slp\|\[from\] move: Rest/.exec(b.log[i]);
		if (m) {
			const side = b.sides[m[1] === 'p1' ? 0 : 1];
			if (side.active[0]) side.active[0].restFlag = true;
		}
	}
}

// One turn: the choices, then any forced switches until both sides want a move again.
function takeTurn(b, rng, scripted, extra) {
	const infos = b.sides.map(requestInfo);
	if (!infos.every((i) => i.kind === 'move')) return null;
	const pre = dump(b);
	const logStart = b.log.length;
	const picks = b.sides.map((s, k) => (scripted ? scriptedAction(s, infos[k], scripted[k]) : pickAction(s, infos[k], rng)));
	const actions = picks.map((p) => p.a);
	const requests = infos.map((i) => ({moves: i.moves, trapped: i.trapped}));
	picks.forEach((p, k) => b.choose(b.sides[k].id, p.cmd));
	markRest(b, logStart);
	let scanned = b.log.length;
	const followups = [];
	let guard = 0;
	while (!b.ended && guard++ < 10) {
		const fi = b.sides.map(requestInfo);
		if (fi.every((i) => i.kind === 'move')) break;
		const round = [null, null];
		const cmds = [null, null];
		b.sides.forEach((s, k) => {
			if (fi[k].kind !== 'switch') return;
			const i = fi[k].switches[rng.random(fi[k].switches.length)];
			if (i === undefined) return;
			round[k] = {type: 'switch', uid: uidOf(s.pokemon[i])};
			cmds[k] = `switch ${i + 1}`;
		});
		if (!cmds.some(Boolean)) break;
		followups.push(round);
		cmds.forEach((c, k) => { if (c) b.choose(b.sides[k].id, c); });
		markRest(b, scanned);
		scanned = b.log.length;
	}
	const post = dump(b);
	return Object.assign({
		turn: pre.turn, pre, post, actions, requests, followups,
		log: b.log.slice(logStart).filter((l) => !l.startsWith('|t:|')), ended: !!b.ended, winner: b.winner || null,
	}, extra || {});
}

function playGame(gi, seed0, maxTurns) {
	const seed = [seed0 + gi, 2, 3, 4];
	const b = new Battle({formatid: 'gen9randombattle', seed});
	b.setPlayer('p1', {name: 'p1', seed: [seed[0], 11, 3, 4]});
	b.setPlayer('p2', {name: 'p2', seed: [seed[0], 22, 3, 4]});
	const rng = new PRNG([seed[0], 5, 9, 13]);
	const out = [];
	while (!b.ended && out.length < maxTurns) {
		const rec = takeTurn(b, rng, null, {seed: seed[0], game: gi});
		if (!rec) break;
		out.push(rec);
	}
	return out;
}

function fillSet(s) {
	const evs = s.evs || {hp: 85, atk: 85, def: 85, spa: 85, spd: 85, spe: 85};
	return Object.assign({name: s.species, item: '', gender: '', nature: 'Hardy', level: 100, ivs: {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31}}, s, {evs});
}

function playScenario(sc, seed) {
	const b = new Battle({formatid: 'gen9customgame', seed: [seed, 2, 3, 4]});
	b.setPlayer('p1', {name: 'p1', team: Teams.pack(sc.p1.map(fillSet)), seed: [seed, 11, 3, 4]});
	b.setPlayer('p2', {name: 'p2', team: Teams.pack(sc.p2.map(fillSet)), seed: [seed, 22, 3, 4]});
	if (sc.weather) b.field.setWeather(sc.weather);
	if (sc.faint) {
		// really faint n bench Pokemon per side (the engine counts fainted team members, the sim counts totalFainted)
		sc.faint.forEach((n, k) => {
			const side = b.sides[k];
			for (let i = 1; i <= n; i++) {
				side.pokemon[i].hp = 0;
				side.pokemon[i].fainted = true;
				side.pokemon[i].status = '';
				side.pokemonLeft--;
				side.totalFainted++;
			}
		});
	}
	if (b.requestState === 'teampreview') b.sides.forEach((side) => b.choose(side.id, 'default'));
	const rng = new PRNG([seed, 5, 9, 13]);
	const out = [];
	sc.script.forEach((cmds, step) => {
		const rec = takeTurn(b, rng, cmds, {seed, game: 0, scenario: sc.name, step});
		if (rec) out.push(rec);
	});
	return out;
}

if (process.argv[3] === 'scenario') {
	const scs = JSON.parse(require('fs').readFileSync(process.argv[4], 'utf8'));
	for (const sc of scs) {
		for (const seed of sc.seeds) {
			let recs;
			try {
				recs = playScenario(sc, seed);
			} catch (e) {
				console.log(JSON.stringify({error: String(e && e.stack || e), scenario: sc.name, seed}));
				continue;
			}
			for (const r of recs) console.log(JSON.stringify(r));
		}
	}
} else {
	const [nGames, seed0, maxTurns] = [+process.argv[3], +process.argv[4] || 1, +process.argv[5] || 120];
	for (let g = 0; g < nGames; g++) {
		let recs;
		try {
			recs = playGame(g, seed0, maxTurns);
		} catch (e) {
			console.log(JSON.stringify({error: String(e && e.stack || e), game: g, seed: seed0 + g}));
			continue;
		}
		for (const r of recs) console.log(JSON.stringify(r));
	}
}
