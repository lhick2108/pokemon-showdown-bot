// Generate random-vs-random gen9randombattle protocol logs (per-side views) from the
// local Showdown checkout. Usage: node gen_protocol.js <showdown-dir> <n> <seed> [regex-filter] > out.jsonl
const [dir, n, seed0] = [process.argv[2], +process.argv[3], +process.argv[4] || 1];
const Sim = require(dir + '/dist/sim');
const {RandomPlayerAI} = require(dir + '/dist/sim/tools/random-player-ai');
(async () => {
  for (let i = 0; i < n; i++) {
    const streams = Sim.getPlayerStreams(new Sim.BattleStream());
    const logs = {p1: [], p2: []};
    const seed = [seed0 + i, 2, 3, 4];
    const ai = {
      p1: new RandomPlayerAI(streams.p1, {seed: [i + 1, 2, 3, 4]}),
      p2: new RandomPlayerAI(streams.p2, {seed: [i + 5, 6, 7, 8]}),
    };
    for (const p of ['p1', 'p2']) {
      const orig = ai[p].receiveLine.bind(ai[p]); ai[p].receiveLine = (l) => { if (!l.startsWith('|request')) logs[p].push(l); return orig(l); };
      ai[p].start();
    }
    streams.omniscient.write(`>start ${JSON.stringify({formatid: 'gen9randombattle', seed})}\n>player p1 {"name":"p1"}\n>player p2 {"name":"p2"}`);
    await new Promise(res => { (async () => { for await (const c of streams.omniscient) if (c.includes('|win|') || c.includes(String.fromCharCode(10) + '|tie' + String.fromCharCode(10))) break; res(); })(); });
    const filter = process.argv[5];
    if (filter && !new RegExp(filter).test(logs.p1.join(' '))) continue;
    console.log(JSON.stringify({tag: `battle-gen9randombattle-${i}`, p1: logs.p1, p2: logs.p2}));
  }
})();
