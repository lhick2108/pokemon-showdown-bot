// Generate seeded gen9randombattle teams with Showdown's own generator, packed for the
// gen9randombattlefixed format. One JSON object per line: {"id": n, "seed": [..], "packed": "..."}.
// Usage: node gen_teams.js <showdown-dir> <n> <seed> > teams.jsonl
const [dir, n, seed0] = [process.argv[2], +process.argv[3], +process.argv[4] || 1];
const {Teams} = require(dir + '/dist/sim/teams');
for (let i = 0; i < n; i++) {
  const seed = [seed0, i, 3, 4];
  const team = Teams.generate('gen9randombattle', {seed});
  console.log(JSON.stringify({id: i, seed, packed: Teams.pack(team)}));
}
