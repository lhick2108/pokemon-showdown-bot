// Node helper for scripts/ko_check.py: exact 16-roll damage from @smogon/calc.
// Install once:  cd scratch/tools && npm install @smogon/calc   (node_modules is gitignored)
// stdin: JSON array of cases; stdout: JSON array of {rolls|null, multihit, error?}.
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";
import fs from "node:fs";

const here = path.dirname(fileURLToPath(import.meta.url));
const toolsDir = process.env.KO_CALC_TOOLS || path.resolve(here, "..", "scratch", "tools");
const require = createRequire(path.join(toolsDir, "noop.js"));
const { Generations, Pokemon, Move, Field, Side, calculate } = require("@smogon/calc");
const gen = Generations.get(9);
// calculate() clones its inputs and clone() rebuilds stats from ev/iv/nature, so re-impose the
// engine's exact stats (and weight / current HP) on every clone.
const origClone = Pokemon.prototype.clone;
Pokemon.prototype.clone = function () {
  const c = origClone.call(this);
  c.rawStats = { ...this.rawStats };
  c.stats = { ...this.rawStats };
  c.originalCurHP = this.originalCurHP;
  c.weightkg = this.weightkg;
  return c;
};
const toID = (s) => String(s).toLowerCase().replace(/[^a-z0-9]/g, "");
const STATUS = { BURN: "brn", PARALYZE: "par", POISON: "psn", TOXIC: "tox", SLEEP: "slp", FREEZE: "frz" };
const WEATHER = { SUN: "Sun", RAIN: "Rain", SAND: "Sand", HAIL: "Hail", SNOW: "Snow", HARSHSUN: "Harsh Sunshine", HEAVYRAIN: "Heavy Rain" };
const TERRAIN = { ELECTRICTERRAIN: "Electric", GRASSYTERRAIN: "Grassy", MISTYTERRAIN: "Misty", PSYCHICTERRAIN: "Psychic" };
const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1).toLowerCase() : s);

function name(coll, id) {
  if (!id || id === "NONE") return undefined;
  const e = coll.get(toID(id));
  return e ? e.name : undefined;
}

function mkMon(m) {
  let sp = gen.species.get(toID(m.species));
  // cosmetic formes (Minior-Green, Florges-Blue, ...): fall back to the longest species-id prefix
  for (let n = toID(m.species).length - 1; !sp && n > 3; n--) sp = gen.species.get(toID(m.species).slice(0, n));
  if (!sp) throw new Error("species " + m.species);
  const o = {
    level: m.level,
    // calc re-applies Download boosts unconditionally and assumes Analytic moves last; neither is knowable from the state
    ability: ["download", "analytic"].includes(toID(m.ability)) ? "Pressure" : name(gen.abilities, m.ability),
    item: name(gen.items, m.item),
    status: STATUS[m.status],
    boosts: m.boosts,
    nature: "Hardy",
    curHP: m.hp,
    overrides: { types: m.types.map(cap) },
  };
  if (m.tera) o.teraType = cap(m.tera);
  if (m.boostedStat) o.boostedStat = m.boostedStat;
  if (m.abilityOn) o.abilityOn = true;
  if (m.alliesFainted) o.alliesFainted = m.alliesFainted;
  const p = new Pokemon(gen, sp.name, o);
  p.rawStats = { ...m.stats };
  p.stats = { ...m.stats };
  p.originalCurHP = m.hp;
  p.weightkg = m.weightkg;
  return p;
}

function mkSide(s) {
  return new Side({
    isReflect: !!s.reflect, isLightScreen: !!s.light_screen, isAuroraVeil: !!s.aurora_veil,
    isTailwind: !!s.tailwind,
  });
}

const cases = JSON.parse(fs.readFileSync(0, "utf8"));
const out = cases.map((c) => {
  try {
    const mv = gen.moves.get(toID(c.move));
    if (!mv) return { rolls: null, error: "move " + c.move };
    const multihit = !!mv.multihit;
    const atk = mkMon(c.atk), def = mkMon(c.def);
    const move = new Move(gen, mv.name, { isCrit: false });
    const field = new Field({
      gameType: "Singles",
      weather: WEATHER[c.field.weather],
      terrain: TERRAIN[c.field.terrain],
      attackerSide: mkSide(c.field.atkSide),
      defenderSide: mkSide(c.field.defSide),
    });
    const r = calculate(gen, atk, def, move, field);
    let d = r.damage;
    if (Array.isArray(d) && Array.isArray(d[0])) d = d[0];
    if (!Array.isArray(d)) return { rolls: null, fixed: typeof d === "number" ? d : null, multihit };
    if (d.length !== 16) return { rolls: null, error: "len" + d.length, multihit };
    return { rolls: d, multihit, bp: mv.basePower, type: mv.type, cat: mv.category, eff: r.defender.types.reduce((a, t) => a * (gen.types.get(toID(mv.type)).effectiveness[t] ?? 1), 1) };
  } catch (e) {
    return { rolls: null, error: String(e.message || e) };
  }
});
process.stdout.write(JSON.stringify(out));
