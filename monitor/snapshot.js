// Prints today's pregame rows as JSON, scored by the same engine the board uses.
// Usage: node monitor/snapshot.js data.json
const fs = require('fs'), path = require('path');
const { makeEngine } = require(path.join(__dirname, '..', 'engine.js'));
const DATA = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const E = makeEngine(DATA);
const r1 = v => v == null ? '' : Math.round(v * 10) / 10;
const r3 = v => v == null ? '' : Math.round(v * 1000) / 1000;
const rows = E.buildRows(DATA.today).map(r => ({
  date: DATA.today, gameId: r.g.id, state: r.g.state, start: r.g.start,
  team: r.ab, opp: r.opp, ha: r.ha,
  s_off: r1(r.s.off), s_def: r1(r.s.def), s_pace: r1(r.s.pace), s_fatigue: r1(r.s.fatigue), s_oppFatigue: r1(r.s.oppFatigue),
  s_avail: r1(r.s.avail), s_l5: r1(r.s.l5),
  score: r1(r.total), proj: r1(r.mu), line: r.line == null ? '' : r1(r.line), game_total: r.g.total ?? '', spread: r.g.spread ?? '',
  p_under: r3(r.pUnder), fatigue: r1(r.fat), opp_fatigue: r1(r.oppFat),
  rest: r.f ? r.f.rest : '', b2b: r.f ? r.f.b2b : '', g4: r.f ? r.f.g4 : '', trip: r.f ? r.f.trip : '', km: r.f ? r.f.km : '',
  lost_share: r3(r.av.lost), out: r.av.who.filter(w => w.w >= 0.75).map(w => w.name).join('; '),
  model_version: DATA.model.version,
}));
process.stdout.write(JSON.stringify(rows));
