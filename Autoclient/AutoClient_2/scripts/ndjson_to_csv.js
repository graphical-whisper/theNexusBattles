// scripts/ndjson_to_csv.js
// Uso: node scripts/ndjson_to_csv.js data/pve_turns.ndjson data/pve_turns.csv [--match=ROOMID-...]
const fs = require('fs');
const readline = require('readline');

const inFile  = process.argv[2] || 'data/pve_turns.ndjson';
const outFile = process.argv[3] || 'data/pve_turns.csv';
const matchArg = (process.argv.find(a => a.startsWith('--match=')) || '').split('=')[1];

function flatten(obj, prefix = '', out = {}) {
  for (const [k, v] of Object.entries(obj || {})) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (v && typeof v === 'object' && !Array.isArray(v)) flatten(v, key, out);
    else out[key] = Array.isArray(v) ? JSON.stringify(v) : v;
  }
  return out;
}

(async () => {
  const rl = readline.createInterface({ input: fs.createReadStream(inFile), crlfDelay: Infinity });
  const rows = [];
  const keys = new Set();

  for await (const line of rl) {
    if (!line.trim()) continue;
    const obj = JSON.parse(line);
    if (matchArg && obj.match_id !== matchArg) continue; // filtra por partida si se pide
    const flat = {
      ts: obj.ts,
      match_id: obj.match_id,
      room_id: obj.room_id,
      turn_index: obj.turn_index,
      actor_id: obj.actor_id,
      enemy_id: obj.enemy_id,
      chosen_action_kind: obj.chosen_action_kind,
      chosen_skill_id: obj.chosen_skill_id,
      chosen_action_group: obj.chosen_action_group,
      dmg_to_enemy: obj.dmg_to_enemy,
      heal_to_self: obj.heal_to_self,
      effect_applied: obj.effect_applied,
      enemy_hp_after: obj.enemy_hp_after,
      actor_hp_after: obj.actor_hp_after,
      battle_finished: obj.battle_finished,
      winner_id: obj.winner_id,
      // aplanamos estado y máscara:
      ...flatten(obj.actor, 'actor'),
      ...flatten(obj.enemy, 'enemy'),
      'mask.num_specials_valid_off': obj.num_specials_valid_off,
      'mask.num_specials_valid_def': obj.num_specials_valid_def,
      'mask.num_support_valid': obj.num_support_valid,
      'mask.num_masters_valid_off': obj.num_masters_valid_off,
      'mask.num_masters_valid_def': obj.num_masters_valid_def,
      'mask.num_masters_support': obj.num_masters_support
    };
    rows.push(flat);
    Object.keys(flat).forEach(k => keys.add(k));
  }

  const header = Array.from(keys);
  const escape = v => {
    if (v === null || v === undefined) return '';
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };

  const out = fs.createWriteStream(outFile, { flags: 'w' });
  out.write(header.join(',') + '\n');
  for (const r of rows) {
    out.write(header.map(k => escape(r[k])).join(',') + '\n');
  }
  out.end();
  out.on('finish', () => console.log(`[csv] wrote ${rows.length} rows -> ${outFile}`));
})();
