// scripts/merge_ndjson.cjs
const fs = require("fs");
const path = require("path");

function parseArgs() {
  const args = process.argv.slice(2);
  const get = (k, def) => {
    const i = args.indexOf(`--${k}`);
    return i >= 0 ? args[i + 1] : def;
  };
  return {
    input: get("input", "data"),
    output: get("output", path.join(get("input", "data"), "merged_turns.ndjson")),
    gapSec: Number(get("gap", "120")),
    room: get("room", undefined),
  };
}

function readAllNdjson(dir, roomFilter) {
  const files = fs.readdirSync(dir).filter(f => f.endsWith(".ndjson"));
  const all = [];
  for (const f of files) {
    const full = path.join(dir, f);
    const txt = fs.readFileSync(full, "utf-8");
    for (const line of txt.split("\n")) {
      const s = line.trim();
      if (!s) continue;
      try {
        const obj = JSON.parse(s);
        if (roomFilter && obj.room_id !== roomFilter) continue;
        all.push(obj);
      } catch {
        console.error(`Línea inválida en ${f}: ${s}`);
      }
    }
  }
  return all;
}

const tsMs = r => new Date(r.ts).getTime();
const playersKey = (a, b) => [a, b].sort().join("|");
function groupBy(arr, keyFn) {
  return arr.reduce((acc, x) => {
    const k = keyFn(x);
    (acc[k] ||= []).push(x);
    return acc;
  }, {});
}

function segmentSessions(records, gapMs) {
  const sorted = [...records].sort((a, b) => tsMs(a) - tsMs(b));
  const sessions = [];
  let current = [];
  let lastTs = -Infinity;

  for (const r of sorted) {
    const t = tsMs(r);
    const startNew =
      current.length === 0 ||
      (t - lastTs) > gapMs ||
      (current[current.length - 1].battle_finished === true);
    if (startNew) {
      if (current.length) sessions.push(current);
      current = [r];
    } else {
      current.push(r);
    }
    lastTs = t;
  }
  if (current.length) sessions.push(current);
  return sessions;
}

function unifySession(session) {
  const sorted = [...session].sort((a, b) => tsMs(a) - tsMs(b));
  const roomId = sorted[0].room_id;
  const firstTs = tsMs(sorted[0]);
  const unified_match_id = `${roomId}-${firstTs}`;
  let idx = 1;
  return sorted.map(r => {
    const copy = { ...r };
    copy.match_id = unified_match_id;
    copy.turn_index = idx++;
    return copy;
  });
}

function main() {
  const { input, output, gapSec, room } = parseArgs();
  const gapMs = gapSec * 1000;

  if (!fs.existsSync(input) || !fs.statSync(input).isDirectory()) {
    console.error(`No existe el directorio de entrada: ${input}`);
    process.exit(1);
  }

  const rows = readAllNdjson(input, room);
  if (rows.length === 0) {
    console.error("No se encontraron filas .ndjson en", input, room ? `(room=${room})` : "");
    process.exit(1);
  }

  const byRoom = groupBy(rows, r => r.room_id);
  const out = [];

  for (const roomId of Object.keys(byRoom)) {
    const roomRows = byRoom[roomId];
    const byPlayers = groupBy(roomRows, r => playersKey(r.actor_id, r.enemy_id));
    for (const pairKey of Object.keys(byPlayers)) {
      const recs = byPlayers[pairKey];
      const sessions = segmentSessions(recs, gapMs);
      for (const s of sessions) {
        out.push(...unifySession(s));
      }
    }
  }

  out.sort((a, b) => {
    const c = a.match_id.localeCompare(b.match_id);
    return c !== 0 ? c : (a.turn_index - b.turn_index);
  });

  const outDir = path.dirname(output);
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

  const stream = fs.createWriteStream(output, { flags: "w" });
  for (const r of out) stream.write(JSON.stringify(r) + "\n");
  stream.end();

  console.log(`OK -> ${output}`);
}

main();

// correr con node scripts/merge_ndjson.cjs --input data --output data/merged_turns.ndjson --gap 120
