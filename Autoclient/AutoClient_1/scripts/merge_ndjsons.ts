import fs from "fs";
import path from "path";

type Row = {
  ts: string;
  match_id: string;
  room_id: string;
  turn_index?: number;
  actor_id: string;
  enemy_id: string;
  battle_finished?: boolean;
  [k: string]: any;
};

type Cli = {
  input: string;          // carpeta de entrada (con .ndjson)
  output: string;         // archivo ndjson de salida
  gapSec: number;         // umbral de corte entre eventos (segundos)
  room?: string;          // (opcional) filtrar por room_id
};

function parseArgs(): Cli {
  const args = process.argv.slice(2);
  const get = (k: string, def?: string) => {
    const i = args.findIndex(a => a === `--${k}`);
    return i >= 0 ? args[i + 1] : def;
  };
  const input = get("input", "data")!;
  const output = get("output", path.join(input, "merged_turns.ndjson"))!;
  const gapSec = Number(get("gap", "120"));
  const room = get("room", undefined);
  return { input, output, gapSec, room };
}

function readAllNdjson(dir: string, roomFilter?: string): Row[] {
  const files = fs.readdirSync(dir).filter(f => f.endsWith(".ndjson"));
  const all: Row[] = [];
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
      } catch (e) {
        console.error(`Línea inválida en ${f}: ${s}`);
      }
    }
  }
  return all;
}

function tsMs(r: Row): number {
  return new Date(r.ts).getTime();
}

function playersKey(a: string, b: string): string {
  return [a, b].sort().join("|");
}

function groupBy<T, K extends string | number>(arr: T[], keyFn: (x: T) => K): Record<K, T[]> {
  return arr.reduce((acc, x) => {
    const k = keyFn(x);
    (acc[k] ||= []).push(x);
    return acc;
  }, {} as Record<K, T[]>);
}

function segmentSessions(records: Row[], gapMs: number): Row[][] {
  // Precondición: mismos room_id y mismo set de jugadores
  const sorted = [...records].sort((a, b) => tsMs(a) - tsMs(b));
  const sessions: Row[][] = [];
  let current: Row[] = [];
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

function unifySession(session: Row[]): Row[] {
  // Orden por ts ascendente
  const sorted = [...session].sort((a, b) => tsMs(a) - tsMs(b));
  // nuevo match id determinista basado en primer timestamp
  const roomId = sorted[0].room_id;
  const firstTs = tsMs(sorted[0]);
  const unified_match_id = `${roomId}-${firstTs}`;

  // reindexar turnos 1..N y sobrescribir match_id
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

  const rows = readAllNdjson(input, room);
  if (rows.length === 0) {
    console.error("No se encontraron filas .ndjson en", input, room ? `(room=${room})` : "");
    process.exit(1);
  }

  // 1) agrupar por room_id
  const byRoom = groupBy(rows, r => r.room_id);

  const out: Row[] = [];

  for (const [roomId, roomRows] of Object.entries(byRoom)) {
    // 2) en cada room, agrupar por set de jugadores {actor_id, enemy_id}
    const byPlayers = groupBy(roomRows, r => playersKey(r.actor_id, r.enemy_id));

    for (const [pairKey, recs] of Object.entries(byPlayers)) {
      // 3) segmentar en sesiones por gap y por "battle_finished"
      const sessions = segmentSessions(recs, gapMs);

      // 4) unificar cada sesión (recalcula match_id + turn_index)
      for (const s of sessions) {
        const unified = unifySession(s);
        out.push(...unified);
        // Si quieres además escribir cada sesión en archivo separado, puedes:
        // const firstTs = new Date(unified[0].ts).getTime();
        // const sessionFile = path.join(input, `merged_${roomId}_${pairKey.replace('|','-')}_${firstTs}.ndjson`);
        // fs.writeFileSync(sessionFile, unified.map(x => JSON.stringify(x)).join("\n") + "\n");
      }
    }
  }

  // 5) escribir dataset global
  out.sort((a, b) => {
    const c = a.match_id.localeCompare(b.match_id);
    return c !== 0 ? c : (a.turn_index! - b.turn_index!);
  });

  const stream = fs.createWriteStream(output, { flags: "w" });
  for (const r of out) stream.write(JSON.stringify(r) + "\n");
  stream.end();

  console.log(`OK -> ${output}`);
}

main();
