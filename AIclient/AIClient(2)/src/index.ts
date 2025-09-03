// AIClient(2)/src/index.ts — IA vs “cliente 1” con consulta de roster por socket (sin GET 404)
// Reqs: npm i socket.io-client axios
// Node >= 18 (fetch nativo). Si Node < 18: npm i undici y usa undici.fetch

import axios from "axios";
import { io, Socket } from "socket.io-client";

// ====== ENV / Config ======
const SOCKET_URL = process.env["SOCKET_URL"] || "http://localhost:3000";
const API_URL    = process.env["API_URL"]    || "http://localhost:3000";
const AI_URL     = process.env["AI_URL"]     || "http://127.0.0.1:8000";

const ROOM_ID    = process.env["ROOM_ID"]    || "ZZZ000";
const MY_ID      = process.env["MY_ID"]      || "playerB";
const TEAM       = process.env["TEAM"]       || "B";

// Héroe configurable (por defecto Rogue Poison lvl 2)
const HERO_TYPE  = (process.env["HERO_TYPE"]  || "ROGUE_POISON").toUpperCase();
const HERO_LEVEL = Number(process.env["HERO_LEVEL"] || 2);
const HERO_ATK   = Number(process.env["HERO_ATK"]   || 12);
const HERO_DEF   = Number(process.env["HERO_DEF"]   || 10);
const HERO_HP    = Number(process.env["HERO_HP"]    || 80);
const HERO_PWR   = Number(process.env["HERO_PWR"]   || 10);

// ====== Catálogo slots → skillId del juego ======
type Slot = "SPECIAL_SKILL_1" | "SPECIAL_SKILL_2" | "SPECIAL_SKILL_3";
const SLOT_TO_SKILL: Record<string, Record<Slot, string>> = {
  TANK:          { SPECIAL_SKILL_1:"GOLPE_ESCUDO",      SPECIAL_SKILL_2:"MANO_PIEDRA",   SPECIAL_SKILL_3:"DEFENSA_FEROZ" },
  WARRIOR_ARMS:  { SPECIAL_SKILL_1:"EMBATE_SANGRIENTO", SPECIAL_SKILL_2:"LANZA_DIOSES",  SPECIAL_SKILL_3:"GOLPE_TORMENTA" },
  MAGE_FIRE:     { SPECIAL_SKILL_1:"MISILES_MAGMA",     SPECIAL_SKILL_2:"VULCANO",       SPECIAL_SKILL_3:"PARED_FUEGO" },
  MAGE_ICE:      { SPECIAL_SKILL_1:"LLUVIA_HIELO",      SPECIAL_SKILL_2:"CONO_HIELO",    SPECIAL_SKILL_3:"BOLA_HIELO" },
  ROGUE_POISON:  { SPECIAL_SKILL_1:"FLOR_LOTO",         SPECIAL_SKILL_2:"AGONIA",        SPECIAL_SKILL_3:"PIQUETE" },
  ROGUE_MACHETE: { SPECIAL_SKILL_1:"CORTADA",           SPECIAL_SKILL_2:"MACHETAZO",     SPECIAL_SKILL_3:"PLANAZO" },
};
function heroKeyForAI(ht: string|number) {
  const s = String(ht).toUpperCase();
  if (s === "POISON_ROGUE")  return "ROGUE_POISON";
  if (s === "MACHETE_ROGUE") return "ROGUE_MACHETE";
  return s;
}

// ====== Tipos IA ======
type ActionKind = "BASIC_ATTACK" | "SPECIAL_SKILL_1" | "SPECIAL_SKILL_2" | "SPECIAL_SKILL_3";
type TargetKind = "ENEMY" | "SELF";
interface ActorState { heroType: string|number; level:number; power:number; attack?:number; defense?:number; health?:number; }
interface EnemyState { health?:number; defense?:number; }
interface PredictRequest { actor:ActorState; enemy:EnemyState; forbidden_actions?: ActionKind[]; }
interface PredictResponse { kind:ActionKind; skillId?:number|null; target:TargetKind; reason:string; }

const fetchFn: typeof fetch = (globalThis.fetch as any);

// ====== Estado de sala/combate ======
const socket: Socket = io(SOCKET_URL, { transports:["websocket"], reconnection:true, timeout:5000 });
const roster = new Set<string>();
let weReady = false;
let oppReady = false;
let battleJoined = false;
let turns: string[] = [];
let currentTurn: string | null = null;

// Stats del héroe local (mismo shape del server)
const HERO_STATS = {
  hero: {
    heroType: HERO_TYPE,
    level: HERO_LEVEL,
    power: HERO_PWR,
    health: HERO_HP,
    defense: HERO_DEF,
    attack: HERO_ATK,
    specialActions: [],
    randomEffects: [],
  },
  equipped: { items:[], armors:[], weapons:[], epicAbilites:[] }
};

// ====== Utils ======
const delay = (ms:number)=>new Promise(r=>setTimeout(r,ms));
function jitter(min:number,max:number){ return min + Math.random()*(max-min); }
function allowedActionsByLevel(level: number): ActionKind[] {
  if (level <= 1) return ["BASIC_ATTACK"];
  if (level <= 4) return ["BASIC_ATTACK", "SPECIAL_SKILL_1"];
  if (level <= 7) return ["BASIC_ATTACK", "SPECIAL_SKILL_1", "SPECIAL_SKILL_2"];
  return ["BASIC_ATTACK", "SPECIAL_SKILL_1", "SPECIAL_SKILL_2", "SPECIAL_SKILL_3"];
}
function localFallback(actor: ActorState): ActionKind {
  const allowed = allowedActionsByLevel(actor.level);
  return allowed.includes("SPECIAL_SKILL_1") ? "SPECIAL_SKILL_1" : "BASIC_ATTACK";
}
async function askAI(req: PredictRequest, timeoutMs=1800): Promise<PredictResponse|null> {
  const ctrl = new AbortController(); const t = setTimeout(()=>ctrl.abort(), timeoutMs);
  try{
    const res = await fetchFn(`${AI_URL}/predict`, {
      method:"POST", headers:{ "Content-Type":"application/json" }, body:JSON.stringify(req), signal:ctrl.signal
    } as RequestInit);
    if(!res.ok) return null;
    return await res.json() as PredictResponse;
  }catch{ return null; } finally{ clearTimeout(t); }
}
function getOpponent(): string { return turns.find(t => t !== MY_ID) || ""; }

// ====== Logger global ======
socket.onAny((ev, ...args)=>{
  try { console.log(`[ai] <= ${ev}`, JSON.stringify(args[0]??"", null, 2)); }
  catch { console.log(`[ai] <= ${ev}`); }
});

// ====== Helpers ACK socket ======
async function emitAckRaw(event: string, payload: any, timeout=1600): Promise<{ok:boolean; data:any}> {
  return new Promise(resolve => {
    let done=false; const timer=setTimeout(()=>{ if(!done){done=true; resolve({ok:false, data:null}); }}, timeout);
    try {
      (socket as any).timeout(timeout).emit(event, payload, (err: any, ack?: any) => {
        if (done) return; clearTimeout(timer); done=true;
        if (err) return resolve({ok:false, data:err});
        resolve({ ok:true, data: ack ?? null });
      });
    } catch (e) {
      if (!done) { clearTimeout(timer); done=true; resolve({ok:false, data:e}); }
    }
  });
}

// ====== Consulta de roster (Socket primero, REST fallback a /api/rooms) ======
async function queryRosterViaSocket(): Promise<boolean> {
  const candidates = ["getRoom","room:get","getRoomState","room:state","getPlayers","players:list","listPlayers"];
  for (const ev of candidates) {
    const { ok, data } = await emitAckRaw(ev, { roomId: ROOM_ID });
    if (!ok || !data) continue;

    const players = data.players || data.participants || data.room?.players || data.room?.participants || data;
    const arr = Array.isArray(players) ? players : [];
    if (!arr.length) continue;

    const prev = new Set(roster);
    roster.clear(); weReady = false; oppReady = false;

    for (const p of arr) {
      const pid = p?.id ?? p?.playerId ?? p?.username ?? p?.name ?? (typeof p === "string" ? p : "");
      if (!pid) continue;
      roster.add(pid);
      const isReady =
        Boolean(p?.ready || p?.isReady) ||
        (typeof p?.state === "string" && p.state.toUpperCase().includes("READY"));
      if (pid === MY_ID && isReady) weReady = true;
      if (pid !== MY_ID && isReady) oppReady = true;
    }
    if (JSON.stringify([...prev]) !== JSON.stringify([...roster])) {
      console.log(`[ai] roster(${ev}) →`, [...roster], "| weReady:", weReady, "oppReady:", oppReady);
    }
    return true;
  }
  return false;
}

async function queryRosterViaRest(): Promise<void> {
  try {
    // En tu server hay “listar todas las salas” (GetAllRooms) montado en /api (RoomController) :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}
    const { data } = await axios.get(`${API_URL}/api/rooms`);
    const rooms = Array.isArray(data) ? data : (Array.isArray(data?.rooms) ? data.rooms : []);
    const room = rooms.find((r: any) => (r.id ?? r.Id ?? r.roomId) === ROOM_ID);
    if (!room) return;

    const players = room.Players ?? room.players ?? room.participants ?? [];
    const prev = new Set(roster);
    roster.clear(); weReady = false; oppReady = false;

    for (const p of players) {
      const pid = p?.id ?? p?.playerId ?? p?.username ?? p?.name ?? "";
      if (!pid) continue;
      roster.add(pid);
      const isReady =
        Boolean(p?.ready || p?.isReady) ||
        (typeof p?.state === "string" && p.state.toUpperCase().includes("READY"));
      if (pid === MY_ID && isReady) weReady = true;
      if (pid !== MY_ID && isReady) oppReady = true;
    }
    if (JSON.stringify([...prev]) !== JSON.stringify([...roster])) {
      console.log(`[ai] roster(REST) →`, [...roster], "| weReady:", weReady, "oppReady:", oppReady);
    }
  } catch (e: any) {
    // Si tu RoomController no expone GET /api/rooms, ignora (pero en tus casos de uso sí existe listAll) :contentReference[oaicite:7]{index=7}
    // console.warn("[ai] GET /api/rooms falló:", e?.message ?? e);
  }
}

// ====== Eventos de sala ======
socket.on("playerJoined", (p: any) => {
  if (p?.id) roster.add(p.id);
  console.log("[ai] playerJoined:", p?.id, "| roster:", [...roster]);
  maybeStartBattle();
});
socket.on("playerLeft", (p: any) => {
  if (p?.id) roster.delete(p.id);
  console.log("[ai] playerLeft:", p?.id, "| roster:", [...roster]);
});
socket.on("playerReady", (p: any) => {
  if (!p?.playerId) return;
  if (p.playerId === MY_ID) weReady = true; else oppReady = true;
  console.log("[ai] playerReady recv:", p.playerId, "| weReady:", weReady, "oppReady:", oppReady);
  maybeStartBattle();
});

// ====== Conexión y handshake ======
socket.on("connect", async () => {
  console.log("[ai] conectado:", socket.id);

  // Crear sala si no existe (idempotente)
  try {
    await axios.post(`${API_URL}/api/rooms`, {
      id: ROOM_ID, mode:"1v1", allowAI:false, credits:100, heroLevel:HERO_STATS.hero.level, ownerId:"ownerB"
    }); // CreateRoom use-case :contentReference[oaicite:8]{index=8}
  } catch {}

  // Unirse por socket + REST (alineado con cliente 1)
  socket.emit("joinRoom", { roomId: ROOM_ID, player: { id: MY_ID, heroLevel: HERO_STATS.hero.level } }); // JoinRoom use-case :contentReference[oaicite:9]{index=9}
  try {
    await axios.post(`${API_URL}/api/rooms/${ROOM_ID}/join`, {
      playerId: MY_ID, heroLevel: HERO_STATS.hero.level, heroStats: HERO_STATS
    });
  } catch {}

  // Stats y listo (no esperar ACK del ready)
  socket.emit("setHeroStats", { roomId: ROOM_ID, playerId: MY_ID, stats: HERO_STATS }); // AssignHeroStats use-case :contentReference[oaicite:10]{index=10}
  socket.emit("playerReady",  { roomId: ROOM_ID, playerId: MY_ID, team: TEAM }); // SetPlayerReady use-case (devuelve allReady) :contentReference[oaicite:11]{index=11}
  weReady = true;

  // Consulta inicial de roster: socket → REST
  const gotViaSocket = await queryRosterViaSocket();
  if (!gotViaSocket) await queryRosterViaRest();
});

// Polling ligero para no depender de eventos históricos
setInterval(async () => {
  if (battleJoined) return;
  const gotViaSocket = await queryRosterViaSocket();
  if (!gotViaSocket) await queryRosterViaRest();
  maybeStartBattle();
}, 1500);

// ====== Arranque de batalla ======
function maybeStartBattle() {
  if (!battleJoined && roster.size >= 2 && weReady && oppReady) {
    battleJoined = true;
    console.log("[ai] ambos listos → joinBattle");
    socket.emit("joinBattle", { roomId: ROOM_ID, playerId: MY_ID }, (ack:any)=>{
      console.log("[ai] joinBattle ack:", ack ?? "(sin ack)");
    });
  }
}

// ====== Inicio de batalla / resolución ======
socket.on("battleStarted", async (data: any) => {
  // Asegurar unión a la batalla
  socket.emit("joinBattle", { roomId: ROOM_ID, playerId: MY_ID });

  turns = data?.turns || [];
  currentTurn = turns[0] || null;
  console.log("[ai] battleStarted, turns:", turns);

  if (currentTurn === MY_ID) await playTurnFromState(data);
});

socket.on("actionResolved", async (data: any) => {
  if (data?.state === "FINISHED" || data?.winner || data?.winningTeam) {
    console.log("⚑ Battle finished.", data?.winner ? `Winner: ${data.winner}` : "");
    return;
  }
  currentTurn = data?.nextTurnPlayer || currentTurn;
  if (currentTurn === MY_ID) await playTurnFromState(data);
});

// ====== Turno IA ======
async function playTurnFromState(payload: any) {
  const actor: ActorState = {
    heroType: heroKeyForAI(HERO_STATS.hero.heroType),
    level: HERO_STATS.hero.level,
    power: HERO_STATS.hero.power,
    attack: HERO_STATS.hero.attack,
    defense: HERO_STATS.hero.defense,
    health: HERO_STATS.hero.health,
  };
  const enemy: EnemyState = {
    health: Number(payload?.enemy_hp_after ?? payload?.enemy?.hp ?? 0),
    defense: Number(payload?.enemy?.defense ?? 0),
  };

  await delay(jitter(300, 600));
  let ai = await askAI({ actor, enemy });
  if (!ai) { await delay(150); ai = await askAI({ actor, enemy }); }

  const chosen: ActionKind = ai?.kind || localFallback(actor);
  const targetId = getOpponent();

  let serverAction: any;
  if (chosen === "BASIC_ATTACK") {
    serverAction = { type: "BASIC_ATTACK", sourcePlayerId: MY_ID, targetPlayerId: targetId };
  } else {
    const slotMap = SLOT_TO_SKILL[HERO_STATS.hero.heroType] || SLOT_TO_SKILL[heroKeyForAI(HERO_STATS.hero.heroType)];
    const skillIdStr = slotMap?.[chosen as Slot];
    serverAction = { type: "SPECIAL_SKILL", sourcePlayerId: MY_ID, targetPlayerId: targetId, skillId: skillIdStr };
  }

  await delay(jitter(120, 280));
  socket.emit("submitAction", { roomId: ROOM_ID, action: serverAction }, (ack:any)=>{
    console.log("[ai] => submitAction", serverAction.type, serverAction.skillId ? `(${serverAction.skillId})` : "", "| ack:", ack ?? "(sin ack)");
  });
}

// ====== Errores ======
socket.on("error", (err) => console.error("[ai] socket error:", err));

// ====== Bootstrap ======
(async () => {
  // Crea la sala si no existe (idempotente). El router está montado en /api (index.ts del server) :contentReference[oaicite:12]{index=12}
  try {
    await axios.post(`${API_URL}/api/rooms`, {
      id: ROOM_ID, mode:"1v1", allowAI:false, credits:100, heroLevel:HERO_STATS.hero.level, ownerId:"ownerB",
    });
  } catch {}
  // El resto ocurre en 'connect'
})();


/*# En la carpeta del proyecto
$env:SOCKET_URL="http://localhost:3000"
$env:API_URL="http://localhost:3000"
$env:AI_URL="http://127.0.0.1:8000"
$env:ROOM_ID="ZZZ000"
$env:MY_ID="playerB"
$env:TEAM="B"
npm run dev
*/
