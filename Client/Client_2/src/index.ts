import axios from "axios";
import { io } from "socket.io-client";
import { Readline } from "./readline";
// NUEVO:
import { startMatchIfEligible, recordDecision, recordOutcome } from "./dataset_logger";



type ActionType = "BASIC_ATTACK" | "SPECIAL_SKILL" | "MASTER_SKILL";

const API_URL = "http://localhost:3000";
const SOCKET_URL = "http://localhost:3000";
const ROOM_ID = "ZZZ000";
const MY_ID = "playerB";
let lastPayload: any = null;
let turnIndex = 0;
const MATCH_ID = `${ROOM_ID}-${Date.now()}`;

const readline = new Readline();
const socket = io(SOCKET_URL);

/** ---------- Specials ---------- */
const ALL_SPECIALS: { id: string; name: string }[] = [
  { id: "GOLPE_ESCUDO", name: "Golpe con escudo" },
  { id: "MANO_PIEDRA", name: "Mano de piedra" },
  { id: "DEFENSA_FEROZ", name: "Defensa feroz" },
  { id: "EMBATE_SANGRIENTO", name: "Embate sangriento" },
  { id: "LANZA_DIOSES", name: "Lanza de los dioses" },
  { id: "GOLPE_TORMENTA", name: "Golpe de tormenta" },
  { id: "MISILES_MAGMA", name: "Misiles de magma" },
  { id: "VULCANO", name: "Vulcano" },
  { id: "PARED_FUEGO", name: "Pared de fuego" },
  { id: "LLUVIA_HIELO", name: "Lluvia de hielo" },
  { id: "CONO_HIELO", name: "Cono de hielo" },
  { id: "BOLA_HIELO", name: "Bola de hielo" },
  { id: "FLOR_LOTO", name: "Flor de loto" },
  { id: "AGONIA", name: "Agonía" },
  { id: "PIQUETE", name: "Piquete" },
  { id: "CORTADA", name: "Cortada" },
  { id: "MACHETAZO", name: "Machetazo" },
  { id: "PLANAZO", name: "Planazo" },
  { id: "TOQUE_VIDA", name: "Toque de la vida" },
  { id: "VINCULO_NATURAL", name: "Vínculo natural" },
  { id: "CANTO_BOSQUE", name: "Canto del bosque" },
  { id: "CURACION_DIRECTA", name: "Curación directa" },
  { id: "NEUTRALIZACION_EFECTOS", name: "Neutralización de efectos" },
  { id: "REANIMACION", name: "Reanimación" },
];

/** ---------- Masters ---------- */
const ALL_MASTERS: { id: string; name: string }[] = [
  { id: "MASTER.TANK_GOLPE_DEFENSA", name: "Golpe de Defensa" },
  { id: "MASTER.ARMS_SEGUNDO_IMPULSO", name: "Segundo Impulso" },
  { id: "MASTER.FIRE_LUZ_CEGADORA", name: "Luz Cegadora" },
  { id: "MASTER.ICE_FRIO_CONCENTRADO", name: "Frío Concentrado" },
  { id: "MASTER.VENENO_TOMA_LLEVA", name: "Toma y Lleva" },
  { id: "MASTER.MACHETE_INTIMIDACION_SANGRIENTA", name: "Intimidación Sangrienta" },
  { id: "MASTER.SHAMAN_TE_CHANGUA", name: "Té Changua" },
  { id: "MASTER.MEDIC_REANIMADOR_3000", name: "Reanimador 3000" },
];

const SPECIAL_NAME_TO_ID = Object.fromEntries(ALL_SPECIALS.map(s => [normalizeKey(s.name), s.id]));
const MASTER_NAME_TO_ID  = Object.fromEntries(ALL_MASTERS.map(m => [normalizeKey(m.name), m.id]));
const VALID_SPECIAL_IDS  = new Set(ALL_SPECIALS.map(s => s.id));
const VALID_MASTER_IDS   = new Set(ALL_MASTERS.map(m => m.id));

function normalizeKey(s: string) {
  return (s || "").normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase().trim();
}
function toServerSkillId(input: string, type: "SPECIAL" | "MASTER"): string {
  const raw = (input || "").trim();
  if (!raw) return raw;
  const id = raw.toUpperCase().replace(/\s+/g, "_");
  if (type === "SPECIAL") {
    if (VALID_SPECIAL_IDS.has(id)) return id;
    const mapped = SPECIAL_NAME_TO_ID[normalizeKey(raw)];
    return mapped || raw;
  } else {
    if (VALID_MASTER_IDS.has(id)) return id;
    const mapped = MASTER_NAME_TO_ID[normalizeKey(raw)];
    return mapped || raw;
  }
}

/** ---------- Héroe pruebas ---------- */
const HERO_STATS = {
  hero: {
    heroType: "WEAPONS_PAL",
    level: 1,
    power: 8 * 1,
    health: 44 * 1,
    defense: 11 * 1,
    attack: 10 * 1,
    attackBoost: { min: 1, max: 6 },  // 10 + 1d6
    damage:      { min: 1, max: 6 },  // 1d6


    specialActions: ALL_SPECIALS.map(s => ({
      name: s.name,
      actionType: "ATTACK",
      powerCost: 1,
      cooldown: 0,
      isAvailable: true,
      effect: [],
    })),
    randomEffects: [
      { randomEffectType: "DAMAGE",        percentage: 55, valueApply: { min: 0, max: 0 } },
      { randomEffectType: "CRITIC_DAMAGE", percentage: 10, valueApply: { min: 2, max: 4 } },
      { randomEffectType: "EVADE",         percentage: 5,  valueApply: { min: 0, max: 0 } },
      { randomEffectType: "RESIST",        percentage: 10, valueApply: { min: 0, max: 0 } },
      { randomEffectType: "ESCAPE",        percentage: 0,  valueApply: { min: 0, max: 0 } },
      { randomEffectType: "NEGATE",        percentage: 20, valueApply: { min: 0, max: 0 } },
    ],
  },
  equipped: {
    items: [],
    armors: [],
    weapons: [],
    epicAbilites: ALL_MASTERS.map(m => ({
      name: m.name,
      compatibleHeroType: "TANK",
      effects: [],
      cooldown: 0,
      isAvailable: true,
      masterChance: 0.1,
    })),
  },
};

/** ---------- Estado local y menús ---------- */
let turns: string[] = [];
let currentTurn: string | null = null;
let finished = false;

function otherPlayerId(): string {
  return turns.find(u => u !== MY_ID) || "";
}
function resolveTargetId(raw: string): string {
  const input = (raw ?? "").trim();
  if (!input) return otherPlayerId();
  const hitExact = turns.find(u => u === input);
  if (hitExact) return hitExact;
  const hitCI = turns.find(u => u.toLowerCase() === input.toLowerCase());
  return hitCI || otherPlayerId();
}
function renderSpecialsMenu() {
  return ALL_SPECIALS.map((s, i) => `${i + 1}. ${s.id}  (${s.name})`).join("\n");
}
function renderMastersMenu() {
  return ALL_MASTERS.map((m, i) => `${i + 1}. ${m.id}  (${m.name})`).join("\n");
}

/** ---------- Emit ---------- */
function sendBasic(targetId: string) {
  const action = { type: "BASIC_ATTACK" as ActionType, sourcePlayerId: MY_ID, targetPlayerId: targetId };
  socket.emit("submitAction", { roomId: ROOM_ID, action });
}
function sendSpecial(input: string, targetId: string) {
  const skillId = toServerSkillId(input, "SPECIAL");
  const action = { type: "SPECIAL_SKILL" as ActionType, sourcePlayerId: MY_ID, targetPlayerId: targetId, skillId };
  console.log(`[SEND] SPECIAL_SKILL skillId=${skillId}`);
  socket.emit("submitAction", { roomId: ROOM_ID, action });
}
function sendMaster(input: string, targetId: string) {
  const skillId = toServerSkillId(input, "MASTER");
  const action = { type: "MASTER_SKILL" as ActionType, sourcePlayerId: MY_ID, targetPlayerId: targetId, skillId };
  console.log(`[SEND] MASTER_SKILL skillId=${skillId}`);
  socket.emit("submitAction", { roomId: ROOM_ID, action });
}

/** ---------- Prompt ---------- */
async function promptAndSend() {
  if (finished || currentTurn !== MY_ID) return;

  console.log(`\n>>> YOUR TURN (${MY_ID})`);
  const kind = (await readline.pregunta('Action ("BASIC" | "SPECIAL" | "MASTER"): ')).trim().toUpperCase();
  const target = resolveTargetId(await readline.pregunta(`Target (default: ${otherPlayerId()}): `));

  if (kind === "SPECIAL") {
    console.log("\n== SPECIALS ==\n" + renderSpecialsMenu() + "\n");
    const pick = (await readline.pregunta("Pick (number | name | ID): ")).trim();
    const n = parseInt(pick, 10);
    const chosen = (!Number.isNaN(n) && n >= 1 && n <= ALL_SPECIALS.length) ? ALL_SPECIALS[n - 1]!.id : pick;

    // NUEVO: registrar decisión
    recordDecision(ROOM_ID, MATCH_ID, turnIndex, lastPayload, MY_ID, { kind: "SPECIAL_SKILL", skillId: toServerSkillId(chosen, "SPECIAL") });

    sendSpecial(chosen, target);
  } else if (kind === "MASTER") {
    console.log("\n== MASTERS ==\n" + renderMastersMenu() + "\n");
    const pick = (await readline.pregunta("Pick (number | name | ID): ")).trim();
    const n = parseInt(pick, 10);
    const chosen = (!Number.isNaN(n) && n >= 1 && n <= ALL_MASTERS.length) ? ALL_MASTERS[n - 1]!.id : pick;

    recordDecision(ROOM_ID, MATCH_ID, turnIndex, lastPayload, MY_ID, { kind: "MASTER_SKILL", skillId: toServerSkillId(chosen, "MASTER") });

    sendMaster(chosen, target);
  } else {
    recordDecision(ROOM_ID, MATCH_ID, turnIndex, lastPayload, MY_ID, { kind: "BASIC_ATTACK" });
    sendBasic(target);
  }
}

/** ---------- Sockets ---------- */
function extractBattle(payload: any) {
  return payload?.battle ?? payload;
}
function printQuickPlayers(payload: any) {
  const battle = extractBattle(payload);
  if (!battle?.players && !battle?.teams) return;
  const players = battle.players ?? battle.teams?.flatMap((t: any) => t.players) ?? [];
  console.log("=== Players / Quick Stats ===");
  for (const p of players) {
    const h = p?.heroStats?.hero;
    if (!h) continue;
    console.log(`- ${p.username} :: HP=${h.health} POW=${h.power} ATK=${h.attack} DEF=${h.defense} POWER=${h.power}`);
  }
}
async function printRaw(label: string, payload: any) {
  console.log(`\n--- RAW ${label} :: actionResolved payload ---`);
  console.dir(payload, { depth: null });
  printQuickPlayers(payload);
}

function wireSocket() {
  socket.on("connect", async () => {
    console.log("Socket connected:", socket.id);
    socket.emit("joinRoom", { roomId: ROOM_ID, player: { id: MY_ID, heroLevel: 1 } });
    await axios.post(`${API_URL}/api/rooms/${ROOM_ID}/join`, { playerId: MY_ID, heroLevel: 1, heroStats: HERO_STATS }).catch(() => {});
    socket.emit("setHeroStats", { roomId: ROOM_ID, playerId: MY_ID, stats: HERO_STATS });
    socket.emit("playerReady", { roomId: ROOM_ID, playerId: MY_ID, team: "B" });
  });

  /*
  socket.on("battleStarted", async (data: any) => {
    socket.emit("joinBattle", { roomId: ROOM_ID, playerId: MY_ID });
    console.log("Battle started:", data?.turns);
    turns = data?.turns || [];
    currentTurn = turns[0] || null;
    finished = false;
    await printRaw("battleStarted", data);
    if (currentTurn === MY_ID) promptAndSend();
  });
  */
  socket.on("battleStarted", async (data: any) => {
    socket.emit("joinBattle", { roomId: ROOM_ID, playerId: MY_ID });
    console.log("Battle started:", data?.turns);
    turns = data?.turns || [];
    currentTurn = turns[0] || null;
    finished = false;
    //agregado
    lastPayload = data;
      const ok = startMatchIfEligible(ROOM_ID, MATCH_ID, data, MY_ID);
  if (!ok.ok) console.log("Dataset: partida no elegible:", ok.reason);
    await printRaw("battleStarted", data);
    if (currentTurn === MY_ID) promptAndSend();
  });
/*
  socket.on("actionResolved", async (data: any) => {
    console.log("\n=== Action resolved (B) ===");
    await printRaw("actionResolved(B)", data);

    if (data?.state === "FINISHED" || data?.winner || data?.winningTeam) {
      finished = true;
      console.log("⚑ Battle finished.", data?.winner ? `Winner: ${data.winner}` : "");
      return;
    }
    currentTurn = data?.nextTurnPlayer || currentTurn;
    promptAndSend();
  });
  */
 socket.on("actionResolved", async (data: any) => {
  console.log("\n=== Action resolved ===");
  await printRaw("actionResolved", data);

  // NUEVO: outcome para la decisión previamente registrada (si la hubo)
  recordOutcome(data);

  // fin de batalla o siguiente turno
  if (data?.state === "FINISHED" || data?.winner || data?.winningTeam) {
    finished = true;
    console.log("⚑ Battle finished.", data?.winner ? `Winner: ${data.winner}` : "");
    return;
  }
  // avanza turno (si el siguiente es mío, aumenta índice local)
  const before = currentTurn;
  currentTurn = data?.nextTurnPlayer || currentTurn;
  lastPayload = data; // NUEVO
  if (currentTurn === MY_ID && before !== MY_ID) turnIndex += 1;

  if (currentTurn === MY_ID) promptAndSend();
});


  socket.on("battleEnded", (data: any) => { finished = true; console.log("⚑ battleEnded:", data); });
  socket.on("error", (err) => console.error("Socket error:", err));
}

// --- Run ---
(async () => {
  // A ya creó la sala
  wireSocket();
})();
