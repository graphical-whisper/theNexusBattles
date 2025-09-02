// index.ts (cliente 1 - automático)
import axios from "axios";
import { io } from "socket.io-client";
import { startMatchIfEligible, recordDecision, recordOutcome } from "./dataset_logger";

// ===== Config =====
type ActionType = "BASIC_ATTACK" | "SPECIAL_SKILL" | "MASTER_SKILL";
const API_URL     = "http://localhost:3000";
const SOCKET_URL  = "http://localhost:3000";
const ROOM_ID     = "ZZZ000";
const MY_ID       = "playerA";               // Cliente 1
const MY_TEAM     = "A";
const MATCH_ID    = `${ROOM_ID}-${Date.now()}`;
const EXTRA_DELAY_MS = 1000;                 // Espera obligatoria

// ===== Patrón cíclico =====
// Tokens válidos: "BASIC", "SPECIAL:<ID o nombre>", "MASTER:<ID o nombre>"

const PATTERN: string[] = [
  "SPECIAL:CORTADA",
  //"MASTER:MASTER.ICE_FRIO_CONCENTRADO",
  "BASIC",
];

// ===== Listas de habilidades (IDs y nombres amigables) =====
const ALL_SPECIALS: { id: string; name: string }[] = [
  // Tank
  { id: "GOLPE_ESCUDO", name: "Golpe con escudo" },
  { id: "MANO_PIEDRA", name: "Mano de piedra" },
  { id: "DEFENSA_FEROZ", name: "Defensa feroz" },
  // Warrior Arms
  { id: "EMBATE_SANGRIENTO", name: "Embate sangriento" },
  { id: "LANZA_DIOSES", name: "Lanza de los dioses" },
  { id: "GOLPE_TORMENTA", name: "Golpe de tormenta" },
  // Mage Fire
  { id: "MISILES_MAGMA", name: "Misiles de magma" },
  { id: "VULCANO", name: "Vulcano" },
  { id: "PARED_FUEGO", name: "Pared de fuego" },
  // Mage Ice
  { id: "LLUVIA_HIELO", name: "Lluvia de hielo" },
  { id: "CONO_HIELO", name: "Cono de hielo" },
  { id: "BOLA_HIELO", name: "Bola de hielo" },
  // Rogue Poison
  { id: "FLOR_LOTO", name: "Flor de loto" },
  { id: "AGONIA", name: "Agonía" },
  { id: "PIQUETE", name: "Piquete" },
  // Rogue Machete
  { id: "CORTADA", name: "Cortada" },
  { id: "MACHETAZO", name: "Machetazo" },
  { id: "PLANAZO", name: "Planazo" },
  // Shaman
  { id: "TOQUE_VIDA", name: "Toque de la vida" },
  { id: "VINCULO_NATURAL", name: "Vínculo natural" },
  { id: "CANTO_BOSQUE", name: "Canto del bosque" },
  // Medic
  { id: "CURACION_DIRECTA", name: "Curación directa" },
  { id: "NEUTRALIZACION_EFECTOS", name: "Neutralización de efectos" },
  { id: "REANIMACION", name: "Reanimación" },
];

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

// ===== Utilidades de nombres→ID (idéntico flujo al cliente 1 manual) =====
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

// ===== Héroe (mismo esquema del cliente 1) =====
const HERO_STATS = {
  hero: {
    heroType: "MACHETE_ROGUE",
    level: 2,
    power: 8 * 2,
    health: 36 * 2,
    defense: 8 * 2,
    attack: 10 * 2,
    attackBoost: { min: 1, max: 10 },
    damage: { min: 1, max: 8 },


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
    items: [], armors: [], weapons: [],
    epicAbilites: ALL_MASTERS.map(m => ({
      name: m.name,
      compatibleHeroType: "WARRIOR_ARMS",
      effects: [],
      cooldown: 0,
      isAvailable: true,
      masterChance: 0.1,
    })),
  },
};

// ===== Estado local =====
const socket = io(SOCKET_URL);
let turns: string[] = [];
let currentTurn: string | null = null;
let finished = false;
let lastPayload: any = null;
let turnIndex = 0;
let patternIdx = 0;

const delay = (ms: number) => new Promise(res => setTimeout(res, ms));
const extractBattle = (payload: any) => payload?.battle ?? payload;
const otherPlayerId = () => turns.find(u => u !== MY_ID) || "";

// ===== Emit =====
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

// ===== Turno automático =====
async function actAutomatically() {
  if (finished || currentTurn !== MY_ID) return;

  // Espera adicional obligatoria
  await delay(EXTRA_DELAY_MS);

  const targetId = otherPlayerId();
  const token = (PATTERN[patternIdx % PATTERN.length] ?? "BASIC").trim();
  patternIdx++;

  const upper = token.toUpperCase();
  let chosenKind: ActionType;
  let chosenSkillId: string | undefined;

  if (upper === "BASIC") {
    chosenKind = "BASIC_ATTACK";
  } else if (upper.startsWith("SPECIAL:")) {
    chosenKind = "SPECIAL_SKILL";
    chosenSkillId = token.split(":")[1] || "";
  } else if (upper.startsWith("MASTER:")) {
    chosenKind = "MASTER_SKILL";
    chosenSkillId = token.split(":")[1] || "";
  } else {
    chosenKind = "BASIC_ATTACK";
  }

  // Registrar en dataset + enviar acción
  if (chosenKind === "SPECIAL_SKILL") {
    recordDecision(ROOM_ID, MATCH_ID, turnIndex, lastPayload, MY_ID, {
      kind: "SPECIAL_SKILL", skillId: toServerSkillId(chosenSkillId || "", "SPECIAL")
    });
    sendSpecial(chosenSkillId || "", targetId);
  } else if (chosenKind === "MASTER_SKILL") {
    recordDecision(ROOM_ID, MATCH_ID, turnIndex, lastPayload, MY_ID, {
      kind: "MASTER_SKILL", skillId: toServerSkillId(chosenSkillId || "", "MASTER")
    });
    sendMaster(chosenSkillId || "", targetId);
  } else {
    recordDecision(ROOM_ID, MATCH_ID, turnIndex, lastPayload, MY_ID, { kind: "BASIC_ATTACK" });
    sendBasic(targetId);
  }
}

// ===== Logs útiles =====
function printQuickPlayers(payload: any) {
  const battle = extractBattle(payload);
  if (!battle?.players && !battle?.teams) return;
  const players = battle.players ?? battle.teams?.flatMap((t: any) => t.players) ?? [];
  console.log("=== Players / Quick Stats ===");
  for (const p of players) {
    const h = p?.heroStats?.hero; if (!h) continue;
    console.log(`- ${p.username} :: HP=${h.health} POW=${h.power} ATK=${h.attack} DEF=${h.defense} POWER=${h.power}`);
  }
}
async function printRaw(label: string, payload: any) {
  console.log(`\n--- RAW ${label} :: actionResolved payload ---`);
  console.dir(payload, { depth: null });
  printQuickPlayers(payload);
}

// ===== Sockets =====
function wireSocket() {
  socket.on("connect", async () => {
    console.log("Socket connected:", socket.id);
    // Crear sala (si ya existe, se ignora el error) — igual que el cliente 1 manual
    await axios.post(`${API_URL}/api/rooms`, {
      id: ROOM_ID, mode: "1v1", allowAI: false, credits: 100, heroLevel: 1, ownerId: "ownerA",
    }).catch(() => {});

    socket.emit("joinRoom", { roomId: ROOM_ID, player: { id: MY_ID, heroLevel: 1 } });
    await axios.post(`${API_URL}/api/rooms/${ROOM_ID}/join`, { playerId: MY_ID, heroLevel: 1, heroStats: HERO_STATS }).catch(() => {});
    socket.emit("setHeroStats", { roomId: ROOM_ID, playerId: MY_ID, stats: HERO_STATS });
    socket.emit("playerReady", { roomId: ROOM_ID, playerId: MY_ID, team: MY_TEAM });
  });

  socket.on("battleStarted", async (data: any) => {
    socket.emit("joinBattle", { roomId: ROOM_ID, playerId: MY_ID });
    console.log("Battle started:", data?.turns);
    turns = data?.turns || [];
    currentTurn = turns[0] || null;
    finished = false;
    lastPayload = data;

    const ok = startMatchIfEligible(ROOM_ID, MATCH_ID, data, MY_ID);
    if (!ok.ok) console.log("Dataset: partida no elegible:", ok.reason);

    await printRaw("battleStarted", data);
    if (currentTurn === MY_ID) actAutomatically();
  });

  socket.on("actionResolved", async (data: any) => {
    console.log("\n=== Action resolved ===");
    await printRaw("actionResolved", data);

    // Outcome para la decisión previa
    recordOutcome(data);

    // Final de batalla
    if (data?.state === "FINISHED" || data?.winner || data?.winningTeam) {
      finished = true;
      console.log("⚑ Battle finished.", data?.winner ? `Winner: ${data.winner}` : "");
      return;
    }

    // Avance de turno
    const before = currentTurn;
    currentTurn = data?.nextTurnPlayer || currentTurn;
    lastPayload = data;
    if (currentTurn === MY_ID && before !== MY_ID) turnIndex += 1;

    if (currentTurn === MY_ID) actAutomatically();
  });

  socket.on("battleEnded", (data: any) => { finished = true; console.log("⚑ battleEnded:", data); });
  socket.on("error", (err) => console.error("Socket error:", err));
}

// --- Run ---
wireSocket();
