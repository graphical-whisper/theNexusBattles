import axios from "axios";
import { io } from "socket.io-client";
// Importar las funciones necesarias
import { startMatchIfEligible, recordDecision, recordOutcome } from "./dataset_logger";

type ActionType = "BASIC_ATTACK" | "SPECIAL_SKILL" | "MASTER_SKILL";

const API_URL = "http://localhost:3000";
const SOCKET_URL = "http://localhost:3000";
const ROOM_ID = "ZZZ000b";
const MY_ID = "playerB";
let lastPayload: any = null; // Asegúrate de que esta variable sea usada
let turnIndex = 0;
const MATCH_ID = `${ROOM_ID}-${Date.now()}`;

// Crear la conexión con Socket.IO
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
    heroType: "ICE_MAGE",
    level: 1,
    power: 10 * 1,
    health: 40 * 1,
    defense: 10 * 1,
    attack: 10 * 1,
    attackBoost: { min: 1, max: 8 },  // 10 + 1d8
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

async function promptAndSend() {
  if (finished || currentTurn !== MY_ID) return;

  console.log(`\n>>> YOUR TURN (${MY_ID})`);
  const target = resolveTargetId(otherPlayerId());  // Target opponent automatically

  const actions = [
    { type: "SPECIAL_SKILL" as ActionType, skillId: "GOLPE_ESCUDO" }, // Habilidad 1
    { type: "SPECIAL_SKILL" as ActionType, skillId: "MANO_PIEDRA" },  // Habilidad 2
  ];

  // Bucle hasta que la batalla termine
  for (let i = 0; i < actions.length; i++) {
    if (finished) {
      console.log("Batalla terminada, no se enviarán más habilidades.");
      break;  // Salimos del bucle si la batalla ya ha terminado
    }

    const action = actions[i];
    if (!action || !action.skillId) continue;

    const powerCost = (HERO_STATS.hero?.specialActions?.find(s => s.name === action.skillId)?.powerCost) ?? 0;

    // Ejecutar habilidad o ataque básico si se tiene suficiente poder
    if (HERO_STATS.hero && HERO_STATS.hero.power >= powerCost) {
      console.log(`Ejecutando habilidad: ${action.skillId}`);
      HERO_STATS.hero.power -= powerCost;

      if (action.type === "SPECIAL_SKILL") {
        sendSpecial(action.skillId, target);
      } else if (action.type === "MASTER_SKILL") {
        sendMaster(action.skillId, target);
      } else {
        sendBasic(target);
      }

    } else {
      // Si no hay suficiente poder, ejecuta un ataque básico
      console.log("No hay suficiente poder, ejecutando ataque básico.");
      sendBasic(target);
    }

    // Espera siempre 5 segundos entre habilidades
    console.log(`Esperando 5 segundos antes de la siguiente habilidad...`);
    await new Promise(resolve => setTimeout(resolve, 5000));  // Espera 5 segundos

    // Si ya hemos llegado al final de las habilidades, volver al principio
    if (i + 1 >= actions.length) {
      console.log("Se acabaron las habilidades. Volviendo al inicio.");
      i = -1;  // Esto hará que vuelva a la primera habilidad
    }
  }
  console.log("Batalla terminada, no se enviarán más habilidades.");
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
    if (currentTurn === MY_ID) promptAndSend();
  });

  socket.on("actionResolved", async (data: any) => {
    console.log("\n=== Action resolved ===");
    await printRaw("actionResolved", data);

    // Registrar el resultado de la acción si es necesario
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
    lastPayload = data; // Actualización de lastPayload
    if (currentTurn === MY_ID && before !== MY_ID) turnIndex += 1;

    if (currentTurn === MY_ID) promptAndSend();
  });

  socket.on("battleEnded", (data: any) => { finished = true; console.log("⚑ battleEnded:", data); });
  socket.on("error", (err) => console.error("Socket error:", err));
}

// --- Run ---
(async () => {
  wireSocket();
})();
