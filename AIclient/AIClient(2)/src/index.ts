// aiclient/index.ts (AI client basado en cliente 2: NO crea sala)
// Instalar npm i axios socket.io-client @tensorflow/tfjs-node
import axios from "axios";
import { io } from "socket.io-client";
import * as tf from "@tensorflow/tfjs-node";
import * as fs from "fs";
import * as path from "path";

// Reutiliza tu logger y catálogo 
import { startMatchIfEligible, recordDecision, recordOutcome } from "../dataset_logger";
import { SPECIAL_KIND, MASTER_KIND } from "../ai_action_catalog";

type ActionType = "BASIC_ATTACK" | "SPECIAL_SKILL" | "MASTER_SKILL";

// ===== Config =====
const API_URL     = process.env['API_URL']     || "http://localhost:3000";
const SOCKET_URL  = process.env['SOCKET_URL']  || "http://localhost:3000";
const ROOM_ID     = process.env['ROOM_ID']     || "ZZZ000";
const MY_ID       = process.env['MY_ID']       || "playerB";
const MY_TEAM     = process.env['MY_TEAM']     || "B";
const EXTRA_DELAY_MS = Number(process.env['EXTRA_DELAY_MS'] || 5000);

const DATA_DIR    = process.env['DATA_DIR']    || path.resolve("data");
const MERGED_FILE = process.env['MERGED_FILE'] || path.join(DATA_DIR, "merged_turns.ndjson"); 
const MODEL_DIR   = process.env['MODEL_DIR']   || path.resolve("models/policy_ai");

const socket = io(SOCKET_URL);
let turns: string[] = [];
let currentTurn: string | null = null;
let finished = false;
let lastPayload: any = null;
let turnIndex = 0;
const MATCH_ID = `${ROOM_ID}-${Date.now()}`;

const delay = (ms: number) => new Promise(res => setTimeout(res, ms));
const otherPlayerId = () => turns.find(u => u !== MY_ID) || "";

// ===== Habilidades conocidas =====
const ALL_SPECIALS = [
  "GOLPE_ESCUDO","MANO_PIEDRA","DEFENSA_FEROZ",
  "EMBATE_SANGRIENTO","LANZA_DIOSES","GOLPE_TORMENTA",
  "MISILES_MAGMA","VULCANO","PARED_FUEGO",
  "LLUVIA_HIELO","CONO_HIELO","BOLA_HIELO",
  "FLOR_LOTO","AGONIA","PIQUETE",
  "CORTADA","MACHETAZO","PLANAZO",
  "TOQUE_VIDA","VINCULO_NATURAL","CANTO_BOSQUE",
  "CURACION_DIRECTA","NEUTRALIZACION_EFECTOS","REANIMACION",
] as const;

const ALL_MASTERS = [
  "MASTER.TANK_GOLPE_DEFENSA","MASTER.ARMS_SEGUNDO_IMPULSO","MASTER.FIRE_LUZ_CEGADORA",
  "MASTER.ICE_FRIO_CONCENTRADO","MASTER.VENENO_TOMA_LLEVA","MASTER.MACHETE_INTIMIDACION_SANGRIENTA",
  "MASTER.SHAMAN_TE_CHANGUA","MASTER.MEDIC_REANIMADOR_3000",
] as const;

const VALID_SPECIAL_IDS = new Set(ALL_SPECIALS);
const VALID_MASTER_IDS  = new Set(ALL_MASTERS);
function toServerSkillId(input: string, type: "SPECIAL" | "MASTER"): string {
  const raw = (input || "").trim().toUpperCase().replace(/\s+/g, "_");
  return type === "SPECIAL" ? (VALID_SPECIAL_IDS.has(raw as any) ? raw : input)
                            : (VALID_MASTER_IDS.has(raw as any)  ? raw : input);
}

// ====== HERO TYPE one-hot + nivel (parches clave) ======
const HERO_TYPE_LIST = [
  "TANK","WEAPONS_PAL","FIRE_MAGE","ICE_MAGE",
  "POISON_ROGUE","MACHETE_ROGUE","SHAMAN","MEDIC"
] as const;
type HeroTypeStr = typeof HERO_TYPE_LIST[number];

function normalizeHeroTypeLive(raw: any): HeroTypeStr | "UNKNOWN" {
  if (typeof raw === "number") return (HERO_TYPE_LIST[raw] ?? "UNKNOWN") as any;
  if (typeof raw === "string") {
    const s = raw.toUpperCase().trim();
    return ((HERO_TYPE_LIST as readonly string[]).includes(s) ? s : "UNKNOWN") as any;
  }
  return "UNKNOWN";
}
function oneHotType(t: string | undefined): number[] {
  const v = (t ?? "UNKNOWN").toUpperCase();
  return HERO_TYPE_LIST.map(x => (x === v ? 1 : 0));
}

// ====== Features (alineadas con tu logger/NDJSON) ======
type Sample = { x: number[]; y: number }; // y: 0 BASIC, 1 SPECIAL, 2 MASTER
const LABEL_TO_Y: Record<string, number> = { BASIC_ATTACK:0, SPECIAL_SKILL:1, MASTER_SKILL:2 };
const safeNum = (v: any, d = 0) => (Number.isFinite(Number(v)) ? Number(v) : d);

// Dataset → vector con: tipos(8+8) + niveles(2) + base(13) = 31 features
function featuresFromRow(r: any): number[] {
  const actorTypeOH = oneHotType(r?.actor?.type);
  const enemyTypeOH = oneHotType(r?.enemy?.type);
  const actorLvl = Number(r?.actor?.level ?? 1);
  const enemyLvl = Number(r?.enemy?.level ?? 1);

  const base = [
    safeNum(r?.actor?.hp_pct), safeNum(r?.actor?.power),
    safeNum(r?.actor?.attack), safeNum(r?.actor?.defense),
    safeNum(r?.enemy?.hp_pct), safeNum(r?.enemy?.attack),
    safeNum(r?.enemy?.defense),
    safeNum(r?.num_specials_valid_off),
    safeNum(r?.num_specials_valid_def),
    safeNum(r?.num_support_valid),
    safeNum(r?.num_masters_valid_off),
    safeNum(r?.num_masters_valid_def),
    safeNum(r?.num_masters_support),
  ];
  return [...actorTypeOH, ...enemyTypeOH, actorLvl, enemyLvl, ...base];
}

// Live payload → MISMO ORDEN de features que arriba
function featuresFromLive(payload: any, myId: string) {
  const battle = payload?.battle ?? payload;
  const players = battle?.players ?? battle?.teams?.flatMap((t: any) => t.players) ?? [];
  const me  = players.find((p: any) => (p?.username || p?.id || p?.playerId) === myId);
  const foe = players.find((p: any) => (p?.username || p?.id || p?.playerId) !== myId);

  const hMe  = me?.heroStats?.hero;
  const hFoe = foe?.heroStats?.hero;

  // Disponibilidad real (server aplica restricciones por tipo/nivel/power/cooldown)
  const specialsAvail = (hMe?.specialActions ?? [])
    .filter((s: any) => !!s?.isAvailable && (s?.cooldown ?? 0) === 0 && (hMe?.power ?? 0) >= (s?.powerCost ?? 0))
    .map((s: any) => (s.id || s.name));

  const mastersAvail = (me?.heroStats?.equipped?.epicAbilites ?? [])
    .filter((m: any) => !!m?.isAvailable && (m?.cooldown ?? 0) === 0)
    .map((m: any) => (m.id || m.name));

  // Agregación por grupo (como en tu catálogo)
  const sAgg: Record<string, number> = {};
  const mAgg: Record<string, number> = {};
  specialsAvail.forEach((id: string) => { const g = SPECIAL_KIND[id]; sAgg[g] = (sAgg[g] || 0) + 1; });
  mastersAvail.forEach((id: string) => { const g = MASTER_KIND[id];  mAgg[g] = (mAgg[g] || 0) + 1; });

  // Tipo y nivel
  const meType   = normalizeHeroTypeLive(hMe?.type ?? hMe?.heroType);
  const foeType  = normalizeHeroTypeLive(hFoe?.type ?? hFoe?.heroType);
  const actorOH  = oneHotType(meType);
  const enemyOH  = oneHotType(foeType);
  const actorLvl = Number(hMe?.level ?? 1);
  const enemyLvl = Number(hFoe?.level ?? 1);

  // Numéricas base + máscara
  const base = [
    safeNum(hMe?.hp_pct ?? (hMe?.health ? hMe.health/40 : 0)),
    safeNum(hMe?.power),   safeNum(hMe?.attack),  safeNum(hMe?.defense),
    safeNum(hFoe?.hp_pct ?? (hFoe?.health ? hFoe.health/40 : 0)),
    safeNum(hFoe?.attack), safeNum(hFoe?.defense),
    safeNum(sAgg["offense"]), safeNum(sAgg["defense"]), safeNum(sAgg["support"]),
    safeNum(mAgg["offense"]), safeNum(mAgg["defense"]), safeNum(mAgg["support"]),
  ];

  const x = [...actorOH, ...enemyOH, actorLvl, enemyLvl, ...base];
  return { x, specialsAvail, mastersAvail };
}

// ====== Modelo: cargar o entrenar con merged_turns.ndjson ======
async function loadOrTrain(): Promise<tf.LayersModel> {
  const modelPath = path.join(MODEL_DIR, "model.json");
  if (fs.existsSync(modelPath)) {
    const m = await tf.loadLayersModel(`file://${modelPath}`);
    console.log("Modelo cargado:", MODEL_DIR);
    return m;
  }

  if (!fs.existsSync(MERGED_FILE)) {
    throw new Error(`No existe dataset mergeado: ${MERGED_FILE}`);
  }

  // El merged preserve turnos por sesión (gap/fin de batalla) e impone match_id y turn_index consistentes
  // ideal para entrenar sin fugas temporales. :contentReference[oaicite:4]{index=4}
  const lines = fs.readFileSync(MERGED_FILE, "utf-8").split("\n").map(s => s.trim()).filter(Boolean);
  const X: number[][] = []; const Y: number[] = [];
  for (const s of lines) {
    try {
      const r = JSON.parse(s);
      const y = LABEL_TO_Y[r?.chosen_action_kind];
      if (y === undefined) continue; // solo filas con etiqueta
      X.push(featuresFromRow(r)); Y.push(y);
    } catch { /* ignora */ }
  }
  if (X.length < 10) throw new Error(`Dataset insuficiente (X=${X.length}). Genera más partidas.`);

  const xT = tf.tensor2d(X);
  const yT = tf.oneHot(tf.tensor1d(Y, "int32"), 3);
  const INPUT_DIM = X[0].length; 

  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 64, activation: "relu", inputShape: [INPUT_DIM] }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 32, activation: "relu" }));
  model.add(tf.layers.dense({ units: 3, activation: "softmax" })); // BASIC / SPECIAL / MASTER
  model.compile({ optimizer: tf.train.adam(0.01), loss: "categoricalCrossentropy", metrics: ["accuracy"] });

  await model.fit(xT, yT, { epochs: 12, batchSize: 64, verbose: 1 });

  fs.mkdirSync(MODEL_DIR, { recursive: true });
  await model.save(`file://${MODEL_DIR}`);
  xT.dispose(); yT.dispose();
  console.log("Modelo entrenado y guardado en", MODEL_DIR);
  return model;
}

let MODEL: tf.LayersModel | null = null;

// ===== elegir skill concreta dentro del grupo =====
function pickSkill(kind: ActionType, specialsAvail: string[], mastersAvail: string[]) {
  if (kind === "SPECIAL_SKILL") {
    const pref = [
      ...specialsAvail.filter(id => SPECIAL_KIND[id] === "offense"),
      ...specialsAvail.filter(id => SPECIAL_KIND[id] === "defense"),
      ...specialsAvail.filter(id => SPECIAL_KIND[id] === "support"),
    ];
    if (pref[0]) return { kind, skillId: toServerSkillId(pref[0], "SPECIAL") as string };
  }
  if (kind === "MASTER_SKILL") {
    const pref = [
      ...mastersAvail.filter(id => MASTER_KIND[id] === "offense"),
      ...mastersAvail.filter(id => MASTER_KIND[id] === "defense"),
      ...mastersAvail.filter(id => MASTER_KIND[id] === "support"),
    ];
    if (pref[0]) return { kind, skillId: toServerSkillId(pref[0], "MASTER") as string };
  }
  return { kind: "BASIC_ATTACK" as ActionType };
}

// ===== Emit =====
function sendBasic(targetId: string) {
  const action = { type: "BASIC_ATTACK" as ActionType, sourcePlayerId: MY_ID, targetPlayerId: targetId };
  socket.emit("submitAction", { roomId: ROOM_ID, action });
}
function sendSpecial(input: string, targetId: string) {
  const action = { type: "SPECIAL_SKILL" as ActionType, sourcePlayerId: MY_ID, targetPlayerId: targetId, skillId: toServerSkillId(input, "SPECIAL") };
  console.log(`[SEND] SPECIAL_SKILL skillId=${action.skillId}`); socket.emit("submitAction", { roomId: ROOM_ID, action });
}
function sendMaster(input: string, targetId: string) {
  const action = { type: "MASTER_SKILL" as ActionType, sourcePlayerId: MY_ID, targetPlayerId: targetId, skillId: toServerSkillId(input, "MASTER") };
  console.log(`[SEND] MASTER_SKILL skillId=${action.skillId}`); socket.emit("submitAction", { roomId: ROOM_ID, action });
}

// ===== Turno con IA =====
async function actWithAI() {
  if (finished || currentTurn !== MY_ID) return;
  await delay(EXTRA_DELAY_MS);

  const targetId = otherPlayerId();
  const { x, specialsAvail, mastersAvail } = featuresFromLive(lastPayload, MY_ID);

  let predicted: ActionType = "BASIC_ATTACK";
  try {
    if (!MODEL) throw new Error("Modelo no cargado");
    const logits = MODEL.predict(tf.tensor2d([x])) as tf.Tensor;
    const [[pBasic, pSpec, pMaster]] = (await logits.array()) as number[][];
    logits.dispose();

    if (pSpec >= pBasic && pSpec >= pMaster && specialsAvail.length) predicted = "SPECIAL_SKILL";
    else if (pMaster >= pBasic && pMaster >= pSpec && mastersAvail.length) predicted = "MASTER_SKILL";
    else predicted = "BASIC_ATTACK";
  } catch {
    predicted = "BASIC_ATTACK";
  }

  const chosen = pickSkill(predicted, specialsAvail, mastersAvail);

  if (chosen.kind === "SPECIAL_SKILL") {
    recordDecision(ROOM_ID, MATCH_ID, turnIndex, lastPayload, MY_ID, { kind: "SPECIAL_SKILL", skillId: chosen.skillId });
    sendSpecial(chosen.skillId!, targetId);
  } else if (chosen.kind === "MASTER_SKILL") {
    recordDecision(ROOM_ID, MATCH_ID, turnIndex, lastPayload, MY_ID, { kind: "MASTER_SKILL", skillId: chosen.skillId });
    sendMaster(chosen.skillId!, targetId);
  } else {
    recordDecision(ROOM_ID, MATCH_ID, turnIndex, lastPayload, MY_ID, { kind: "BASIC_ATTACK" });
    sendBasic(targetId);
  }
}

// ===== Debug/ayudas =====
function extractBattle(payload: any) { return payload?.battle ?? payload; }
function printQuickPlayers(payload: any) {
  const battle = extractBattle(payload);
  const players = battle?.players ?? battle?.teams?.flatMap((t: any) => t.players) ?? [];
  console.log("=== Players / Quick Stats ===");
  for (const p of players) {
    const h = p?.heroStats?.hero; if (!h) continue;
    console.log(`- ${p.username} :: HP=${h.health} POW=${h.power} ATK=${h.attack} DEF=${h.defense}`);
  }
}
async function printRaw(label: string, payload: any) {
  console.log(`\n--- RAW ${label} ---`);
  console.dir(payload, { depth: null });
  printQuickPlayers(payload);
}

// ===== Sockets (NO crea sala; como cliente 2) =====
function wireSocket() {
  socket.on("connect", async () => {
    console.log("Socket connected:", socket.id);
    // La sala/batalla la crea el otro jugador (cliente 1). Este cliente SOLO se une. :contentReference[oaicite:5]{index=5}
    socket.emit("joinRoom", { roomId: ROOM_ID, player: { id: MY_ID, heroLevel: 1 } });

    // Héroe de pruebas (disponibilidad la controla el server según tipo/nivel/power/cooldown) :contentReference[oaicite:6]{index=6}
    const HERO_STATS = {
      hero: {
        heroType: "POISON_ROGUE", level: 1, power: 8, health: 36, defense: 8, attack: 10,
        attackBoost: { min: 1, max: 10 }, damage: { min: 1, max: 6 },
        specialActions: ALL_SPECIALS.map(id => ({ id, name: id, actionType: "ATTACK", powerCost: 1, cooldown: 0, isAvailable: true, effect: [] })),
        randomEffects: [
          { randomEffectType: "DAMAGE", percentage: 55, valueApply: { min: 0, max: 0 } },
          { randomEffectType: "CRITIC_DAMAGE", percentage: 10, valueApply: { min: 2, max: 4 } },
          { randomEffectType: "EVADE", percentage: 5, valueApply: { min: 0, max: 0 } },
          { randomEffectType: "RESIST", percentage: 10, valueApply: { min: 0, max: 0 } },
          { randomEffectType: "ESCAPE", percentage: 0, valueApply: { min: 0, max: 0 } },
          { randomEffectType: "NEGATE", percentage: 20, valueApply: { min: 0, max: 0 } },
        ],
      },
      equipped: {
        items: [], armors: [], weapons: [],
        epicAbilites: ALL_MASTERS.map(id => ({ id, name: id, compatibleHeroType: "WARRIOR_ARMS", effects: [], cooldown: 0, isAvailable: true, masterChance: 0.1 })),
      },
    };

    await axios.post(`${API_URL}/api/rooms/${ROOM_ID}/join`, { playerId: MY_ID, heroLevel: 1, heroStats: HERO_STATS }).catch(() => {});
    socket.emit("setHeroStats", { roomId: ROOM_ID, playerId: MY_ID, stats: HERO_STATS });
    socket.emit("playerReady", { roomId: ROOM_ID, playerId: MY_ID, team: MY_TEAM });
  });

  socket.on("battleStarted", async (data: any) => {
    socket.emit("joinBattle", { roomId: ROOM_ID, playerId: MY_ID });
    turns = data?.turns || [];
    currentTurn = turns[0] || null;
    finished = false;
    lastPayload = data;

    const ok = startMatchIfEligible(ROOM_ID, MATCH_ID, data, MY_ID);
    if (!ok.ok) console.log("Dataset: partida no elegible:", ok.reason);

    await printRaw("battleStarted", data);
    if (currentTurn === MY_ID) actWithAI();
  });

  socket.on("actionResolved", async (data: any) => {
    await printRaw("actionResolved", data);
    recordOutcome(data); // Escribe fila NDJSON con outcome
    if (data?.state === "FINISHED" || data?.winner || data?.winningTeam) {
      finished = true;
      console.log("⚑ Battle finished.", data?.winner ? `Winner: ${data.winner}` : "");
      return;
    }
    const before = currentTurn;
    currentTurn = data?.nextTurnPlayer || currentTurn;
    lastPayload = data;
    if (currentTurn === MY_ID && before !== MY_ID) turnIndex += 1;
    if (currentTurn === MY_ID) actWithAI();
  });

  socket.on("battleEnded", (data: any) => { finished = true; console.log("⚑ battleEnded:", data); });
  socket.on("error", (err) => console.error("Socket error:", err));
}

// ==== boot ====
(async () => {
  try {
    // Entrena/carga desde tu MERGED
    MODEL = await loadOrTrain();
  } catch (e) {
    console.error("No se pudo cargar/entrenar el modelo:", (e as Error).message);
    // Jugará con fallback BASIC si la IA falla
  }
  wireSocket();
})();
