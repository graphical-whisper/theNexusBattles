// dataset_logger.ts (actualizado)
import { createWriteStream, mkdirSync, existsSync } from "fs";
import path from "path";
import { SPECIAL_KIND, MASTER_KIND, ActionGroup } from "./ai_action_catalog";

type ActionType = "BASIC_ATTACK" | "SPECIAL_SKILL" | "MASTER_SKILL";

type HeroState = {
  type?: string | number;
  heroType?: string | number;
  level: number;
  health: number;
  power: number;
  attack: number;
  defense: number;
  damage?: { min: number; max: number };
  attackBoost?: { min: number; max: number };
  randomEffects?: { randomEffectType: string; percentage: number }[];
  specialActions?: { name: string; powerCost: number; isAvailable: boolean }[]; // Agregar aquí
};


type PlayerNode = {
  username?: string; id?: string; playerId?: string;
  heroStats?: { hero?: HeroState; equipped?: any }
};

function ensureDir(p: string) { if (!existsSync(p)) mkdirSync(p, { recursive: true }); }
const DATA_DIR = path.resolve(process.env["DATA_DIR"] || "data");
ensureDir(DATA_DIR);
const STREAM = createWriteStream(path.join(DATA_DIR, "pve_turns2.ndjson"), { flags: "a" });

// --- Normalización de tipos de héroe desde el enum del server ---
const HERO_TYPE_MAP = [
  "TANK",
  "WEAPONS_PAL",
  "FIRE_MAGE",
  "ICE_MAGE",
  "POISON_ROGUE",
  "MACHETE_ROGUE",
  "SHAMAN",
  "MEDIC"
] as const;

type HeroTypeString = typeof HERO_TYPE_MAP[number];

function normalizeHeroType(raw?: string | number): HeroTypeString | "UNKNOWN" {
  if (typeof raw === "number") {
    return (HERO_TYPE_MAP[raw] ?? "UNKNOWN");
  }
  if (typeof raw === "string") {
    const s = raw.toUpperCase().trim();
    // Si ya viene como string del enum, lo aceptamos
    if ((HERO_TYPE_MAP as readonly string[]).includes(s)) return s as HeroTypeString;
    return (s as any) || "UNKNOWN";
  }
  return "UNKNOWN";
}

// Bases de HP para hp_pct
const HP_BASE_BY_TYPE: Record<HeroTypeString, number> = {
  TANK: 44, WEAPONS_PAL: 44,
  FIRE_MAGE: 40, ICE_MAGE: 40,
  POISON_ROGUE: 36, MACHETE_ROGUE: 36,
  SHAMAN: 28, MEDIC: 28
};

function idOf(p?: PlayerNode) {
  return (p?.username || p?.id || p?.playerId || "").toString();
}

function hpPct(h: HeroState, heroTypeNorm: HeroTypeString | "UNKNOWN") {
  const base =
    (heroTypeNorm !== "UNKNOWN" ? HP_BASE_BY_TYPE[heroTypeNorm as HeroTypeString] : undefined) ?? 40;
  return base > 0 ? Number((h.health / base).toFixed(4)) : 1;
}

function buildSideFeatures(h: HeroState) {
  const heroTypeNorm = normalizeHeroType(h.type ?? h.heroType);
  const re = (kind: string) =>
    h?.randomEffects?.find(e => e.randomEffectType === kind)?.percentage ?? null;

  return {
    // type: siempre canónico y presente
    type: heroTypeNorm,
    level: h.level,
    hp: h.health,
    hp_pct: hpPct(h, heroTypeNorm),
    power: h.power,
    attack: h.attack,
    defense: h.defense,
    dmg_min: h.damage?.min ?? null,
    dmg_max: h.damage?.max ?? null,
    atk_boost_min: h.attackBoost?.min ?? 0,
    atk_boost_max: h.attackBoost?.max ?? 0,
    pct_damage: re("DAMAGE"),
    pct_crit: re("CRITIC_DAMAGE"),
    pct_evade: re("EVADE"),
    pct_resist: re("RESIST"),
    pct_escape: re("ESCAPE"),
    pct_negate: re("NEGATE"),
  };
}

function buildMask(hero: HeroState, equipped: any) {
  const specials = (hero as any)?.specialActions ?? [];
  const masters = (equipped?.epicAbilites ?? []).map((m: any) => ({
    id: m.id || m.name,
    cooldown: m.cooldown ?? 0,
    isAvailable: !!m.isAvailable
  }));

  const isValidSpecial = (s: any) =>
    !!s?.isAvailable && (s?.cooldown ?? 0) === 0 && (hero.power ?? 0) >= (s.powerCost ?? 0);

  const agg = (ids: string[], dict: Record<string, ActionGroup>) =>
    ids.reduce((acc, id) => {
      const g = dict[id];
      if (g) acc[g] = (acc[g] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

  const specialsValid = specials.filter(isValidSpecial).map((s: any) => (s.id || s.name));
  const mastersValid = masters.filter((m: any) => m.isAvailable && m.cooldown === 0).map((m: any) => m.id);

  const sAgg = agg(specialsValid, SPECIAL_KIND);
  const mAgg = agg(mastersValid, MASTER_KIND);

  return {
    num_specials_valid_off: sAgg["offense"] || 0,
    num_specials_valid_def: sAgg["defense"] || 0,
    num_support_valid: sAgg["support"] || 0,
    num_masters_valid_off: mAgg["offense"] || 0,
    num_masters_valid_def: mAgg["defense"] || 0,
    num_masters_support: mAgg["support"] || 0,
  };
}

// --- Estado interno de la última decisión pendiente de outcome ---
let pending: null | {
  matchId: string; roomId: string; turnIndex: number;
  actorId: string; enemyId: string;
  actorBefore: ReturnType<typeof buildSideFeatures>;
  enemyBefore: ReturnType<typeof buildSideFeatures>;
  maskStats: ReturnType<typeof buildMask>;
  chosen: { kind: ActionType; skillId?: string; group: ActionGroup; damage: number; heal: number; powerCost: number };
} = null;

// --- API expuesta ---
export function startMatchIfEligible(roomId: string, matchId: string, battlePayload: any, loggerId: string) {
  void roomId; void loggerId; void matchId;
  const battle = battlePayload?.battle ?? battlePayload;

  // Preferir players/teams si existen
  const players = battle?.players ?? battle?.teams?.flatMap((t: any) => t.players) ?? [];
  if (players.length === 2) {
    const [p1, p2] = players;
    const h1 = p1?.heroStats?.hero as HeroState | undefined;
    const h2 = p2?.heroStats?.hero as HeroState | undefined;
    if (h1 && h2 && h1.level !== h2.level) return { ok: false, reason: "diff_level" };
    return { ok: true, via: "players" };
  }

  // Fallback: por turns
  const turns = battle?.turns ?? battlePayload?.turns ?? [];
  if (Array.isArray(turns) && new Set(turns).size === 2) {
    return { ok: true, via: "turns_only" };
  }
  return { ok: false, reason: "not_1v1" };
}

export function recordDecision(
  roomId: string,
  matchId: string,
  turnIndex: number,
  payloadBefore: any,
  actorId: string,
  chosen: { kind: ActionType; skillId?: string, damage?: number, heal?: number, powerCost?: number }
) {
  const battle = payloadBefore?.battle ?? payloadBefore;
  const ps: PlayerNode[] = battle?.players ?? battle?.teams?.flatMap((t: any) => t.players) ?? [];
  const me = ps.find(p => idOf(p) === actorId);
  const foe = ps.find(p => idOf(p) !== actorId);
  if (!me || !foe) return;

  const hMe = me.heroStats?.hero as HeroState;
  const hFoe = foe.heroStats?.hero as HeroState;

  const actorBefore = buildSideFeatures(hMe);
  const enemyBefore = buildSideFeatures(hFoe);

  const mask = buildMask(hMe, me.heroStats?.equipped);
  const chosen_group: ActionGroup =
    chosen.kind === "BASIC_ATTACK" ? "offense" :
      (chosen.kind === "SPECIAL_SKILL"
        ? (SPECIAL_KIND[chosen.skillId || ""] || "offense")
        : (MASTER_KIND[chosen.skillId || ""] || "support"));

  // Ahora simplemente usamos los valores que ya fueron calculados previamente
  const damage = chosen.damage ?? 0; // Tomamos el daño calculado por el cliente
  const heal = chosen.heal ?? 0;      // Tomamos la curación calculada por el cliente
  const powerCost = chosen.powerCost ?? 0; // Tomamos el costo de poder calculado por el cliente

  // Registrar los valores sin calcularlos nuevamente
  pending = {
    matchId, roomId, turnIndex,
    actorId, enemyId: idOf(foe),
    actorBefore, enemyBefore,
    maskStats: mask,
    chosen: { ...chosen, group: chosen_group, damage, heal, powerCost }
  };
}



export function recordOutcome(payloadAfter: any) {
  if (!pending) return;
  const ts = new Date().toISOString();

  const battle = payloadAfter?.battle ?? payloadAfter;
  const ps: PlayerNode[] = battle?.players ?? battle?.teams?.flatMap((t: any) => t.players) ?? [];
  const me = ps.find(p => idOf(p) === pending!.actorId);
  const foe = ps.find(p => idOf(p) === pending!.enemyId);
  const hMe = me?.heroStats?.hero as HeroState | undefined;
  const hFoe = foe?.heroStats?.hero as HeroState | undefined;

  // Inferir deltas por HP
  const enemy_hp_after = hFoe?.health ?? null;
  const actor_hp_after = hMe?.health ?? null;
  const dmg_to_enemy = (pending.enemyBefore.hp ?? 0) - (enemy_hp_after ?? pending.enemyBefore.hp);
  const heal_to_self = (actor_hp_after ?? 0) - (pending.actorBefore.hp ?? 0);
  const effect_applied =
    dmg_to_enemy > 0 ? "DAMAGE" :
      heal_to_self > 0 ? "HEAL" : "NONE";

  // Ahora registramos directamente los valores calculados (daño, curación) que ya están en `pending`
  const row = {
    ts,
    match_id: pending.matchId,
    room_id: pending.roomId,
    turn_index: pending.turnIndex,
    actor_id: pending.actorId,
    enemy_id: pending.enemyId,

    actor_hero_type: pending.actorBefore.type,
    enemy_hero_type: pending.actorBefore.type,

    actor: pending.actorBefore,
    enemy: pending.enemyBefore,

    ...pending.maskStats,

    chosen_action_kind: pending.chosen.kind,
    chosen_skill_id: pending.chosen.skillId || null,
    chosen_action_group: pending.chosen.group,

    dmg_to_enemy,
    heal_to_self,
    effect_applied,
    enemy_hp_after,
    actor_hp_after,

    battle_finished: !!(payloadAfter?.state === "FINISHED" || payloadAfter?.winner || payloadAfter?.winningTeam),
    winner_id: payloadAfter?.winner || null
  };

  STREAM.write(JSON.stringify(row) + "\n");
  pending = null;
}


