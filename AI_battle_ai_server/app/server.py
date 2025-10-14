# server.py — IA Server + Bot spawner (2 pasos + Misiones) con mapeo completo de HERO_STATS
from __future__ import annotations
import os
import hashlib
from typing import List, Literal, Optional, Tuple, Dict, Any

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ============================ Config ============================
INVENTORY_BASE_URL = os.getenv("INVENTORY_BASE_URL", "http://localhost:9000")
INVENTORY_HERO_STATS_PATH = os.getenv("INVENTORY_HERO_STATS_PATH", "/v1/bot/hero-stats")
INVENTORY_TIMEOUT_SECS = float(os.getenv("INVENTORY_TIMEOUT_SECS", "5.0"))

# ================= Optional ML policy (lazy import) ==============
def _try_load_model():
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    for fname in ("hero_action_selector.keras", "hero_action_selector.h5"):
        fpath = os.path.join(model_dir, fname)
        if os.path.exists(fpath):
            try:
                from tensorflow import keras  # lazy import
                return keras.models.load_model(fpath)
            except Exception as e:
                print(f"[warn] Could not load model {fpath}: {e}")
    return None

# ============================ Domain ============================
ActionKind = Literal["BASIC", "ATTACK", "SPECIAL_SKILL_1", "SPECIAL_SKILL_2", "SPECIAL_SKILL_3"]

class BattleSide(BaseModel):
    hero_type: str
    hp: float = Field(..., ge=0.0)
    mp: float = Field(0.0, ge=0.0)
    level: int = Field(1, ge=1, le=99)
    cooldowns: Dict[str, int] = Field(default_factory=dict)
    buffs: Dict[str, Any] = Field(default_factory=dict)
    debuffs: Dict[str, Any] = Field(default_factory=dict)

class DecideRequest(BaseModel):
    actor: BattleSide
    enemy: BattleSide
    turn: int = Field(1, ge=1)
    rng: Optional[int] = None
    forbidden_actions: Optional[List[ActionKind]] = None

class DecideResponse(BaseModel):
    action: ActionKind
    reason: str
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    skill_id: Optional[str] = None
    skill_name: Optional[str] = None

# ========================= HERO SPECIALS ========================
HERO_SPECIALS: Dict[str, List[Dict[str, Any]]] = {
    "TANK": [
        {"slot": "SPECIAL_SKILL_1", "id": "GOLPE_ESCUDO",   "name": "Golpe con escudo", "level_req": 2, "cost": 2, "effect": "+2 al ataque"},
        {"slot": "SPECIAL_SKILL_2", "id": "MANO_PIEDRA",    "name": "Mano de piedra",   "level_req": 5, "cost": 4, "effect": "+12 a la defensa"},
        {"slot": "SPECIAL_SKILL_3", "id": "DEFENSA_FEROZ",  "name": "Defensa feroz",    "level_req": 8, "cost": 6, "effect": "Inmune físico y (3d6) al mágico"},
    ],
    "WARRIOR_ARMS": [
        {"slot": "SPECIAL_SKILL_1", "id": "EMBATE_SANGRIENTO", "name": "Embate sangriento", "level_req": 2, "cost": 4, "effect": "+2 ATK y +1 daño"},
        {"slot": "SPECIAL_SKILL_2", "id": "LANZA_DIOSES",      "name": "Lanza de los dioses","level_req": 5, "cost": 4, "effect": "+2 daño"},
        {"slot": "SPECIAL_SKILL_3", "id": "GOLPE_TORMENTA",    "name": "Golpe de tormenta", "level_req": 8, "cost": 6, "effect": "+(3d6) ATK y +2 daño"},
    ],
    "MAGE_FIRE": [
        {"slot": "SPECIAL_SKILL_1", "id": "MISILES_MAGMA", "name": "Misiles de magma", "level_req": 2, "cost": 2, "effect": "+1 ATK y +2 daño"},
        {"slot": "SPECIAL_SKILL_2", "id": "VULCANO",       "name": "Vulcano",          "level_req": 5, "cost": 6, "effect": "+3 ATK y +(3d9) daño"},
        {"slot": "SPECIAL_SKILL_3", "id": "PARED_FUEGO",   "name": "Pared de fuego",   "level_req": 8, "cost": 4, "effect": "+1 ATK y refleja daño previo"},
    ],
    "MAGE_ICE": [
        {"slot": "SPECIAL_SKILL_1", "id": "LLUVIA_HIELO", "name": "Lluvia de hielo", "level_req": 2, "cost": 2, "effect": "+2 ATK y +2 daño"},
        {"slot": "SPECIAL_SKILL_2", "id": "CONO_HIELO",   "name": "Cono de hielo",   "level_req": 5, "cost": 6, "effect": "+2 daño y -ATK enemigo (1d3) x2T"},
        {"slot": "SPECIAL_SKILL_3", "id": "BOLA_HIELO",   "name": "Bola de hielo",   "level_req": 8, "cost": 4, "effect": "+2 ATK y -daño oponente (0d4)"},
    ],
    "ROGUE_POISON": [
        {"slot": "SPECIAL_SKILL_1", "id": "FLOR_LOTO", "name": "Flor de loto", "level_req": 2, "cost": 2, "effect": "+(4d8) daño"},
        {"slot": "SPECIAL_SKILL_2", "id": "AGONIA",    "name": "Agonía",       "level_req": 5, "cost": 4, "effect": "+(2d9) daño"},
        {"slot": "SPECIAL_SKILL_3", "id": "PIQUETE",   "name": "Piquete",      "level_req": 8, "cost": 4, "effect": "+1 ATK (2T) y +2 daño (1T)"},
    ],
    "ROGUE_MACHETE": [
        {"slot": "SPECIAL_SKILL_1", "id": "CORTADA",    "name": "Cortada",    "level_req": 2, "cost": 2, "effect": "+2 daño por 2T"},
        {"slot": "SPECIAL_SKILL_2", "id": "MACHETAZO",  "name": "Machetazo",  "level_req": 5, "cost": 4, "effect": "+(2d8) daño y +1 ATK"},
        {"slot": "SPECIAL_SKILL_3", "id": "PLANAZO",    "name": "Planazo",    "level_req": 8, "cost": 4, "effect": "+(2d8) ATK y +1 daño"},
    ],
}
SLOTS = ("SPECIAL_SKILL_1", "SPECIAL_SKILL_2", "SPECIAL_SKILL_3")

# ===== Alias server ↔ cliente (para heroType) =====
SERVER_TO_CLIENT_HERO = {
    "ROGUE_POISON": "POISON_ROGUE",
    "ROGUE_MACHETE": "MACHETE_ROGUE",
    "MAGE_FIRE": "FIRE_MAGE",
    "MAGE_ICE": "ICE_MAGE",
    "WARRIOR_ARMS": "WARRIOR_ARMS",
    "TANK": "TANK",
}
CLIENT_TO_SERVER_HERO = {v: k for k, v in SERVER_TO_CLIENT_HERO.items()}

def _to_client_hero_type(h: str) -> str:
    return SERVER_TO_CLIENT_HERO.get((h or "").upper(), h)

def _to_server_hero_type(h: str) -> str:
    return CLIENT_TO_SERVER_HERO.get((h or "").upper(), h)

# ======================= helpers de CD/política ==================
def _normalize_cd(cooldowns: Dict[str, int]) -> Dict[str, int]:
    out = {k.upper(): int(v) for k, v in (cooldowns or {}).items()}
    alias = {"SPECIAL1":"SPECIAL_SKILL_1","SPECIAL2":"SPECIAL_SKILL_2","SPECIAL3":"SPECIAL_SKILL_3",
             "SPECIAL_1":"SPECIAL_SKILL_1","SPECIAL_2":"SPECIAL_SKILL_2","SPECIAL_3":"SPECIAL_SKILL_3",
             "SP1":"SPECIAL_SKILL_1","SP2":"SPECIAL_SKILL_2","SP3":"SPECIAL_SKILL_3"}
    for k, v in list(out.items()):
        if k in alias: out[alias[k]] = v
    for i, slot in enumerate(SLOTS, start=1):
        if f"special{i}" in (cooldowns or {}) and slot not in out:
            out[slot] = int(cooldowns[f"special{i}"])
    return {k: int(out.get(k, 0)) for k in SLOTS}

def _get_skill_meta(hero_type: str, slot: ActionKind):
    if slot not in SLOTS: return None
    for s in HERO_SPECIALS.get(hero_type, []):
        if s.get("slot") == slot:
            return s
    return None

def _is_available(actor: BattleSide, act: ActionKind):
    if act in ("BASIC","ATTACK"):
        return True, "basic/attack always available"
    meta = _get_skill_meta(actor.hero_type, act)
    if not meta: return False, f"{act} not defined for {actor.hero_type}"
    cd = _normalize_cd(actor.cooldowns).get(act, 0)
    if cd and int(cd) > 0: return False, f"{act} on cooldown ({cd})"
    if actor.level < int(meta["level_req"]): return False, f"{act} requires level {meta['level_req']}"
    if actor.mp < float(meta["cost"]): return False, f"Not enough MP for {act} (cost {meta['cost']})"
    return True, "available"

def _rule_based_policy(actor: BattleSide, enemy: BattleSide, forbidden: Optional[List[ActionKind]] = None):
    fset = set(forbidden or [])
    if enemy.hp <= max(10.0, 0.15 * 100) and "ATTACK" not in fset:
        ok, _ = _is_available(actor, "ATTACK")
        if ok: return "ATTACK", "Enemy low HP → ATTACK", 0.65
    for slot in ("SPECIAL_SKILL_3","SPECIAL_SKILL_2","SPECIAL_SKILL_1"):
        if slot in fset: continue
        ok, _ = _is_available(actor, slot)
        if ok:
            meta = _get_skill_meta(actor.hero_type, slot)
            return slot, f"{slot} available (lvl≥{meta['level_req']}, MP≥{meta['cost']})", 0.72
    if "ATTACK" not in fset: return "ATTACK", "No specials available → ATTACK", 0.58
    return "BASIC", "All else blocked → BASIC", 0.55

def _featurize(actor: BattleSide, enemy: BattleSide) -> np.ndarray:
    hero_ids = {k: i for i, k in enumerate(sorted(HERO_SPECIALS.keys()))}
    a_type = hero_ids.get(actor.hero_type, len(hero_ids))
    e_type = hero_ids.get(enemy.hero_type, len(hero_ids))
    cd = _normalize_cd(actor.cooldowns)
    vec = np.array([
        a_type, actor.hp, actor.mp, actor.level,
        e_type, enemy.hp, enemy.mp, enemy.level,
        int(cd.get("SPECIAL_SKILL_1", 0)),
        int(cd.get("SPECIAL_SKILL_2", 0)),
        int(cd.get("SPECIAL_SKILL_3", 0)),
    ], dtype=np.float32)
    return vec[None, :]

# ======================= App & policy ============================
app = FastAPI(title="NexusBattle IA Server", version="4.3.0")
_model = None

@app.on_event("startup")
def _maybe_load_model():
    global _model
    _model = _try_load_model()
    print("[info] ML model loaded." if _model is not None else "[info] Rule-based mode.")

@app.get("/health")
def health():
    return {
        "ok": True,
        "mode": "ml" if _model is not None else "rules",
        "supported_actions": ["BASIC","ATTACK","SPECIAL_SKILL_1","SPECIAL_SKILL_2","SPECIAL_SKILL_3"],
        "heroes": list(HERO_SPECIALS.keys()),
        "skills_by_hero": HERO_SPECIALS,
        "inventory_base_url": INVENTORY_BASE_URL,
        "inventory_path": INVENTORY_HERO_STATS_PATH,
    }

@app.post("/v1/decide", response_model=DecideResponse)
def decide(req: DecideRequest) -> DecideResponse:
    if req.rng is not None:
        np.random.seed(req.rng)
    forbidden = req.forbidden_actions or []

    if _model is not None:
        try:
            x = _featurize(req.actor, req.enemy)
            logits = _model.predict(x, verbose=0)[0]
            acts = ["BASIC","ATTACK","SPECIAL_SKILL_1","SPECIAL_SKILL_2","SPECIAL_SKILL_3"]
            mask = np.array([(-1e9 if a in forbidden else 0.0) for a in acts], dtype=np.float32)
            scores = logits + mask
            idx = int(np.argmax(scores))
            chosen = acts[idx] if 0 <= idx < len(acts) else "BASIC"
            ok, why = _is_available(req.actor, chosen)
            if not ok:
                act, reason, conf = _rule_based_policy(req.actor, req.enemy, forbidden)
                meta = _get_skill_meta(req.actor.hero_type, act)
                return DecideResponse(action=act, reason=f"ML chose unavailable ({chosen}: {why}). Fallback → {reason}",
                                      confidence=conf, skill_id=(meta or {}).get("id"), skill_name=(meta or {}).get("name"))
            ex = np.exp(scores - np.max(scores))
            conf = float(ex[idx] / np.sum(ex))
            meta = _get_skill_meta(req.actor.hero_type, chosen)
            return DecideResponse(action=chosen, reason="Chosen by ML policy", confidence=conf,
                                  skill_id=(meta or {}).get("id"), skill_name=(meta or {}).get("name"))
        except Exception as e:
            act, reason, conf = _rule_based_policy(req.actor, req.enemy, forbidden)
            meta = _get_skill_meta(req.actor.hero_type, act)
            return DecideResponse(action=act, reason=f"ML error → {e}. Fallback → {reason}", confidence=conf,
                                  skill_id=(meta or {}).get("id"), skill_name=(meta or {}).get("name"))
    act, reason, conf = _rule_based_policy(req.actor, req.enemy, forbidden)
    meta = _get_skill_meta(req.actor.hero_type, act)
    return DecideResponse(action=act, reason=reason, confidence=conf,
                          skill_id=(meta or {}).get("id"), skill_name=(meta or {}).get("name"))

# =================== Utilidades de inventario ====================
def _allowed_hero_types() -> List[str]:
    return list(HERO_SPECIALS.keys())

def _choose_hero_type(room_id: str, player_id: str, level: int, rng: Optional[int]) -> str:
    pool = _allowed_hero_types()
    if not pool:
        raise HTTPException(status_code=500, detail="No hay tipos de héroe disponibles")
    
    print(f"[debug] Hero pool available: {pool}")  # Debug para ver qué héroes están disponibles
    
    seed_src = f"{room_id}|{player_id}|{level}|{rng if rng is not None else ''}"
    h = hashlib.sha256(seed_src.encode("utf-8")).digest()
    idx = int.from_bytes(h[:4], "big") % len(pool)
    chosen = pool[idx]
    
    print(f"[debug] Chosen hero type: {chosen} (index: {idx})")  # Debug para ver qué se eligió
    
    return chosen

def _inventory_url() -> str:
    return INVENTORY_BASE_URL.rstrip("/") + INVENTORY_HERO_STATS_PATH

# =================== Estadísticas quemadas de héroes ===================
def _local_hero_stats(hero_type: str, hero_level: int) -> Dict[str, Any]:
    """
    Estadísticas base quemadas para cada tipo de héroe, escaladas por nivel.
    Estas se usan como fallback cuando el servicio de inventario no está disponible.
    """
    
    # Estadísticas base por tipo de héroe (nivel 1) - EXACTAMENTE como las proporcionaste
    BASE_STATS = {
        "TANK": {
            "power": 10, "health": 44, "defense": 11, "attack": 10,
            "attackBoost": {"min": 1, "max": 6},
            "damage": {"min": 1, "max": 4}
        },
        "WARRIOR_ARMS": {
            "power": 8, "health": 44, "defense": 11, "attack": 10,
            "attackBoost": {"min": 1, "max": 6},
            "damage": {"min": 1, "max": 6}
        },
        "MAGE_FIRE": {
            "power": 8, "health": 40, "defense": 10, "attack": 10,
            "attackBoost": {"min": 1, "max": 8},
            "damage": {"min": 1, "max": 8}
        },
        "MAGE_ICE": {
            "power": 10, "health": 40, "defense": 10, "attack": 10,
            "attackBoost": {"min": 1, "max": 8},
            "damage": {"min": 1, "max": 6}
        },
        "ROGUE_POISON": {
            "power": 8, "health": 36, "defense": 8, "attack": 10,
            "attackBoost": {"min": 1, "max": 10},
            "damage": {"min": 1, "max": 6}
        },
        "ROGUE_MACHETE": {
            "power": 8, "health": 36, "defense": 8, "attack": 10,
            "attackBoost": {"min": 1, "max": 10},
            "damage": {"min": 1, "max": 8}
        }
    }
    
    # Obtener stats base o usar defaults si el tipo no existe
    base = BASE_STATS.get(hero_type, {
        "power": 8, "health": 40, "defense": 10, "attack": 10,
        "attackBoost": {"min": 1, "max": 8},
        "damage": {"min": 1, "max": 6}
    })
    
    lvl = int(hero_level)
    
    # Fórmulas de escalado por nivel - multiplicando por nivel como especificaste
    power_scaling = base["power"] * lvl
    health_scaling = base["health"] * lvl
    defense_scaling = base["defense"] * lvl
    attack_scaling = base["attack"] * lvl
    
    # Los rangos (attackBoost y damage) NO aumentan con el nivel
    # Se mantienen exactamente iguales a los valores base
    attack_boost = base["attackBoost"].copy()
    damage_range = base["damage"].copy()
    
    return {
        "heroType": _to_client_hero_type(hero_type),
        "level": lvl,
        "power": power_scaling,
        "health": health_scaling,
        "defense": defense_scaling,
        "attack": attack_scaling,
        "attackBoost": attack_boost,  # Rangos sin cambios
        "damage": damage_range,       # Rangos sin cambios
        "source": "local_fallback"
    }

# Función mejorada para fetch con mejor logging
def _fetch_hero_stats_from_inventory(hero_type: str, hero_level: int) -> Dict[str, Any]:
    """Intenta obtener stats del inventario, falla a local si es necesario"""
    url = _inventory_url()
    payload = {"hero_type": hero_type, "hero_level": int(hero_level)}
    
    try:
        print(f"[inventory] Fetching stats for {hero_type} level {hero_level} from {url}")
        resp = requests.post(url, json=payload, timeout=INVENTORY_TIMEOUT_SECS)
        resp.raise_for_status()
        data = resp.json()
        print(f"[inventory] Successfully fetched stats for {hero_type} level {hero_level}")
        return data
    except requests.exceptions.ConnectionError:
        print(f"[inventory] Connection failed, using local stats for {hero_type} level {hero_level}")
        return _local_hero_stats(hero_type, hero_level)
    except requests.exceptions.Timeout:
        print(f"[inventory] Timeout, using local stats for {hero_type} level {hero_level}")
        return _local_hero_stats(hero_type, hero_level)
    except Exception as e:
        print(f"[inventory] Error ({type(e).__name__}: {e}), using local stats for {hero_type} level {hero_level}")
        return _local_hero_stats(hero_type, hero_level)

# Endpoint para verificar las stats locales (opcional)
@app.get("/v1/hero-stats/local/{hero_type}/{level}")
def get_local_hero_stats(hero_type: str, level: int = 1):
    """Endpoint para verificar las estadísticas quemadas de cualquier héroe"""
    server_hero_type = _to_server_hero_type(hero_type)
    stats = _local_hero_stats(server_hero_type, level)
    return {
        "ok": True,
        "hero_type": hero_type,
        "server_hero_type": server_hero_type,
        "level": level,
        "stats": stats
    }

# =================== Enriquecimiento de HERO_STATS ===============
_DEFAULT_RANDOM_EFFECTS = [
    {"randomEffectType": "DAMAGE",        "percentage": 55, "valueApply": {"min": 0, "max": 0}},
    {"randomEffectType": "CRITIC_DAMAGE", "percentage": 10, "valueApply": {"min": 2, "max": 4}},
    {"randomEffectType": "EVADE",         "percentage": 5,  "valueApply": {"min": 0, "max": 0}},
    {"randomEffectType": "RESIST",        "percentage": 10, "valueApply": {"min": 0, "max": 0}},
    {"randomEffectType": "ESCAPE",        "percentage": 0,  "valueApply": {"min": 0, "max": 0}},
    {"randomEffectType": "NEGATE",        "percentage": 20, "valueApply": {"min": 0, "max": 0}},
]

def _build_special_actions(server_hero_type: str) -> List[Dict[str, Any]]:
    out = []
    for s in HERO_SPECIALS.get(server_hero_type, []):
        out.append({
            "name": s["name"],
            "actionType": "ATTACK",           # acorde a tu cliente Node
            "powerCost": int(s.get("cost", 1)),
            "cooldown": 0,
            "isAvailable": True,
            "effect": [],                     # mantenemos efecto como lista vacía (server Node lo calcula)
        })
    return out

def _default_equipped(client_hero_type: str) -> Dict[str, Any]:
    # Estructura mínima válida; si quieres poblarla más, añade tus ítems/armas
    return {
        "items": [],
        "armors": [],
        "weapons": [],
        "epicAbilites": [
            {
                "name": f"Maestría de {client_hero_type}",
                "compatibleHeroType": client_hero_type,
                "effects": [],
                "cooldown": 0,
                "isAvailable": True,
                "masterChance": 0.1,
            }
        ],
    }

def _ensure_full_hero(hero: Dict[str, Any]) -> Dict[str, Any]:
    """Completa hero con campos faltantes y normaliza heroType (cliente)."""
    hero = dict(hero or {})
    # Asegurar heroType en alias cliente
    if "heroType" in hero:
        hero["heroType"] = _to_client_hero_type(_to_server_hero_type(str(hero["heroType"])))
    # Defaults de daño/boost
    hero.setdefault("attackBoost", {"min": 1, "max": 10})
    hero.setdefault("damage", {"min": 1, "max": 6})
    # Completar specialActions si faltan
    if not hero.get("specialActions"):
        server_t = _to_server_hero_type(hero.get("heroType", ""))
        hero["specialActions"] = _build_special_actions(server_t)
    # Completar randomEffects si faltan
    if not hero.get("randomEffects"):
        hero["randomEffects"] = list(_DEFAULT_RANDOM_EFFECTS)
    return hero

def _compose_full_stats(server_hero_type: str, core_stats: Dict[str, Any], equipped: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    core_stats: stats de inventario/local (pueden venir con heroType en alias cliente o no)
    Devuelve: {"hero": {...}, "equipped": {...}} listo para setHeroStats.
    """
    # Normalizar heroType a cliente y asegurar campos base
    hero = _ensure_full_hero(core_stats)
    # Si inventario devolvió heroType interno, forzamos alias cliente
    if "heroType" not in hero or hero["heroType"].upper() not in CLIENT_TO_SERVER_HERO:
        hero["heroType"] = _to_client_hero_type(server_hero_type)
    # Adjuntar equipped
    if equipped is None:
        equipped = _default_equipped(hero["heroType"])
    return {"hero": hero, "equipped": equipped}

def _ensure_full_stats_package(stats_pkg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cuando el cliente (misión) te trae su propio hero, garantizar que tiene todo.
    stats_pkg esperado ~ {"hero": {...}, "equipped": {...}} o {"heroType":..., ...}
    """
    if "hero" in stats_pkg:
        hero = _ensure_full_hero(stats_pkg["hero"])
        equipped = stats_pkg.get("equipped") or _default_equipped(hero.get("heroType", ""))
        return {"hero": hero, "equipped": equipped}
    else:
        # Si vino solo el hero "plano", lo empacamos
        hero = _ensure_full_hero(stats_pkg)
        equipped = _default_equipped(hero.get("heroType", ""))
        return {"hero": hero, "equipped": equipped}

# ======================= Bot spawn (2 pasos) =====================
class SpawnBotRequest(BaseModel):
    room_id: str
    player_id: str
    team: Literal["A", "B"]
    hero_level: int = 1
    socket_url: str = Field("http://localhost:3000", description="Socket.IO base URL del game server")
    api_url: Optional[str] = Field("http://localhost:3000", description="Base URL HTTP del game server (opcional)")
    rng: Optional[int] = Field(None, description="Semilla opcional para elección determinística del héroe")

class StopBotRequest(BaseModel):
    room_id: str
    player_id: str

# Registro en memoria de bots en ejecución
_BOTS: Dict[Tuple[str, str], Any] = {}

@app.post("/v1/bot/spawn")
def bot_spawn(req: SpawnBotRequest):
    key = (req.room_id, req.player_id)
    if key in _BOTS:
        return {"ok": True, "status": "already-running", "room_id": req.room_id, "player_id": req.player_id}

    server_hero_type = _choose_hero_type(req.room_id, req.player_id, req.hero_level, req.rng)
    inv_stats = _fetch_hero_stats_from_inventory(server_hero_type, req.hero_level)
    # Componer payload completo al estilo del cliente Node
    hero_stats_payload = _compose_full_stats(server_hero_type, inv_stats, equipped=None)

    from app.bot_client import BotClient, BotConfig  # type: ignore
    cfg = BotConfig(
        socket_url=req.socket_url, api_url=req.api_url,
        room_id=req.room_id, player_id=req.player_id, team=req.team,
        hero_stats=hero_stats_payload, hero_level=req.hero_level
    )
    bot = BotClient(cfg)
    bot.start()
    _BOTS[key] = bot

    return {
        "ok": True,
        "status": "spawned",
        "room_id": req.room_id,
        "player_id": req.player_id,
        "team": req.team,
        "hero_type": _to_client_hero_type(server_hero_type),
        "hero_stats": hero_stats_payload,
        "inventory_url": _inventory_url(),
    }

@app.post("/v1/bot/stop")
def bot_stop(req: StopBotRequest):
    key = (req.room_id, req.player_id)
    bot = _BOTS.pop(key, None)
    if not bot:
        raise HTTPException(status_code=404, detail="bot not found")
    try:
        bot.stop()
    except Exception:
        pass
    return {"ok": True, "status": "stopped"}

@app.get("/v1/bot/list")
def bot_list():
    return [{"room_id": k[0], "player_id": k[1]} for k in _BOTS.keys()]

# ======================= Misiones (enemigo IA) ===================
class MissionSpawnRequest(BaseModel):
    room_id: str
    socket_url: str = Field("http://localhost:3000")
    api_url: Optional[str] = Field("http://localhost:3000")
    player_id: str = Field("playerA")
    player_team: Literal["A","B"] = Field("A")
    enemy_id: str = Field("playerB")
    enemy_team: Literal["A","B"] = Field("B")
    # Héroe del jugador (puede venir incompleto; lo completamos)
    player_hero: Dict[str, Any]
    master: Optional[bool] = Field(None, description="Si True, el enemigo es maestro (+2 niveles). Si None, usa master_chance.")
    master_chance: Optional[float] = Field(0.2, ge=0.0, le=1.0)
    rng: Optional[int] = Field(None)

class MissionStopRequest(BaseModel):
    room_id: str
    player_id: Optional[str] = None
    enemy_id: Optional[str] = None

def _decide_master(room_id: str, player_id: str, enemy_id: str, base_level: int,
                   explicit: Optional[bool], chance: Optional[float], rng: Optional[int]) -> bool:
    if explicit is not None:
        return bool(explicit)
    p = 0.2 if chance is None else max(0.0, min(1.0, float(chance)))
    seed = f"master|{room_id}|{player_id}|{enemy_id}|{base_level}|{rng if rng is not None else ''}"
    h = hashlib.sha256(seed.encode("utf-8")).digest()
    v = int.from_bytes(h[:4], "big") / 2**32
    return v < p

@app.post("/v1/mission/spawn")
def mission_spawn(req: MissionSpawnRequest):
    # 1) Normalizar/Completar héroe del jugador → paquete completo
    p_pkg = _ensure_full_stats_package(req.player_hero if "hero" in req.player_hero else {"hero": req.player_hero})
    p_hero = p_pkg["hero"]
    p_level = int(p_hero.get("level", 1))

    # 2) Elegir enemigo y stats completos
    enemy_is_master = _decide_master(req.room_id, req.player_id, req.enemy_id, p_level,
                                     req.master, req.master_chance, req.rng)
    e_level = min(99, p_level + (2 if enemy_is_master else 0))
    server_enemy_type = _choose_hero_type(req.room_id, req.enemy_id, e_level, req.rng)
    inv_enemy = _fetch_hero_stats_from_inventory(server_enemy_type, e_level)
    e_pkg = _compose_full_stats(server_enemy_type, inv_enemy, equipped=None)

    # 3) Crear bots IA (jugador y enemigo)
    from app.bot_client import BotClient, BotConfig  # type: ignore

    key_player = (req.room_id, req.player_id)
    if key_player not in _BOTS:
        cfg_a = BotConfig(
            socket_url=req.socket_url, api_url=req.api_url,
            room_id=req.room_id, player_id=req.player_id, team=req.player_team,
            hero_stats=p_pkg, hero_level=p_level
        )
        bot_a = BotClient(cfg_a)
        bot_a.start()
        _BOTS[key_player] = bot_a

    key_enemy = (req.room_id, req.enemy_id)
    if key_enemy not in _BOTS:
        cfg_b = BotConfig(
            socket_url=req.socket_url, api_url=req.api_url,
            room_id=req.room_id, player_id=req.enemy_id, team=req.enemy_team,
            hero_stats=e_pkg, hero_level=e_level
        )
        bot_b = BotClient(cfg_b)
        bot_b.start()
        _BOTS[key_enemy] = bot_b

    return {
        "ok": True,
        "status": "mission-spawned",
        "room_id": req.room_id,
        "player": {"id": req.player_id, "team": req.player_team, "hero": p_pkg["hero"], "equipped": p_pkg["equipped"]},
        "enemy": {
            "id": req.enemy_id, "team": req.enemy_team, "is_master": enemy_is_master,
            "hero": e_pkg["hero"], "equipped": e_pkg["equipped"]
        },
        "inventory_url": _inventory_url(),
    }

@app.post("/v1/mission/stop")
def mission_stop(req: MissionStopRequest):
    stopped = []
    if req.player_id:
        k = (req.room_id, req.player_id)
        b = _BOTS.pop(k, None)
        if b:
            try: b.stop()
            except Exception: pass
            stopped.append({"room_id": req.room_id, "player_id": req.player_id})
    if req.enemy_id:
        k = (req.room_id, req.enemy_id)
        b = _BOTS.pop(k, None)
        if b:
            try: b.stop()
            except Exception: pass
            stopped.append({"room_id": req.room_id, "player_id": req.enemy_id})
    if not req.player_id and not req.enemy_id:
        for pid in list(_BOTS.keys()):
            if pid[0] == req.room_id:
                b = _BOTS.pop(pid, None)
                if b:
                    try: b.stop()
                    except Exception: pass
                    stopped.append({"room_id": pid[0], "player_id": pid[1]})
    return {"ok": True, "stopped": stopped}

@app.get("/v1/debug/hero-pool")
def debug_hero_pool():
    """Endpoint para debuggear el pool de héroes disponibles"""
    pool = _allowed_hero_types()
    return {
        "ok": True,
        "hero_pool": pool,
        "total_heroes": len(pool),
        "all_heroes_in_specials": list(HERO_SPECIALS.keys())
    }
    
# ========================= Entrypoint ============================
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.server:app", host=host, port=port, reload=False)
