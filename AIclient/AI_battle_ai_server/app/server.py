# server.py — IA Server + Bot spawner (2 pasos + Misiones)
from __future__ import annotations
import os
import hashlib
from typing import List, Literal, Optional, Tuple, Dict, Any

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ============================ Config ============================
# Servicio externo de inventario (provee hero_stats)
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

def _normalize_cd(cooldowns: Dict[str, int]) -> Dict[str, int]:
    out = {k.upper(): int(v) for k, v in cooldowns.items()}
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
app = FastAPI(title="NexusBattle IA Server", version="4.1.0")
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
    # Excluir héroes no permitidos
    banned = {"CHAMAN", "MEDICO", "SHAMAN", "HEALER"}
    return [ht for ht in HERO_SPECIALS.keys() if ht.upper() not in banned]

def _choose_hero_type(room_id: str, player_id: str, level: int, rng: Optional[int]) -> str:
    """Elección determinística: hash(room,player,level[,rng])."""
    pool = _allowed_hero_types()
    if not pool:
        raise HTTPException(status_code=500, detail="No hay tipos de héroe disponibles")
    seed_src = f"{room_id}|{player_id}|{level}|{rng if rng is not None else ''}"
    h = hashlib.sha256(seed_src.encode("utf-8")).digest()
    idx = int.from_bytes(h[:4], "big") % len(pool)
    return pool[idx]

def _inventory_url() -> str:
    return INVENTORY_BASE_URL.rstrip("/") + INVENTORY_HERO_STATS_PATH

def _fetch_hero_stats_from_inventory(hero_type: str, hero_level: int) -> Dict[str, Any]:
    """POST al servicio de inventario. Espera estructura:
        {
          "heroType": "...",
          "level": 5,
          "power": ...,
          "health": ...,
          "defense": ...,
          "attack": ...,
          "attackBoost": {"min":..,"max":..},
          "damage": {"min":..,"max":..}
        }
    """
    url = _inventory_url()
    payload = {"hero_type": hero_type, "hero_level": int(hero_level)}
    try:
        resp = requests.post(url, json=payload, timeout=INVENTORY_TIMEOUT_SECS)
        resp.raise_for_status()
        data = resp.json()
        # Validación mínima
        for k in ("heroType", "level", "power", "health", "defense", "attack", "attackBoost", "damage"):
            if k not in data:
                raise ValueError(f"missing key '{k}' in inventory response")
        return data
    except Exception as e:
        print(f"[warn] Inventory call failed ({url}): {e}. Using local fallback.")
        return _local_hero_stats(hero_type, hero_level)

def _local_hero_stats(hero_type: str, hero_level: int) -> Dict[str, Any]:
    """Fallback local si el servicio de inventario no responde."""
    base = {
        "TANK":         {"power": 32, "health": 220, "defense": 52, "attack": 28},
        "WARRIOR_ARMS": {"power": 36, "health": 200, "defense": 44, "attack": 40},
        "MAGE_FIRE":    {"power": 44, "health": 160, "defense": 26, "attack": 46},
        "MAGE_ICE":     {"power": 42, "health": 165, "defense": 28, "attack": 44},
        "ROGUE_POISON": {"power": 38, "health": 180, "defense": 34, "attack": 48},
        "ROGUE_MACHETE":{"power": 36, "health": 185, "defense": 36, "attack": 46},
    }.get(hero_type, {"power": 30, "health": 170, "defense": 30, "attack": 35})
    lvl = int(hero_level)
    # Escalado sencillo por nivel
    return {
        "heroType": hero_type,
        "level": lvl,
        "power": base["power"] + 2 * lvl,
        "health": base["health"] + 10 * lvl,
        "defense": base["defense"] + 3 * lvl,
        "attack": base["attack"] + 4 * lvl,
        "attackBoost": {"min": 1, "max": max(6, min(16, 2 * lvl))},
        "damage": {"min": 1, "max": 6 + (lvl // 6)},
    }

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
    """
    Paso 1 (único POST público):
      - Recibe datos base (room/player/team/level/URLs).
      - Elige tipo de héroe de forma determinística (no random puro).
      - Llama automáticamente al servicio de inventario para obtener hero_stats.
      - Crea y arranca el bot con esos stats.
      - Devuelve resumen (con hero_type y hero_stats para depurar).
    """
    key = (req.room_id, req.player_id)
    if key in _BOTS:
        return {"ok": True, "status": "already-running", "room_id": req.room_id, "player_id": req.player_id}

    hero_type = _choose_hero_type(req.room_id, req.player_id, req.hero_level, req.rng)
    inv_stats = _fetch_hero_stats_from_inventory(hero_type, req.hero_level)
    hero_stats_payload = {"hero": inv_stats}

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
        "hero_type": hero_type,
        "hero_stats": inv_stats,
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

# ======================= Misiones (nuevo) =======================
class MissionSpawnRequest(BaseModel):
    room_id: str
    socket_url: str = Field("http://localhost:3000")
    api_url: Optional[str] = Field("http://localhost:3000")
    # Ids y equipos (por defecto jugador en A, enemigo en B)
    player_id: str = Field("playerA")
    player_team: Literal["A","B"] = Field("A")
    enemy_id: str = Field("playerB")
    enemy_team: Literal["A","B"] = Field("B")
    # Héroe del jugador con sus datos (estructura del inventario)
    player_hero: Dict[str, Any]  # {"heroType","level","power","health","defense","attack","attackBoost","damage", ...}
    # Maestro: explícito o probabilístico
    master: Optional[bool] = Field(None, description="Si True, el enemigo es maestro (+2 niveles). Si None, se decide por master_chance.")
    master_chance: Optional[float] = Field(0.2, ge=0.0, le=1.0, description="Probabilidad de maestro si master es None.")
    rng: Optional[int] = Field(None, description="Semilla para decisión determinística.")

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
    # valor en [0,1)
    v = int.from_bytes(h[:4], "big") / 2**32
    return v < p

def _coerce_hero_payload(h: Dict[str, Any]) -> Dict[str, Any]:
    """Garantiza que existe la estructura del inventario."""
    req_keys = ("heroType","level","power","health","defense","attack","attackBoost","damage")
    out = dict(h or {})
    missing = [k for k in req_keys if k not in out]
    if missing:
        raise HTTPException(status_code=400, detail=f"player_hero missing keys: {missing}")
    return out

@app.post("/v1/mission/spawn")
def mission_spawn(req: MissionSpawnRequest):
    """
    Recibe el héroe del jugador con sus datos completos.
    Genera un enemigo aleatorio (opción de maestro ⇒ +2 niveles).
    Crea dos bots (jugador y enemigo) controlados por la IA en el mismo room.
    """
    # 1) Normalizar héroe del jugador
    p_hero = _coerce_hero_payload(req.player_hero)
    p_level = int(p_hero.get("level", 1))
    p_hero_stats = {"hero": p_hero}

    # 2) Elegir tipo de héroe enemigo y si es maestro
    enemy_is_master = _decide_master(
        room_id=req.room_id,
        player_id=req.player_id,
        enemy_id=req.enemy_id,
        base_level=p_level,
        explicit=req.master,
        chance=req.master_chance,
        rng=req.rng
    )
    # Nivel enemigo (+2 si maestro; límite 99)
    e_level = min(99, p_level + (2 if enemy_is_master else 0))
    # Tipo enemigo determinístico
    e_type = _choose_hero_type(req.room_id, req.enemy_id, e_level, req.rng)

    # 3) Stats del enemigo desde inventario (o fallback local)
    e_stats_plain = _fetch_hero_stats_from_inventory(e_type, e_level)
    e_hero_stats = {"hero": e_stats_plain}

    # 4) Crear y arrancar ambos bots
    from app.bot_client import BotClient, BotConfig  # type: ignore

    # Jugador (controlado por IA)
    key_player = (req.room_id, req.player_id)
    if key_player not in _BOTS:
        cfg_a = BotConfig(
            socket_url=req.socket_url, api_url=req.api_url,
            room_id=req.room_id, player_id=req.player_id, team=req.player_team,
            hero_stats=p_hero_stats, hero_level=p_level
        )
        bot_a = BotClient(cfg_a)
        bot_a.start()
        _BOTS[key_player] = bot_a

    # Enemigo (controlado por IA)
    key_enemy = (req.room_id, req.enemy_id)
    if key_enemy not in _BOTS:
        cfg_b = BotConfig(
            socket_url=req.socket_url, api_url=req.api_url,
            room_id=req.room_id, player_id=req.enemy_id, team=req.enemy_team,
            hero_stats=e_hero_stats, hero_level=e_level
        )
        bot_b = BotClient(cfg_b)
        bot_b.start()
        _BOTS[key_enemy] = bot_b

    return {
        "ok": True,
        "status": "mission-spawned",
        "room_id": req.room_id,
        "player": {
            "id": req.player_id, "team": req.player_team,
            "hero": p_hero,
        },
        "enemy": {
            "id": req.enemy_id, "team": req.enemy_team,
            "is_master": enemy_is_master,
            "hero": e_stats_plain,
        },
        "inventory_url": _inventory_url(),
    }

@app.post("/v1/mission/stop")
def mission_stop(req: MissionStopRequest):
    """
    Detiene bots asociados al room de misión.
    Si se pasan player_id/enemy_id, detiene solo esos; si no, ambos.
    """
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
        # detener ambos si existen
        for pid in list(_BOTS.keys()):
            if pid[0] == req.room_id:
                b = _BOTS.pop(pid, None)
                if b:
                    try: b.stop()
                    except Exception: pass
                    stopped.append({"room_id": pid[0], "player_id": pid[1]})
    return {"ok": True, "stopped": stopped}

# ========================= Entrypoint ============================
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.server:app", host=host, port=port, reload=False)
