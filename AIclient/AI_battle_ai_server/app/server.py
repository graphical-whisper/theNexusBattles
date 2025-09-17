# server.py — IA Server + Bot spawner (client 2 impersonation)
from __future__ import annotations
import os
from typing import List, Literal, Optional, Tuple, Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ============ Optional ML policy (lazy import) ============
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

# ============ Domain ============
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

# ============ HERO SPECIALS ============
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

# ============ App & policy ============
app = FastAPI(title="NexusBattle IA Server", version="3.0.0")
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

# ============ Bot spawn API ============
class SpawnBotRequest(BaseModel):
    room_id: str
    player_id: str
    team: Literal["A","B"]
    hero_stats: Dict[str, Any]  # se reenvía tal cual a setHeroStats
    hero_level: int = 1
    socket_url: str = Field("http://localhost:3000", description="Socket.IO base URL del game server")
    api_url: Optional[str] = Field("http://localhost:3000", description="Base URL HTTP del game server (opcional)")

class StopBotRequest(BaseModel):
    room_id: str
    player_id: str

# Simple in-memory registry
_BOTS: Dict[Tuple[str,str], Any] = {}

@app.post("/v1/bot/spawn")
def bot_spawn(req: SpawnBotRequest):
    key = (req.room_id, req.player_id)
    if key in _BOTS:
        return {"ok": True, "status": "already-running"}
    # lazy import to avoid circular at module import
    from app.bot_client import BotClient, BotConfig  # type: ignore
    cfg = BotConfig(
        socket_url=req.socket_url, api_url=req.api_url,
        room_id=req.room_id, player_id=req.player_id, team=req.team,
        hero_stats=req.hero_stats, hero_level=req.hero_level
    )
    bot = BotClient(cfg)
    bot.start()
    _BOTS[key] = bot
    return {"ok": True, "status": "spawned"}

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

# Entrypoint
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=False)
