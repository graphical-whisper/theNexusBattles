# server.py — IA de batalla (Keras 3 listo) | Specials por nivel | Fallback básico
import os, json
from typing import List, Literal, Optional, Tuple

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow import keras  # Keras 3 ok para .keras/.h5

# ================== Rutas robustas ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODEL_DIR, "hero_action_selector.keras")  # archivo .keras
META_PATH  = os.path.join(MODEL_DIR, "metadata.json")

app = FastAPI(title="NexusBattleAI", version="1.3.0")

# ================== Metadatos ==================
with open(META_PATH, "r", encoding="utf-8") as f:
    META = json.load(f)

FEATURES: List[str] = META["features"]  # ["hero_type_id","level","power","attack","defense","health","enemy_health","enemy_def"]
ACTION_KINDS: List[str] = META["action_kinds"]  # ["BASIC_ATTACK","SPECIAL_SKILL_1","SPECIAL_SKILL_2","SPECIAL_SKILL_3"]
STRING_ID_MAP = META.get("string_id_map", {})   # {"hero::TANK": <id>, ...}

# ================== Carga de modelo (Keras 3) ==================
def load_inference_model(model_file: str, features_count: int):
    """
    1) Intenta cargar .keras (recomendado para Keras 3).
    2) Si no existe, intenta envolver carpeta SavedModel con TFSMLayer.
    """
    if os.path.isfile(model_file):
        return keras.models.load_model(model_file)  # ✅ Keras 3 soporta .keras/.h5

    # Fallback: carpeta SavedModel (hero_action_selector/)
    stem = os.path.join(MODEL_DIR, "hero_action_selector")
    if os.path.isdir(stem):
        from keras.layers import TFSMLayer  # del paquete keras 3
        layer = TFSMLayer(stem, call_endpoint="serving_default")
        inp = keras.Input(shape=(features_count,), dtype="float32", name="features")
        out = layer(inp)
        if isinstance(out, dict):  # algunos SavedModel devuelven dict
            out = list(out.values())[0]
        return keras.Model(inp, out)

    raise FileNotFoundError(f"No encontré modelo en {model_file!r} ni carpeta SavedModel en {stem!r}")

model = load_inference_model(MODEL_FILE, features_count=len(FEATURES))

# ================== Catálogo de specials (sin Shaman/Medic) ==================
HERO_SPECIALS = {
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

# ================== Utilidades de catálogo / reglas ==================
def canonical_hero_name(ht) -> str:
    if isinstance(ht, str):
        return ht.strip().upper()
    return f"ID::{ht}"

def skill_row(hero_type: str | int, kind: str) -> Optional[dict]:
    rows = HERO_SPECIALS.get(canonical_hero_name(hero_type))
    if not rows: return None
    for r in rows:
        if r["slot"] == kind:
            return r
    return None

def skill_cost(hero_type: str | int, kind: str) -> int:
    r = skill_row(hero_type, kind)
    if r and "cost" in r:
        return int(r["cost"])
    defaults = {"BASIC_ATTACK": 0, "SPECIAL_SKILL_1": 20, "SPECIAL_SKILL_2": 35, "SPECIAL_SKILL_3": 50}
    return defaults.get(kind, 9999)

def meets_level(hero_type: str | int, kind: str, level: int) -> bool:
    r = skill_row(hero_type, kind)
    return bool(r and level >= int(r.get("level_req", 99)))

def allowed_actions_by_level(level: int) -> List[str]:
    if level <= 1:
        return ["BASIC_ATTACK"]
    elif 2 <= level <= 4:
        return ["BASIC_ATTACK", "SPECIAL_SKILL_1"]
    elif 5 <= level <= 7:
        return ["BASIC_ATTACK", "SPECIAL_SKILL_1", "SPECIAL_SKILL_2"]
    else:
        return ["BASIC_ATTACK", "SPECIAL_SKILL_1", "SPECIAL_SKILL_2", "SPECIAL_SKILL_3"]

def resolve_skill_for_hero(hero_type, kind: str) -> Tuple[Optional[int], Optional[str]]:
    r = skill_row(hero_type, kind)
    if not r: return None, None
    sid = abs(hash(r["id"])) % (10**7)  # ID numérico estable
    return sid, r["name"]

def str_to_id(s) -> int:
    return STRING_ID_MAP.get(s, 0)

# ================== Esquemas ==================
ActionKind = Literal["BASIC_ATTACK", "SPECIAL_SKILL_1", "SPECIAL_SKILL_2", "SPECIAL_SKILL_3"]

class ActorState(BaseModel):
    heroType: str | int
    level: int
    power: float
    attack: float = 0
    defense: float = 0
    health: float = 0

class EnemyState(BaseModel):
    health: float = 0
    defense: float = 0

class PredictRequest(BaseModel):
    actor: ActorState
    enemy: EnemyState = EnemyState()
    forbidden_actions: Optional[List[ActionKind]] = None

class PredictResponse(BaseModel):
    kind: ActionKind
    skillId: Optional[int] = None
    target: Literal["ENEMY", "SELF"] = "ENEMY"
    reason: str

# ================== Features & Predicción ==================
def build_features(actor: ActorState, enemy: EnemyState) -> np.ndarray:
    # Usa el mismo mapeo que en el entrenamiento
    hero_type_id = actor.heroType if isinstance(actor.heroType, (int, float)) else str_to_id(f"hero::{actor.heroType}")
    feats = [
        float(hero_type_id),
        float(actor.level),
        float(actor.power),
        float(actor.attack),
        float(actor.defense),
        float(actor.health),
        float(enemy.health),
        float(enemy.defense),
    ]
    return np.asarray([feats], dtype=np.float32)

def model_predict(x: np.ndarray) -> np.ndarray:
    """Compatibilidad con Model o TFSMLayer envuelto."""
    try:
        y = model.predict(x, verbose=0)
    except Exception:
        y = model(x, training=False)
    if isinstance(y, dict):
        y = list(y.values())[0]
    y = np.array(y)
    if y.ndim == 1:
        y = y[None, :]
    return y

# ================== Reglas + Fallback ==================
def pick_with_rules(pred_kind: str, actor: ActorState, forbidden: set[str]) -> Tuple[str, str]:
    allowed = set(allowed_actions_by_level(actor.level))

    def valid(kind: str) -> bool:
        if kind not in allowed: return False
        if kind in forbidden: return False
        if kind == "BASIC_ATTACK": return True
        if not meets_level(actor.heroType, kind, actor.level): return False
        return actor.power >= skill_cost(actor.heroType, kind)

    if valid(pred_kind):
        return pred_kind, "predicted_and_valid"

    # Degradación ordenada
    for k in ["SPECIAL_SKILL_1", "SPECIAL_SKILL_2", "SPECIAL_SKILL_3", "BASIC_ATTACK"]:
        if valid(k):
            return (k, "fallback_basic" if k == "BASIC_ATTACK" else f"fallback_{k.lower()}")

    return "BASIC_ATTACK", "force_basic_last_resort"

# ================== Endpoints ==================
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    x = build_features(req.actor, req.enemy)
    probs = model_predict(x)[0]
    order = np.argsort(probs)[::-1]
    forb = set(req.forbidden_actions or [])
    reason_steps = []

    for idx in order:
        kind = ACTION_KINDS[idx]
        chosen, reason = pick_with_rules(kind, req.actor, forb)
        reason_steps.append(f"try {kind} -> {reason}")
        if chosen == kind and reason == "predicted_and_valid":
            skill_id = None
            if chosen != "BASIC_ATTACK":
                skill_id, _ = resolve_skill_for_hero(req.actor.heroType, chosen)
                if skill_id is None:
                    chosen, reason = "BASIC_ATTACK", "no_slot_for_hero->basic"
                    reason_steps.append("resolve_skill_failed")
                    return PredictResponse(kind=chosen, target="ENEMY", reason=" | ".join(reason_steps), skillId=None)
            return PredictResponse(kind=chosen, target="ENEMY", reason=" | ".join(reason_steps), skillId=skill_id)

    chosen, reason = pick_with_rules("BASIC_ATTACK", req.actor, forb)
    reason_steps.append(f"force BASIC -> {reason}")
    return PredictResponse(kind=chosen, target="ENEMY", reason=" | ".join(reason_steps), skillId=None)

@app.get("/health")
def health():
    return {
        "ok": True,
        "model_file": os.path.basename(MODEL_FILE),
        "actions": ACTION_KINDS,
        "features": FEATURES,
        "heroes": list(HERO_SPECIALS.keys()),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
