# train_ai.py — compatible con tu NDJSON (Keras 3, guarda .keras)
import os, json, re
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NDJSON_PATH = os.path.join(BASE_DIR, "data", "merged_turns.ndjson")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

ACTION_KINDS = ["BASIC_ATTACK", "SPECIAL_SKILL_1", "SPECIAL_SKILL_2", "SPECIAL_SKILL_3"]
ACTION_TO_ID = {a:i for i,a in enumerate(ACTION_KINDS)}

def get_path(d, path):
    cur = d
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur

def try_keys(d, paths):
    for p in paths:
        v = get_path(d, p)
        if v is not None:
            return v
    return None

def norm_action_kind(v):
    if v is None: return None
    s = str(v).upper().strip()
    if s in {"BASIC","BASIC_ATTACK","ATTACK_BASIC","BASICO","BASIC-ATTACK"}:
        return "BASIC_ATTACK"
    if s in {"SPECIAL","SPECIAL_SKILL","SPECIAL1","SPECIAL_1","S1"}:
        return "SPECIAL_SKILL_1"
    if s in {"SPECIAL2","SPECIAL_2","S2"}:
        return "SPECIAL_SKILL_2"
    if s in {"SPECIAL3","SPECIAL_3","S3"}:
        return "SPECIAL_SKILL_3"
    m = re.search(r"SPECIAL[_\-\s]*SKILL[_\-\s]*(\d+)", s) or re.search(r"SPECIAL[_\-\s]*(\d+)", s)
    if m:
        k = min(max(int(m.group(1)),1),3)
        return f"SPECIAL_SKILL_{k}"
    return None

str_id_cache = {}
def str_id(s: str) -> int:
    if s not in str_id_cache:
        str_id_cache[s] = len(str_id_cache)+1
    return str_id_cache[s]

def f2(x):
    try: return float(x)
    except: return 0.0

reasons = Counter()
X, y = [], []

if not os.path.isfile(NDJSON_PATH):
    raise FileNotFoundError(f"No se encontró: {NDJSON_PATH}")

total = 0
with open(NDJSON_PATH, "r", encoding="utf-8") as f:
    for line in f:
        total += 1
        line = line.strip()
        if not line:
            reasons["linea_vacia"] += 1
            continue
        try:
            row = json.loads(line)
        except:
            reasons["json_invalido"] += 1
            continue

        # hero type (no obligatorio)
        heroType = try_keys(row, [
            ("actor_hero_type",),      # tu campo top-level
            ("actor","type"),          # y dentro de actor
            ("heroType",), ("actor","heroType"),
        ])

        # nivel / poder (sí, estos son críticos para el modelo)
        level = try_keys(row, [("actor","level"), ("level",), ("actor","lvl")])
        power = try_keys(row, [("actor","power"), ("power",), ("energy",), ("mana",)])
        if level is None:
            reasons["sin_level"] += 1
            continue
        if power is None:
            reasons["sin_power"] += 1
            continue

        # stats opcionales
        attack  = try_keys(row, [("actor","attack"), ("attack",)]) or 0
        defense = try_keys(row, [("actor","defense"), ("defense",)]) or 0
        health  = try_keys(row, [("actor","hp"), ("health",), ("actor","health")]) or 0
        enemy_health = try_keys(row, [("enemy_hp_after",), ("enemy","hp"), ("enemy","health"), ("enemy_health",)]) or 0
        enemy_def    = try_keys(row, [("enemy","defense",), ("enemy_defense",)]) or 0

        # acción elegida
        action_kind = try_keys(row, [
            ("chosen_action_kind",), ("action_kind",),
            ("action","kind"), ("chosen_action","kind"),
            ("decision_kind",), ("decision","kind"),
        ])
        action_kind = norm_action_kind(action_kind)
        if action_kind is None:
            reasons["sin_action_kind"] += 1
            continue

        # encode features
        hero_type_id = heroType if isinstance(heroType, (int,float)) else (str_id(f"hero::{heroType}") if heroType else 0)
        try:
            level = int(level)
        except:
            reasons["level_invalido"] += 1
            continue

        X.append([
            float(hero_type_id),
            float(level),
            f2(power),
            f2(attack),
            f2(defense),
            f2(health),
            f2(enemy_health),
            f2(enemy_def),
        ])
        y.append(ACTION_TO_ID[action_kind])

X = np.asarray(X, dtype=np.float32)
if X.size == 0:
    print(f"\n[DIAGNÓSTICO] {NDJSON_PATH}")
    print(f"  Líneas: {total}")
    print("  Motivos de descarte:", dict(reasons))
    raise SystemExit("Sin ejemplos válidos.")

y = keras.utils.to_categorical(np.asarray(y), num_classes=len(ACTION_KINDS))

# split (más flexible si pocos datos)
test_size = 0.15 if X.shape[0] >= 20 else 0.25
strat = y if X.shape[0] >= 40 else None
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42, stratify=strat)

# modelo
inp = keras.Input(shape=(X.shape[1],), name="features")
h = layers.Dense(64, activation="relu")(inp)
h = layers.Dense(64, activation="relu")(h)
out = layers.Dense(len(ACTION_KINDS), activation="softmax")(h)
model = keras.Model(inp, out)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_tr, y_tr, validation_data=(X_te, y_te), epochs=8, batch_size=256, verbose=2)

# guardar
model_path = os.path.join(MODEL_DIR, "hero_action_selector.keras")
model.save(model_path)
with open(os.path.join(MODEL_DIR, "metadata.json"), "w", encoding="utf-8") as mf:
    json.dump({
        "features": ["hero_type_id","level","power","attack","defense","health","enemy_health","enemy_def"],
        "action_kinds": ACTION_KINDS,
        "string_id_map": str_id_cache
    }, mf, ensure_ascii=False, indent=2)

print("\n[OK] Modelo guardado en:", model_path)
print("[OK] Metadatos en:", os.path.join(MODEL_DIR, "metadata.json"))
print(f"[INFO] Muestras usadas: {X.shape[0]}  |  Clases:", Counter(np.argmax(y, axis=1)))

