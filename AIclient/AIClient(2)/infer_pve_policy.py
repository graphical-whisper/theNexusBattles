#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_pve_policy.py
-------------------
Inferencia de la política PvE entrenada (artefactos generados por train_pve_policy.py).
Devuelve la estructura requerida con *fallback* a BASIC_ATTACK:

Salida requerida (JSON):
{
  "type": "BASIC_ATTACK" | "SPECIAL_SKILL" | "MASTER_SKILL",
  "skillId": "<ID>" (omitido si BASIC_ATTACK),
  "targetId": "<playerId>"
}

Uso:
  python infer_pve_policy.py --state state.json --artifacts artifacts
  # state.json debe incluir al menos:
  {
    "actor_id": "playerA",
    "enemy_id": "playerB",
    "actor": {
      "type": "MAGE_ICE",
      "hp_pct": 0.35,
      "power": 6, "attack": 10, "defense": 10,
      "dmg_min": 3, "dmg_max": 5,
      "atk_boost_min": 0, "atk_boost_max": 0,
      "pct_damage": 55, "pct_crit": 10, "pct_evade": 5,
      "pct_resist": 10, "pct_escape": 0, "pct_negate": 20
    },
    "enemy": {
      "type": "WARRIOR_ARMS",
      "hp_pct": 0.62,
      "power": 4, "attack": 11, "defense": 10,
      "dmg_min": 4, "dmg_max": 6,
      "atk_boost_min": 0, "atk_boost_max": 0,
      "pct_damage": 55, "pct_crit": 10, "pct_evade": 5,
      "pct_resist": 10, "pct_escape": 0, "pct_negate": 20
    },
    "valid": {
      "basic": true,
      "specials": [{"id":"LLUVIA_HIELO","cooldown":0,"powerCost":2,"isAvailable":true},
                   {"id":"CONO_HIELO","cooldown":0,"powerCost":6,"isAvailable":true}],
      "masters":  [{"id":"MASTER.ICE_FRIO_CONCENTRADO","cooldown":0,"isAvailable":true}]
    }
  }

Requisitos:
  pip install torch scikit-learn pandas numpy
"""
from __future__ import annotations
import os, json, argparse, pickle
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# --- Mismos mappings que en el entrenamiento ---
SPECIAL_KIND = {
    "GOLPE_ESCUDO": "offense",
    "MANO_PIEDRA": "defense",
    "DEFENSA_FEROZ": "defense",
    "EMBATE_SANGRIENTO": "offense",
    "LANZA_DIOSES": "offense",
    "GOLPE_TORMENTA": "offense",
    "MISILES_MAGMA": "offense",
    "VULCANO": "offense",
    "PARED_FUEGO": "defense",
    "LLUVIA_HIELO": "offense",
    "CONO_HIELO": "defense",
    "BOLA_HIELO": "defense",
    "FLOR_LOTO": "offense",
    "AGONIA": "offense",
    "PIQUETE": "offense",
    "CORTADA": "offense",
    "MACHETAZO": "offense",
    "PLANAZO": "offense",
    "TOQUE_VIDA": "support",
    "VINCULO_NATURAL": "support",
    "CANTO_BOSQUE": "support",
    "CURACION_DIRECTA": "support",
    "NEUTRALIZACION_EFECTOS": "support",
    "REANIMACION": "support",
}
MASTER_KIND = {
    "MASTER.TANK_GOLPE_DEFENSA": "offense",
    "MASTER.ARMS_SEGUNDO_IMPULSO": "support",
    "MASTER.FIRE_LUZ_CEGADORA": "defense",
    "MASTER.ICE_FRIO_CONCENTRADO": "defense",
    "MASTER.VENENO_TOMA_LLEVA": "offense",
    "MASTER.MACHETE_INTIMIDACION_SANGRIENTA": "offense",
    "MASTER.SHAMAN_TE_CHANGUA": "support",
    "MASTER.MEDIC_REANIMADOR_3000": "support",
}
SELF_CAST = set([
    # defensivas/soporte que se lanzan a sí mismo
    "DEFENSA_FEROZ","MANO_PIEDRA","PARED_FUEGO",
    "TOQUE_VIDA","VINCULO_NATURAL","CANTO_BOSQUE",
    "CURACION_DIRECTA","NEUTRALIZACION_EFECTOS","REANIMACION",
    "MASTER.FIRE_LUZ_CEGADORA","MASTER.ICE_FRIO_CONCENTRADO",
    "MASTER.ARMS_SEGUNDO_IMPULSO","MASTER.SHAMAN_TE_CHANGUA","MASTER.MEDIC_REANIMADOR_3000"
])

# --- Columnas esperadas (mismas que train_pve_policy) ---
NUM_FEATURES_BASE = [
    "actor.hp_pct", "actor.power", "actor.attack", "actor.defense",
    "actor.dmg_min", "actor.dmg_max",
    "actor.atk_boost_min", "actor.atk_boost_max",
    "actor.pct_damage", "actor.pct_crit", "actor.pct_evade",
    "actor.pct_resist", "actor.pct_escape", "actor.pct_negate",
    "enemy.hp_pct", "enemy.power", "enemy.attack", "enemy.defense",
    "enemy.dmg_min", "enemy.dmg_max",
    "enemy.atk_boost_min", "enemy.atk_boost_max",
    "enemy.pct_damage", "enemy.pct_crit", "enemy.pct_evade",
    "enemy.pct_resist", "enemy.pct_escape", "enemy.pct_negate",
    "num_specials_valid_off", "num_specials_valid_def", "num_support_valid",
    "num_masters_valid_off", "num_masters_valid_def", "num_masters_support",
]
CAT_FEATURES = ["actor.type", "enemy.type"]

DERIVED_FEATURES = ["diff.hp_pct", "diff.power", "diff.attack", "diff.defense"]

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["diff.hp_pct"]  = (df["actor.hp_pct"]  - df["enemy.hp_pct"]).astype(float)
    df["diff.power"]   = (df["actor.power"]   - df["enemy.power"]).astype(float)
    df["diff.attack"]  = (df["actor.attack"]  - df["enemy.attack"]).astype(float)
    df["diff.defense"] = (df["actor.defense"] - df["enemy.defense"]).astype(float)
    return df

class MLPPolicy(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(0.15)]
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def load_artifacts(artifacts_dir: str):
    with open(os.path.join(artifacts_dir, "policy_encoder.pkl"), "rb") as f:
        enc_obj = pickle.load(f)
    pre = enc_obj["pre"]
    ckpt = torch.load(os.path.join(artifacts_dir, "policy_model.pt"), map_location="cpu")
    model = MLPPolicy(ckpt["in_dim"], ckpt["hidden"], ckpt["out_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    labels = json.load(open(os.path.join(artifacts_dir, "label_mappings.json"), "r", encoding="utf-8"))["labels"]
    return pre, model, labels

def flatten_state_to_row(state: Dict[str, Any]) -> Dict[str, Any]:
    """Convierte el 'state' (actor/enemy/valid) en un row_dict con las columnas esperadas."""
    actor = state.get("actor", {})
    enemy = state.get("enemy", {})
    def g(side, key, default=0.0): return float(side.get(key, default)) if isinstance(side.get(key, None), (int, float)) else float(side.get(key, default) or 0.0)
    def s(side, key, default="UNKNOWN"): return str(side.get(key, default) or default)

    row = {
        # actor
        "actor.hp_pct": g(actor, "hp_pct"), "actor.power": g(actor, "power"),
        "actor.attack": g(actor, "attack"), "actor.defense": g(actor, "defense"),
        "actor.dmg_min": g(actor, "dmg_min"), "actor.dmg_max": g(actor, "dmg_max"),
        "actor.atk_boost_min": g(actor, "atk_boost_min"), "actor.atk_boost_max": g(actor, "atk_boost_max"),
        "actor.pct_damage": g(actor, "pct_damage"), "actor.pct_crit": g(actor, "pct_crit"),
        "actor.pct_evade": g(actor, "pct_evade"), "actor.pct_resist": g(actor, "pct_resist"),
        "actor.pct_escape": g(actor, "pct_escape"), "actor.pct_negate": g(actor, "pct_negate"),
        "actor.type": s(actor, "type"),
        # enemy
        "enemy.hp_pct": g(enemy, "hp_pct"), "enemy.power": g(enemy, "power"),
        "enemy.attack": g(enemy, "attack"), "enemy.defense": g(enemy, "defense"),
        "enemy.dmg_min": g(enemy, "dmg_min"), "enemy.dmg_max": g(enemy, "dmg_max"),
        "enemy.atk_boost_min": g(enemy, "atk_boost_min"), "enemy.atk_boost_max": g(enemy, "atk_boost_max"),
        "enemy.pct_damage": g(enemy, "pct_damage"), "enemy.pct_crit": g(enemy, "pct_crit"),
        "enemy.pct_evade": g(enemy, "pct_evade"), "enemy.pct_resist": g(enemy, "pct_resist"),
        "enemy.pct_escape": g(enemy, "pct_escape"), "enemy.pct_negate": g(enemy, "pct_negate"),
        "enemy.type": s(enemy, "type"),
        # mask counts (derivados de 'valid')
        "num_specials_valid_off": 0, "num_specials_valid_def": 0, "num_support_valid": 0,
        "num_masters_valid_off": 0, "num_masters_valid_def": 0, "num_masters_support": 0,
    }

    # Derivar conteos a partir de valid.specials/masters
    valid = state.get("valid", {})
    specials = [x for x in valid.get("specials", []) if x.get("isAvailable", True) and (x.get("cooldown", 0) == 0)]
    masters  = [x for x in valid.get("masters", [])  if x.get("isAvailable", True) and (x.get("cooldown", 0) == 0)]
    for s in specials:
        gk = SPECIAL_KIND.get(str(s.get("id", "")), None)
        if gk == "offense": row["num_specials_valid_off"] += 1
        elif gk == "defense": row["num_specials_valid_def"] += 1
        elif gk == "support": row["num_support_valid"] += 1
    for m in masters:
        gk = MASTER_KIND.get(str(m.get("id", "")), None)
        if gk == "offense": row["num_masters_valid_off"] += 1
        elif gk == "defense": row["num_masters_valid_def"] += 1
        elif gk == "support": row["num_masters_support"] += 1

    # Derivadas
    df = pd.DataFrame([row])
    df = add_derived_features(df)
    row.update(df.iloc[0].to_dict())
    return row

def encode_row(pre, row_dict: Dict[str, Any]) -> np.ndarray:
    # Garantiza columnas requeridas
    for col in NUM_FEATURES_BASE:
        row_dict.setdefault(col, 0.0)
    for col in CAT_FEATURES:
        row_dict.setdefault(col, "UNKNOWN")
    # Construir DF y transformar
    X_df = pd.DataFrame([row_dict])[NUM_FEATURES_BASE + DERIVED_FEATURES + CAT_FEATURES]
    X_enc = pre.transform(X_df).astype(np.float32)
    return X_enc

def apply_guardrails(logits: np.ndarray, labels: List[str], valid_ids: Optional[set], defensive_ids: Optional[set], hp_pct: float) -> np.ndarray:
    mask = np.ones_like(logits, dtype=bool)
    if valid_ids:
        mask = np.array([(lab in valid_ids) for lab in labels], dtype=bool)
    if hp_pct < 0.40 and defensive_ids:
        pref = np.array([(lab in defensive_ids) for lab in labels], dtype=bool)
        if pref.any():
            mask = mask & pref
    if not mask.any():
        mask = np.ones_like(mask, dtype=bool)
    out = logits.copy()
    out[~mask] = -1e9
    return out

def decide_action(state: Dict[str, Any], artifacts_dir: str) -> Dict[str, Any]:
    """Devuelve { type, (skillId), targetId } con fallback a BASIC_ATTACK."""
    pre, model, labels = load_artifacts(artifacts_dir)
    row = flatten_state_to_row(state)
    X = encode_row(pre, row)

    xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        logits = model(xt).numpy()[0]

    # Construir máscaras de validez
    valid_cfg = state.get("valid", {})
    valid_ids = set()
    if valid_cfg.get("basic", True):
        valid_ids.add("BASIC_ATTACK")
    for s in valid_cfg.get("specials", []):
        if s.get("isAvailable", True) and s.get("cooldown", 0) == 0:
            valid_ids.add(str(s.get("id", "")))
    for m in valid_cfg.get("masters", []):
        if m.get("isAvailable", True) and m.get("cooldown", 0) == 0:
            valid_ids.add(str(m.get("id", "")))

    # Defensivas/soporte (para priorizar con hp<40%)
    defensive_ids = set()
    for s in valid_cfg.get("specials", []):
        sid = str(s.get("id", ""))
        if SPECIAL_KIND.get(sid) in ("defense", "support"):
            defensive_ids.add(sid)
    for m in valid_cfg.get("masters", []):
        mid = str(m.get("id", ""))
        if MASTER_KIND.get(mid) in ("defense", "support"):
            defensive_ids.add(mid)
    hp = float(row.get("actor.hp_pct", 1.0))

    logits = apply_guardrails(logits, labels, valid_ids, defensive_ids, hp)

    # Predicción
    probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
    idx = int(np.argmax(probs))
    pred_label = labels[idx]

    # --- Fallbacks robustos ---
    # 1) Si la predicción no es válida este turno -> BASIC_ATTACK si es posible
    if pred_label not in valid_ids:
        pred_label = "BASIC_ATTACK" if "BASIC_ATTACK" in valid_ids else pred_label

    # 2) Si seguimos sin una acción válida (caso extremo), forzar BASIC_ATTACK
    if pred_label not in valid_ids:
        pred_label = "BASIC_ATTACK"

    # --- Construir salida requerida ---
    actor_id = state.get("actor_id") or "playerA"
    enemy_id = state.get("enemy_id") or "playerB"

    if pred_label == "BASIC_ATTACK":
        action = {
            "type": "BASIC_ATTACK",
            "targetId": enemy_id  # ataque básico dirigido al enemigo
        }
        return action

    # Tipo por prefijo
    if pred_label.startswith("MASTER."):
        action_type = "MASTER_SKILL"
    else:
        action_type = "SPECIAL_SKILL"

    # Targets: self para skills típicamente defensivas/soporte
    target_id = actor_id if (pred_label in SELF_CAST) else enemy_id

    action = {
        "type": action_type,
        "skillId": pred_label,
        "targetId": target_id
    }
    return action

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="JSON con estado del turno")
    ap.add_argument("--artifacts", default="artifacts", help="Directorio con policy_model.pt, policy_encoder.pkl, label_mappings.json")
    args = ap.parse_args()

    with open(args.state, "r", encoding="utf-8") as f:
        state = json.load(f)

    out = decide_action(state, args.artifacts)
    print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
