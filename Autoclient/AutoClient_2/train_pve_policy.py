#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_pve_policy.py
-------------------
Entrena una política (clasificador) de IA PvE a partir del dataset generado por los clientes Node.
- Acepta NDJSON o CSV (detecta por extensión).
- Preprocesa features (numéricas + one-hot de tipos de héroe).
- Entrena una MLP (PyTorch) para predecir la acción elegida: BASIC_ATTACK o skillId (SPECIAL/MASTER).
- Guarda artefactos en ./artifacts: modelo .pt, encoder.pkl, label_mappings.json, metrics.json y opcional ONNX.
- Incluye función de inferencia offline (arg --predict-sample).

Requisitos:
    pip install torch scikit-learn pandas numpy onnx onnxruntime

Uso:
    python train_pve_policy.py --data data/pve_turns.ndjson --out artifacts --epochs 40 --batch-size 256 --export-onnx
    python train_pve_policy.py --data data/pve_turns.csv     --out artifacts --predict-sample
"""
from __future__ import annotations
import os, sys, json, argparse, math, random, pickle, pathlib
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import onnx
    import onnxruntime as ort  # type: ignore
    HAS_ONNX = True
except Exception:
    HAS_ONNX = False


# ------------------------------
# Utilidades de E/S
# ------------------------------

def load_ndjson(path: str) -> pd.DataFrame:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_any(path: str) -> pd.DataFrame:
    if path.endswith('.ndjson') or path.endswith('.jsonl'):
        return load_ndjson(path)
    elif path.endswith('.csv'):
        return pd.read_csv(path)
    else:
        # intenta NDJSON por defecto
        return load_ndjson(path)


# ------------------------------
# Mapeos de acciones (grupo por ID)
# Deben reflejar los IDs que usa el servidor/cliente Node.
# ------------------------------

SPECIAL_KIND: Dict[str, str] = {
    # Tank
    "GOLPE_ESCUDO": "offense",
    "MANO_PIEDRA": "defense",
    "DEFENSA_FEROZ": "defense",
    # Warrior Arms
    "EMBATE_SANGRIENTO": "offense",
    "LANZA_DIOSES": "offense",
    "GOLPE_TORMENTA": "offense",
    # Mage Fire
    "MISILES_MAGMA": "offense",
    "VULCANO": "offense",
    "PARED_FUEGO": "defense",
    # Mage Ice
    "LLUVIA_HIELO": "offense",
    "CONO_HIELO": "defense",
    "BOLA_HIELO": "defense",
    # Rogue Poison
    "FLOR_LOTO": "offense",
    "AGONIA": "offense",
    "PIQUETE": "offense",
    # Rogue Machete
    "CORTADA": "offense",
    "MACHETAZO": "offense",
    "PLANAZO": "offense",
    # Shaman
    "TOQUE_VIDA": "support",
    "VINCULO_NATURAL": "support",
    "CANTO_BOSQUE": "support",
    # Medic
    "CURACION_DIRECTA": "support",
    "NEUTRALIZACION_EFECTOS": "support",
    "REANIMACION": "support",
}

MASTER_KIND: Dict[str, str] = {
    "MASTER.TANK_GOLPE_DEFENSA": "offense",
    "MASTER.ARMS_SEGUNDO_IMPULSO": "support",
    "MASTER.FIRE_LUZ_CEGADORA": "defense",
    "MASTER.ICE_FRIO_CONCENTRADO": "defense",
    "MASTER.VENENO_TOMA_LLEVA": "offense",
    "MASTER.MACHETE_INTIMIDACION_SANGRIENTA": "offense",
    "MASTER.SHAMAN_TE_CHANGUA": "support",
    "MASTER.MEDIC_REANIMADOR_3000": "support",
}


# ------------------------------
# Preparación de dataset
# ------------------------------

NUM_FEATURES_BASE = [
    # estado del actor
    "actor.hp_pct", "actor.power", "actor.attack", "actor.defense",
    "actor.dmg_min", "actor.dmg_max",
    "actor.atk_boost_min", "actor.atk_boost_max",
    "actor.pct_damage", "actor.pct_crit", "actor.pct_evade",
    "actor.pct_resist", "actor.pct_escape", "actor.pct_negate",
    # estado del enemigo
    "enemy.hp_pct", "enemy.power", "enemy.attack", "enemy.defense",
    "enemy.dmg_min", "enemy.dmg_max",
    "enemy.atk_boost_min", "enemy.atk_boost_max",
    "enemy.pct_damage", "enemy.pct_crit", "enemy.pct_evade",
    "enemy.pct_resist", "enemy.pct_escape", "enemy.pct_negate",
    # máscara (conteos de acciones válidas)
    "num_specials_valid_off", "num_specials_valid_def", "num_support_valid",
    "num_masters_valid_off", "num_masters_valid_def", "num_masters_support",
]

CAT_FEATURES = [
    "actor.type", "enemy.type"
]

# features derivadas útiles
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["diff.hp_pct"]  = (df["actor.hp_pct"]  - df["enemy.hp_pct"]).astype(float)
    df["diff.power"]   = (df["actor.power"]   - df["enemy.power"]).astype(float)
    df["diff.attack"]  = (df["actor.attack"]  - df["enemy.attack"]).astype(float)
    df["diff.defense"] = (df["actor.defense"] - df["enemy.defense"]).astype(float)
    return df


DERIVED_FEATURES = ["diff.hp_pct", "diff.power", "diff.attack", "diff.defense"]


def build_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Construye etiqueta: BASIC o skillId (SPECIAL/MASTER)
    y = df["chosen_skill_id"].fillna("BASIC_ATTACK").astype(str)

    # Garantiza columnas requeridas
    for col in NUM_FEATURES_BASE + CAT_FEATURES:
        if col not in df.columns:
            df[col] = 0

    # Tipos/casts básicos
    for col in NUM_FEATURES_BASE:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in CAT_FEATURES:
        df[col] = df[col].fillna("UNKNOWN").astype(str)

    # Derivadas
    df = add_derived_features(df)

    # Filtra filas con etiqueta no vacía
    mask = y.notna() & (y != "")
    X = df[NUM_FEATURES_BASE + DERIVED_FEATURES + CAT_FEATURES][mask]
    y = y[mask]

    return X, y


# ------------------------------
# Modelo MLP en PyTorch
# ------------------------------

class MLPPolicy(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int, dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------
# Entrenamiento
# ------------------------------

def compute_class_weights(y_idx: np.ndarray, num_classes: int) -> torch.Tensor:
    # inverso de la frecuencia (clamp para estabilidad)
    counts = np.bincount(y_idx, minlength=num_classes).astype(np.float64)
    weights = 1.0 / np.clip(counts, 1.0, None)
    weights = weights * (num_classes / weights.sum())
    return torch.tensor(weights, dtype=torch.float32)


def train_model(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xva: np.ndarray, yva: np.ndarray,
    in_dim: int, out_dim: int,
    hidden: List[int],
    epochs: int = 40, batch_size: int = 256, lr: float = 1e-3, device: str = "cpu"
) -> Tuple[MLPPolicy, Dict[str, Any]]:

    model = MLPPolicy(in_dim, hidden, out_dim, dropout=0.15).to(device)

    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    yva_t = torch.tensor(yva, dtype=torch.long, device=device)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)

    class_weights = compute_class_weights(ytr, out_dim).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0
    best_state = None
    patience = 6
    bad = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # --- train
        model.train()
        perm = torch.randperm(Xtr_t.size(0), device=device)
        losses = []
        for i in range(0, Xtr_t.size(0), batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # --- val
        model.eval()
        with torch.no_grad():
            v_logits = model(Xva_t)
            v_loss = criterion(v_logits, yva_t).item()
            v_pred = v_logits.argmax(dim=1)
            v_acc = (v_pred == yva_t).float().mean().item()

        tl = float(np.mean(losses)) if losses else 0.0
        history["train_loss"].append(tl)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        print(f"[{epoch:03d}] train_loss={tl:.4f}  val_loss={v_loss:.4f}  val_acc={v_acc:.4f}")

        # early stopping
        if v_acc > best_val_acc + 1e-4:
            best_val_acc = v_acc
            best_state = model.state_dict()
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch}. Best val_acc={best_val_acc:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_val_acc": best_val_acc, **history}


# ------------------------------
# Inferencia offline + guardrails C.A-2/C.A-3 (opcional)
# ------------------------------

def apply_guardrails(logits: np.ndarray, labels: List[str], state: Dict[str, Any]) -> np.ndarray:
    """
    Aplica restricciones:
      - C.A-2: si actor.hp_pct < 0.40 prioriza defensivas/curación (si el llamador provee listas válidas).
      - C.A-3: filtra acciones no válidas (cooldown>0, power insuficiente), si el llamador provee el conjunto válido.
    Para mantener el script autocontenido, este método espera que 'state' traiga opcionalmente:
        state["valid_ids"]: Set[str] de skillIds válidos + "BASIC_ATTACK"
        state["defensive_ids"]: Set[str] de skillIds defensivos/curación (según catálogo) + (opcionalmente) "BASIC_ATTACK"
        state["actor.hp_pct"]: float entre 0 y 1
    Si no se proveen, no se aplica el filtro (la política sigue siendo válida como clasificador puro).
    """
    mask = np.ones_like(logits, dtype=bool)

    valid_ids = set(state.get("valid_ids") or [])
    if valid_ids:
        # construye máscara de validez
        mask = np.array([(lab in valid_ids) for lab in labels], dtype=bool)

    # C.A-2: priorizar defensivas si hp<0.4 y existen en 'defensive_ids'
    try:
        hp = float(state.get("actor.hp_pct", 1.0))
    except Exception:
        hp = 1.0

    if hp < 0.40 and state.get("defensive_ids"):
        def_ids = set(state["defensive_ids"])
        pref_mask = np.array([(lab in def_ids) for lab in labels], dtype=bool)
        if pref_mask.any():
            # mezcla: si hay defensivas válidas, anulamos las demás
            mask = mask & pref_mask

    # Si la máscara deja todo en False (poco probable), no aplicar (evitar NaNs)
    if not mask.any():
        mask = np.ones_like(mask, dtype=bool)

    out = logits.copy()
    out[~mask] = -1e9  # suprime acciones inválidas
    return out


def predict_label(model: MLPPolicy, enc: Pipeline, label_list: List[str], row_dict: Dict[str, Any], device: str = "cpu",
                  guardrails_state: Optional[Dict[str, Any]] = None) -> Tuple[str, float]:
    df = pd.DataFrame([row_dict])
    X_df, _ = build_dataframe(df.assign(chosen_skill_id="BASIC_ATTACK"))
    X_enc = enc.transform(X_df).astype(np.float32)
    xt = torch.tensor(X_enc, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(xt).cpu().numpy()[0]

    if guardrails_state is not None:
        logits = apply_guardrails(logits, label_list, guardrails_state)

    probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
    idx = int(np.argmax(probs))
    return label_list[idx], float(probs[idx])


# ------------------------------
# Main
# ------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Ruta a NDJSON o CSV (dataset de turnos)")
    parser.add_argument("--out", default="artifacts", help="Directorio de salida de artefactos")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--predict-sample", action="store_true", help="Tras entrenar, predice sobre la primera fila")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)

    print(f"[data] leyendo: {args.data}")
    df = load_any(args.data)

    # Construcción de X/y
    X_df, y = build_dataframe(df)

    if X_df.empty or y.empty:
        print("[x] Dataset vacío después de filtrar. ¿Seguro que el NDJSON/CSV tiene las columnas esperadas?")
        sys.exit(1)

    print(f"[data] filas totales: {len(X_df)}  clases distintas: {y.nunique()}")

    # ColumnTransformer: one-hot para categóricas, standardize numéricas
    num_cols = [c for c in X_df.columns if c in (NUM_FEATURES_BASE + DERIVED_FEATURES)]
    cat_cols = [c for c in X_df.columns if c in CAT_FEATURES]

    pre = ColumnTransformer(transformers=[
        ("num", StandardScaler(with_mean=True, with_std=True), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
    ])

    # Ajuste del encoder con TODO el dataset (para evitar categorías no vistas entre train/val)
    pre.fit(X_df)

    X_all = pre.transform(X_df).astype(np.float32)

    # Label mapping
    labels = sorted(y.astype(str).unique().tolist())
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    y_idx = y.map(label_to_idx).values.astype(np.int64)

    # Split estratificado
    Xtr, Xva, ytr, yva = train_test_split(X_all, y_idx, test_size=args.test_size, random_state=args.seed, stratify=y_idx)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}  in_dim={X_all.shape[1]}  out_dim={len(labels)}  hidden={args.hidden}  batch={args.batch_size}  epochs={args.epochs}")

    model, history = train_model(
        Xtr, ytr, Xva, yva,
        in_dim=X_all.shape[1], out_dim=len(labels), hidden=args.hidden,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device
    )

    # Métricas finales en validación
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xva, dtype=torch.float32, device=device))
        pred = logits.argmax(dim=1).cpu().numpy()
        val_acc = float((pred == yva).mean())

    # Guardar artefactos
    # Pipeline de preprocesado (ColumnTransformer) + mapeos de label
    enc_path = os.path.join(args.out, "policy_encoder.pkl")
    with open(enc_path, "wb") as f:
        pickle.dump({"pre": pre, "num_cols": num_cols, "cat_cols": cat_cols}, f)
    print(f"[save] encoder -> {enc_path}")

    model_path = os.path.join(args.out, "policy_model.pt")
    torch.save({"state_dict": model.state_dict(), "in_dim": X_all.shape[1], "out_dim": len(labels), "hidden": args.hidden}, model_path)
    print(f"[save] model -> {model_path}")

    labels_path = os.path.join(args.out, "label_mappings.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({"labels": labels, "label_to_idx": label_to_idx}, f, ensure_ascii=False, indent=2)
    print(f"[save] labels -> {labels_path}")

    metrics_path = os.path.join(args.out, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"val_acc": val_acc, **history}, f, ensure_ascii=False, indent=2)
    print(f"[save] metrics -> {metrics_path} (val_acc={val_acc:.4f})")

    # Export ONNX (opcional)
    if args.export_onnx and HAS_ONNX:
        onnx_path = os.path.join(args.out, "policy_model.onnx")
        dummy = torch.randn(1, X_all.shape[1], dtype=torch.float32, device=device)
        model.eval()
        torch.onnx.export(model, dummy, onnx_path,
                          input_names=["input"], output_names=["logits"],
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
        print(f"[save] onnx -> {onnx_path}")
    elif args.export_onnx and not HAS_ONNX:
        print("[warn] --export-onnx solicitado, pero onnx/onnxruntime no está instalado. Omite export.")

    # Predicción de prueba
    if args.predict_sample:
        first_row = df.iloc[0].to_dict()
        # Prepara estado mínimo para guardrails (opcional)
        guard = {
            "actor.hp_pct": float(first_row.get("actor", {}).get("hp_pct", first_row.get("actor.hp_pct", 1.0))) if isinstance(first_row.get("actor"), dict) else float(first_row.get("actor.hp_pct", 1.0))
        }
        # Carga encoder
        with open(enc_path, "rb") as f:
            enc_obj = pickle.load(f)
        pre2: ColumnTransformer = enc_obj["pre"]
        # Reconstruir modelo
        ckpt = torch.load(model_path, map_location=device)
        mdl = MLPPolicy(ckpt["in_dim"], ckpt["hidden"], ckpt["out_dim"]).to(device)
        mdl.load_state_dict(ckpt["state_dict"])

        label_info = json.load(open(labels_path, "r", encoding="utf-8"))
        labels_list = label_info["labels"]

        # Construir features de fila (aplanada)
        # Si df es NDJSON con estructura anidada 'actor'/'enemy', ya vienen aplanadas por el logger propuesto.
        # Si no, intentamos usar las columnas planas existentes.
        flat = {}
        for k in (NUM_FEATURES_BASE + DERIVED_FEATURES + CAT_FEATURES):
            if k in df.columns:
                flat[k] = first_row.get(k)
            else:
                # si viene como dict anidado (actor:{hp_pct:...})
                if '.' in k:
                    left, right = k.split('.', 1)
                    if isinstance(first_row.get(left), dict):
                        flat[k] = first_row[left].get(right)
        # Completar faltantes
        for k in NUM_FEATURES_BASE:
            flat.setdefault(k, 0.0)
        for k in CAT_FEATURES:
            flat.setdefault(k, "UNKNOWN")
        # Derivadas
        tmpdf = pd.DataFrame([flat])
        tmpdf = add_derived_features(tmpdf)
        flat.update(tmpdf.iloc[0].to_dict())

        # Predice
        lab, prob = predict_label(mdl, pre2, labels_list, flat, device=device, guardrails_state=guard)
        print(f"[predict-sample] acción predicha: {lab}  (p={prob:.3f})")


if __name__ == "__main__":
    main()
