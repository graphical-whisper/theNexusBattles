# app/bot_client.py — Bot (client 2) con payload compatible, CD local y finisher
from __future__ import annotations
import threading, time, json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import socketio
import requests

from app.server import (  # type: ignore
    HERO_SPECIALS as IA_HERO_SPECIALS,
    DecideRequest, BattleSide, decide as ia_decide,
)

# ---------- helpers: alias y daño ----------
_HERO_ALIASES = {
    "POISON_ROGUE": "ROGUE_POISON",
    "MACHETE_ROGUE": "ROGUE_MACHETE",
    "FIRE_MAGE": "MAGE_FIRE",
    "ICE_MAGE": "MAGE_ICE",
    "ARMS_WARRIOR": "WARRIOR_ARMS",
    "WARRIOR_ARMS": "WARRIOR_ARMS",
    "TANK": "TANK",
}
def _alias_hero(ht: str) -> str:
    return _HERO_ALIASES.get((ht or "").upper(), (ht or "").upper())

def _avg_range(obj: Optional[Dict[str, Any]]) -> float:
    if not isinstance(obj, dict): return 0.0
    lo = float(obj.get("min", 0) or 0); hi = float(obj.get("max", 0) or 0)
    return max(0.0, (lo + hi) / 2.0)

def estimate_basic_damage(hero: Dict[str, Any]) -> float:
    # Conservador: expected = avg(damage) + 0.5 * avg(attackBoost)
    return max(0.0, _avg_range(hero.get("damage")) + 0.5 * _avg_range(hero.get("attackBoost")))

def _slot_of_skill(hero_type_alias: str, skill_id: Optional[str]) -> Optional[str]:
    if not skill_id: return None
    for s in IA_HERO_SPECIALS.get(hero_type_alias, []):
        if s.get("id") == skill_id:
            return s.get("slot")
    return None

# ---------- target / turns ----------
def other_player_id(turns: List[str], my_id: str) -> str:
    for u in turns:
        if isinstance(u, str) and u != my_id:
            return u
    return ""

# ---------- política con CD local + finisher ----------
def decide_action(me_hero: Dict[str, Any], foe_hero: Dict[str, Any],
                  hero_type_alias: str, foe_hero_alias: str,
                  blocked_slot: Optional[str]) -> Tuple[str, Dict[str, Any], Optional[str], str]:
    # Finisher con BASIC si alcanza
    enemy_hp = float(foe_hero.get("health", foe_hero.get("hp", 0)) or 0)
    my_basic = estimate_basic_damage(me_hero)
    if enemy_hp <= my_basic:
        return "BASIC_ATTACK", {}, None, f"Finisher BASIC ({enemy_hp} ≤ {my_basic:.1f})"

    cd_map, forbidden = {}, []
    if blocked_slot in ("SPECIAL_SKILL_1", "SPECIAL_SKILL_2", "SPECIAL_SKILL_3"):
        cd_map[blocked_slot] = 1
        forbidden = [blocked_slot]

    actor = BattleSide(
        hero_type=hero_type_alias,
        hp=float(me_hero.get("health", me_hero.get("hp", 0)) or 0),
        mp=float(me_hero.get("power",  me_hero.get("mp", 0)) or 0),
        level=int(me_hero.get("level", 1)),
        cooldowns=cd_map, buffs={}, debuffs={}
    )
    enemy = BattleSide(
        hero_type=foe_hero_alias or hero_type_alias,
        hp=float(foe_hero.get("health", foe_hero.get("hp", 0)) or 0),
        mp=float(foe_hero.get("power",  foe_hero.get("mp", 0)) or 0),
        level=int(foe_hero.get("level", 1)),
        cooldowns={}, buffs={}, debuffs={}
    )
    r = ia_decide(DecideRequest(actor=actor, enemy=enemy, turn=1, rng=None, forbidden_actions=forbidden))  # type: ignore

    # extracción segura del campo action (Pydantic v2 o dict)
    if hasattr(r, "action"):
        chosen = r.action
    elif hasattr(r, "model_dump"):
        chosen = r.model_dump().get("action")
    else:
        chosen = r["action"]

    if chosen in ("BASIC", "ATTACK"):
        return "BASIC_ATTACK", {}, None, "IA chose BASIC/ATTACK"

    if chosen in ("SPECIAL_SKILL_1", "SPECIAL_SKILL_2", "SPECIAL_SKILL_3"):
        idx = {"SPECIAL_SKILL_1":0,"SPECIAL_SKILL_2":1,"SPECIAL_SKILL_3":2}[chosen]
        skills = IA_HERO_SPECIALS.get(hero_type_alias, [])
        if 0 <= idx < len(skills):
            skill_id = skills[idx]["id"]
            return "SPECIAL_SKILL", {"skillId": skill_id}, chosen, f"IA chose {chosen} → {skill_id}"
        return "BASIC_ATTACK", {}, None, "No skill in slot → BASIC"
    return "BASIC_ATTACK", {}, None, f"Unknown IA action {chosen} → BASIC"

# ---------- Bot ----------
@dataclass
class BotConfig:
    socket_url: str
    api_url: Optional[str]
    room_id: str
    player_id: str
    team: str
    hero_stats: Dict[str, Any]   # esperado como {"hero": {...}}
    hero_level: int = 1

class BotClient:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.sio = socketio.Client(reconnection=True, logger=False, engineio_logger=False)
        self.turns: List[str] = []
        self.current_turn: Optional[str] = None
        self.finished = False
        self.last_special_slot: Optional[str] = None
        self._wire()

    def _wire(self):
        @self.sio.event
        def connect():
            print(f"[bot:{self.cfg.player_id}] connected")
            self.turns, self.current_turn = [], None
            self.last_special_slot = None
            # Igual que Client_2
            self.sio.emit("joinRoom", {
                "roomId": self.cfg.room_id,
                "player": {"id": self.cfg.player_id, "heroLevel": int(self.cfg.hero_level)}
            })
            if self.cfg.api_url:
                try:
                    url = f"{self.cfg.api_url.rstrip('/')}/api/rooms/{self.cfg.room_id}/join"
                    body = {"playerId": self.cfg.player_id, "heroLevel": int(self.cfg.hero_level), "heroStats": self.cfg.hero_stats}
                    requests.post(url, json=body, timeout=5)
                except Exception as e:
                    print(f"[bot:{self.cfg.player_id}] join POST failed: {e}")
            self.sio.emit("setHeroStats", { "roomId": self.cfg.room_id, "playerId": self.cfg.player_id, "stats": self.cfg.hero_stats })
            self.sio.emit("playerReady",  { "roomId": self.cfg.room_id, "playerId": self.cfg.player_id, "team": self.cfg.team })

        @self.sio.on("battleStarted")
        def on_battle_started(data):
            print(f"[bot:{self.cfg.player_id}] battleStarted")
            self.last_special_slot = None
            self.sio.emit("joinBattle", { "roomId": self.cfg.room_id, "playerId": self.cfg.player_id })
            battle = data if isinstance(data, dict) else {}
            self.turns = list(battle.get("turns") or [])
            self.current_turn = self.turns[0] if self.turns else battle.get("nextTurnPlayer")
            print(f"[bot:{self.cfg.player_id}] turns={self.turns} currentTurn={self.current_turn}")
            self._maybe_act(battle)

        @self.sio.on("actionResolved")
        def on_action_resolved(data):
            battle = data if isinstance(data, dict) else {}
            if isinstance(battle.get("turns"), list):
                self.turns = list(battle["turns"])
            before = self.current_turn
            self.current_turn = battle.get("nextTurnPlayer") or self.current_turn
            print(f"[bot:{self.cfg.player_id}] actionResolved → nextTurn={self.current_turn} (before={before}) turns={self.turns}")
            self._maybe_act(battle)

        @self.sio.on("battleEnded")
        def on_battle_ended(_data):
            self.finished = True
            print(f"[bot:{self.cfg.player_id}] battleEnded")

        @self.sio.event
        def disconnect():
            print(f"[bot:{self.cfg.player_id}] disconnected")

        @self.sio.event
        def connect_error(err):
            print(f"[bot:{self.cfg.player_id}] connect_error: {err}")

    # ====== actuar solo cuando sea mi turno ======
    def _maybe_act(self, battle_payload: Dict[str, Any]):
        if self.finished or self.current_turn != self.cfg.player_id:
            return

        # Hero propio del spawn (estructura esperada {"hero": {...}})
        hero = (self.cfg.hero_stats.get("hero") if isinstance(self.cfg.hero_stats, dict) else {}) or {}
        hero_type_alias = _alias_hero((hero.get("heroType") or hero.get("type") or "").upper())
        foe_hero_alias = hero_type_alias  # si no sabemos rival, alias propio

        me_full = {
            "level": hero.get("level", 1), "health": hero.get("health", 0), "power": hero.get("power", 0),
            "attack": hero.get("attack", 0), "defense": hero.get("defense", 0),
            "damage": hero.get("damage"), "attackBoost": hero.get("attackBoost")
        }
        foe_full = { "level": 1, "health": 999, "power": 0 }

        target_id = other_player_id(self.turns, self.cfg.player_id)
        if not target_id:
            print(f"[bot:{self.cfg.player_id}] no target in turns={self.turns}")
            return

        action_type, extra, used_slot, reason = decide_action(
            me_full, foe_full, hero_type_alias, foe_hero_alias, self.last_special_slot
        )
        print(f"[bot:{self.cfg.player_id}] act: {action_type} {extra} ({reason}) → target={target_id}")

        # ---- construir payload EXACTO que suele esperar el server ----
        server_action: Dict[str, Any] = {
            "type": action_type,                         # "BASIC_ATTACK" | "SPECIAL_SKILL"
            "sourcePlayerId": self.cfg.player_id,
            "targetPlayerId": target_id,
        }
        if action_type == "SPECIAL_SKILL":
            # slot del special (obligatorio para el server)
            slot = used_slot or _slot_of_skill(hero_type_alias, extra.get("skillId"))
            if not slot:
                slot = "SPECIAL_SKILL_1"
            server_action["skill"] = slot       # <— clave importante
            server_action["skillKey"] = slot    # alias visto en builds previas
            # metadato opcional (no debería romper)
            if "skillId" in extra:
                server_action["skillId"] = extra["skillId"]

        payload = { "roomId": self.cfg.room_id, "action": server_action }

        def _ack(*args, **kwargs):
            print(f"[bot:{self.cfg.player_id}] submitAction ack:", args or kwargs)

        # log visible del payload que se envía
        print(f"[bot:{self.cfg.player_id}] submitAction payload -> {json.dumps(payload, ensure_ascii=False)}")

        # emitir exactamente como el Client_2: { roomId, action }
        self.sio.emit("submitAction", payload, callback=_ack)

        # CD local: bloquear solo si usamos SPECIAL
        self.last_special_slot = used_slot if action_type == "SPECIAL_SKILL" else None

    # ====== ciclo de vida ======
    def start(self):
        t = threading.Thread(target=self._run, name=f"bot-{self.cfg.player_id}", daemon=True)
        t.start()

    def _run(self):
        url = self.cfg.socket_url.rstrip("/")
        conn_url = url if url.startswith("http") else "http://" + url
        self.sio.connect(conn_url)
        try:
            while not self.finished and self.sio.connected:
                time.sleep(0.25)
        finally:
            if self.sio.connected:
                self.sio.disconnect()

    def stop(self):
        self.finished = True
        if self.sio.connected:
            self.sio.disconnect()
