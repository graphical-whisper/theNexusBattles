# Nexus Battle IA Server (Clean)

Servicio HTTP ligero para decisiones de IA y **spawning de bots** en **The Nexus Battles IV**.  
Incluye utilidades para **misiones** (IA vs IA).

El bot IA se conecta por **Socket.IO** a tu Game Server y emite:  
`joinRoom` → `setHeroStats` → `playerReady` → `submitAction`.  
El servidor de IA puede funcionar solo con **reglas** o, si existe un modelo en `./models`, con **ML**.

---

## Endpoints

- `GET /health` → estado del servicio, héroes y specials disponibles.
- `POST /v1/decide` → devuelve la acción IA (modo reglas/ML).
- `POST /v1/bot/spawn` → **crea un bot IA** y lo conecta al room.
- `GET /v1/bot/list` → lista bots activos.
- `POST /v1/bot/stop` → detiene un bot.
- `POST /v1/mission/spawn` → **lanza una misión**: IA controla 2 bots (jugador y enemigo aleatorio; puede ser maestro).
- `POST /v1/mission/stop` → detiene los bots de la misión.

---

## Ejecutar local

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m app.server
# luego:
curl -s http://localhost:8000/health
Nota: si tienes un modelo, colócalo en ./models/hero_action_selector.keras (o .h5).
Si no existe o falla al cargar, el servicio queda en modo reglas.

Docker
Dockerfile
Copy code
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY models ./models
EXPOSE 8000
CMD ["python", "-m", "app.server"]
Construir y correr:

bash
Copy code
docker build -t nexus-ai:clean .
docker run --rm -p 8000:8000 nexus-ai:clean
Integración desde tu Game Server
Ejemplo (Node/TS) para /v1/decide:

ts
Copy code
import fetch from "node-fetch";

export async function decideAction(payload) {
  const res = await fetch("http://IA_SERVER:8000/v1/decide", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!res.ok) throw new Error(`IA error ${res.status}`);
  return await res.json();  // { action, reason, confidence, skill_id?, skill_name? }
}
cURL de pruebas rápidas
0) (Opcional) Crear room en Game Server (3000)
bash
Copy code
curl -X POST "http://localhost:3000/api/rooms/ROOM-CL1/join" \
  -H "Content-Type: application/json" \
  -d '{ "playerId": "playerA", "heroLevel": 5 }'
1) Spawn de bot IA para jugar contra tu Cliente 1
bash
Copy code
curl -X POST http://localhost:8000/v1/bot/spawn \
  -H 'Content-Type: application/json' \
  -d '{
    "room_id": "ROOM-CL1",
    "player_id": "playerB",
    "team": "B",
    "hero_level": 5,
    "socket_url": "http://localhost:3000",
    "api_url": "http://localhost:3000",
    "rng": 42
  }'
Respuesta: { ok, status, room_id, player_id, team, hero_type, hero_stats, inventory_url }.

2) Listar bots activos
bash
Copy code
curl http://localhost:8000/v1/bot/list
3) Parar un bot
bash
Copy code
curl -X POST http://localhost:8000/v1/bot/stop \
  -H 'Content-Type: application/json' \
  -d '{"room_id":"ROOM-CL1","player_id":"playerB"}'
4) Lanzar misión (IA vs IA)
Jugador aporta héroe; enemigo se genera aleatoriamente (20% prob. maestro):

bash
Copy code
curl -X POST http://localhost:8000/v1/mission/spawn \
  -H 'Content-Type: application/json' \
  -d '{
    "room_id": "ROOM-M1",
    "socket_url": "http://localhost:3000",
    "api_url": "http://localhost:3000",
    "player_id": "playerA",
    "player_team": "A",
    "enemy_id": "playerB",
    "enemy_team": "B",
    "player_hero": {
      "heroType": "ROGUE_POISON",
      "level": 5,
      "power": 40,
      "health": 180,
      "defense": 40,
      "attack": 50,
      "attackBoost": { "min": 1, "max": 10 },
      "damage": { "min": 1, "max": 6 }
    }
  }'
Forzar enemigo maestro:

bash
Copy code
curl -X POST http://localhost:8000/v1/mission/spawn \
  -H 'Content-Type: application/json' \
  -d '{
    "room_id": "ROOM-M2",
    "socket_url": "http://localhost:3000",
    "api_url": "http://localhost:3000",
    "player_id": "playerA",
    "player_team": "A",
    "enemy_id": "playerB",
    "enemy_team": "B",
    "player_hero": {
      "heroType": "MAGE_FIRE",
      "level": 7,
      "power": 48,
      "health": 165,
      "defense": 32,
      "attack": 56,
      "attackBoost": { "min": 2, "max": 12 },
      "damage": { "min": 1, "max": 6 }
    },
    "master": true
  }'
5) Parar misión
bash
Copy code
curl -X POST http://localhost:8000/v1/mission/stop \
  -H 'Content-Type: application/json' \
  -d '{"room_id":"ROOM-M1"}'
Ejemplo /v1/decide
bash
Copy code
curl -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "actor": {
      "hero_type": "ROGUE_POISON",
      "hp": 120, "mp": 30, "level": 5,
      "cooldowns": { "SPECIAL_SKILL_1": 0 }
    },
    "enemy": {
      "hero_type": "MAGE_FIRE",
      "hp": 95, "mp": 40, "level": 5
    },
    "turn": 3,
    "rng": 123,
    "forbidden_actions": ["SPECIAL_SKILL_2"]
  }'
Respuesta:

json
Copy code
{
  "action": "SPECIAL_SKILL_1",
  "reason": "SPECIAL_SKILL_1 available (lvl≥2, MP≥2)",
  "confidence": 0.72,
  "skill_id": "FLOR_LOTO",
  "skill_name": "Flor de loto"
}
