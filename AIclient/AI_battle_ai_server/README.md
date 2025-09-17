
# Nexus Battle IA Server (Clean)

Servicio HTTP ligero para decisiones de IA en **The Nexus Battles IV**.

## Endpoints

- `GET /health` → estado del servicio.
- `POST /v1/decide` → devuelve la acción a ejecutar.

## Ejecutar local

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m app.server
# luego:
curl -s http://localhost:8000/health
```

## Docker

```Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY models ./models
EXPOSE 8000
CMD ["python", "-m", "app.server"]
```

Construir y correr:

```bash
docker build -t nexus-ai:clean .
docker run --rm -p 8000:8000 nexus-ai:clean
```

## Integración desde tu Game Server

Ejemplo (Node/TS):

```ts
import fetch from "node-fetch";

export async function decideAction(payload) {
  const res = await fetch("http://IA_SERVER:8000/v1/decide", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!res.ok) throw new Error(`IA error ${res.status}`);
  return await res.json();  // { action, reason, confidence }
}
```

## Modelos

Coloca tu modelo en `./models/hero_action_selector.keras` (o `.h5`). Si no existe o falla al cargar, el servicio sigue corriendo con reglas.
