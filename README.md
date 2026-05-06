# ChatTool

Lightweight web chat UI for `ghc-api`.

It reads the proxy model list from `ghc-api` and routes requests to the right upstream endpoint automatically:

- `/v1/responses`
- `/v1/chat/completions`
- `/v1/messages`

Current UI scope:

- model picker
- multi-session chat
- browser-only local persistence

## Current behavior

- model list comes from `http://127.0.0.1:8313/v1/models/full/`
- `gpt-5` and `o*` models prefer `/v1/responses`
- `claude*` models prefer `/v1/messages` when supported
- all other models default to `/v1/chat/completions`
- UI does not expose provider, temperature, or system prompt

## Windows local run

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --host 127.0.0.1 --port 13579 --reload
```

Open `http://127.0.0.1:13579`.

Prerequisites:

- `ghc-api` is already running on `http://127.0.0.1:8313`
- `http://127.0.0.1:8313/v1/models/full/` is reachable

## Linux deployment

```bash
cd /opt/chattool
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 127.0.0.1 --port 13579
```

Optional: run it with `systemd`, see `deploy/chattool.service.example`.

## Upstream proxy modes

`ChatTool` can talk to `ghc-api` in three deployment modes:

- `PROXY_TARGET=local`: always use `LOCAL_PROXY_BASE_URL`
- `PROXY_TARGET=tailscale`: always use `TAILSCALE_PROXY_BASE_URL`
- `PROXY_TARGET=auto`: try local first, then fall back to the Tailscale URL

If `OPENAI_BASE_URL` or `ANTHROPIC_BASE_URL` is set explicitly, that explicit value takes precedence.

## Sessions

- sessions are stored in browser `localStorage`
- there is no database
- sessions are lost when you switch browser/device or clear site data
