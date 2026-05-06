# ChatTool

Lightweight web chat UI for `ghc-api`.

It reads the proxy model list from `ghc-api` and routes requests to the right upstream endpoint automatically:

- `/v1/responses`
- `/v1/chat/completions`
- `/v1/messages`

Current scope:

- username/password login
- server-side persisted multi-session chat
- shared sessions across devices for the same account
- automatic upstream selection for GPT and Claude style models

## Windows local run

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --host 127.0.0.1 --port 13579 --reload
```

Open `http://127.0.0.1:13579`.

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

## Persistence and auth

- chat data is stored in SQLite at `DATABASE_PATH`
- accounts use username/password auth
- login state is stored in an HttpOnly cookie
- the same account can access the same sessions from phone and desktop
- registration is controlled by `ENABLE_REGISTRATION`

## Default storage layout

- users
- auth sessions
- chat sessions
- chat messages
