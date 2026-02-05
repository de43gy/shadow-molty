# Shadow-Molty

Autonomous AI agent operating on Moltbook (social network for AI agents).

## Language conventions

- Chat/discussion with the developer: **Russian**
- Agent's public persona (posts, comments on Moltbook): **English**
- Code comments: **English, minimal** — only where logic isn't self-evident
- Variable/function names: English

## Architecture

```
Agent Core (Python 3.11+, async)
├── Moltbook API Client (httpx)
├── Telegram Bot (aiogram 3.x) — owner-only control panel
├── LLM Brain (Anthropic Claude API) — decision-making
├── Task Queue — owner tasks + autonomous tasks
├── Storage (SQLite/JSON) — state persistence
└── Scheduler (APScheduler) — heartbeat & periodic jobs
```

## Key behaviors

- Heartbeat every 30-60 min (randomized): check feed, interact, post
- Rate limits: 1 post/30min, 1 comment/20sec, 50 comments/day
- Telegram bot: /status, /search, /ask, /watch, /digest, /post, /pause, /resume
- Only TELEGRAM_OWNER_ID can use the bot

## Project structure

- `src/moltbook/` — API client & models (implemented)
- `src/agent/` — core logic, persona, LLM brain, task queue
- `src/telegram/` — bot setup & command handlers
- `src/storage/` — persistence layer
- `config/persona.yaml` — agent persona config
- `data/` — runtime data (gitignored)

## Dev notes

- Base URL: `https://www.moltbook.com/api/v1` (must use `www.`)
- Auth: `Authorization: Bearer <MOLTBOOK_API_KEY>`
- Env vars: see `.env.example`
