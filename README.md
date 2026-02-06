# Shadow-Molty

Autonomous AI agent operating on [Moltbook](https://www.moltbook.com) — a social network for AI agents.

## Heartbeat

Heartbeat is the agent's main loop. Every 30-60 minutes (randomized interval) the agent runs a two-phase cycle:

### Phase 1: Obligations

Before doing anything creative, the agent handles its social duties:

**Reply to comments on own posts.** The agent checks its recent posts (last 48 hours, up to 10) for new comments. It replies to top-level comments and direct replies to its own comments in a thread — but ignores conversations between other agents under its post. Up to 2 replies per heartbeat (oldest first) to preserve the daily comment budget (50/day). Each reply is generated with full thread context and memory recall about the commenter.

**Check DMs.** A lightweight `dm/check` call returns early if there's no activity. If there is:
- Pending DM requests are auto-approved (the agent is social). The owner gets a Telegram notification for each new conversation.
- For conversations with unread messages, the agent generates a reply. If the LLM decides the conversation needs human input (collaboration proposals, private info, anything uncertain), it flags the conversation and escalates to the owner via Telegram. The owner can reply manually with `/dm_reply`. Once the owner responds, the flag is cleared and the agent resumes auto-replying.

### Phase 2: Autonomous action

1. Reads the Moltbook feed
2. Decides what to do: write a post, leave a comment, upvote, or skip the cycle
3. Validates the action against safety rules (Task Shield)
4. Executes the action
5. Records the event to episodic memory
6. Checks behavioral stability (StabilityIndex)
7. Triggers reflection if needed (every N heartbeats)

The first heartbeat occurs 30-60 minutes after the container starts.

## Diagnostics

### Container logs

```bash
docker logs shadow-molty --tail 100
```

On successful startup the logs should contain:
```
Scheduler created (first heartbeat in XXXs)
Consolidation job scheduled (every 15 min)
Scheduler started (API key present)
Worker started (poll every 5s)
```

Filter heartbeat-related logs:
```bash
docker logs shadow-molty 2>&1 | grep -E "Heartbeat|Consolidation|Reflection"
```

### Memory inspection

Core memory consists of 4 blocks always present in the agent's prompt (`persona`, `goals`, `social_graph`, `domain_knowledge`). Initialized on first launch.

```bash
docker exec shadow-molty sqlite3 data/agent.db "SELECT * FROM core_memory;"
```

Recent episodes (last 5):
```bash
docker exec shadow-molty sqlite3 data/agent.db "SELECT id, type, importance, created_at FROM episodes ORDER BY id DESC LIMIT 5;"
```

Insights:
```bash
docker exec shadow-molty sqlite3 data/agent.db "SELECT * FROM insights;"
```

Strategy versions:
```bash
docker exec shadow-molty sqlite3 data/agent.db "SELECT version, trigger, created_at FROM strategy_versions;"
```
