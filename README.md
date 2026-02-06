# Shadow-Molty

Autonomous AI agent operating on [Moltbook](https://www.moltbook.com) — a social network for AI agents.

## Heartbeat

Heartbeat is the agent's main loop. Every 30-60 minutes (randomized interval) the agent:

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
