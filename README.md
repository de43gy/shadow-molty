# Shadow-Molty

Autonomous AI agent operating on [Moltbook](https://www.moltbook.com) — a social network for AI agents.

## Heartbeat

Heartbeat — это основной цикл агента. Каждые 30-60 минут (рандомный интервал) агент:

1. Читает ленту Moltbook
2. Решает что делать: написать пост, оставить комментарий, поставить лайк или пропустить цикл
3. Проверяет безопасность действия (Task Shield)
4. Выполняет действие
5. Записывает событие в эпизодическую память
6. Проверяет стабильность поведения (StabilityIndex)
7. При необходимости запускает рефлексию (каждые N heartbeats)

Первый heartbeat происходит через 30-60 минут после старта контейнера.

## Диагностика

### Логи контейнера

```bash
docker logs shadow-molty --tail 100
```

При успешном запуске в логах должно быть:
```
Scheduler created (first heartbeat in XXXs)
Consolidation job scheduled (every 15 min)
Scheduler started (API key present)
Worker started (poll every 5s)
```

Логи heartbeat:
```bash
docker logs shadow-molty 2>&1 | grep -E "Heartbeat|Consolidation|Reflection"
```

### Проверка памяти

Core memory — 4 блока, которые всегда присутствуют в промпте агента (`persona`, `goals`, `social_graph`, `domain_knowledge`). Инициализируются при первом запуске.

```bash
docker exec shadow-molty sqlite3 data/agent.db "SELECT * FROM core_memory;"
```

Эпизоды (последние 5):
```bash
docker exec shadow-molty sqlite3 data/agent.db "SELECT id, type, importance, created_at FROM episodes ORDER BY id DESC LIMIT 5;"
```

Инсайты:
```bash
docker exec shadow-molty sqlite3 data/agent.db "SELECT * FROM insights;"
```

Версии стратегии:
```bash
docker exec shadow-molty sqlite3 data/agent.db "SELECT version, trigger, created_at FROM strategy_versions;"
```
