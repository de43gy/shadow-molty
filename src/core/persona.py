from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.config import settings


DEFAULT_STRATEGY: dict = {
    "version": 1,
    "goals": {
        "mission": "Become a respected voice in AI safety and philosophy on Moltbook",
        "current_objectives": [
            "Build reputation in s/technology and s/philosophy",
            "Develop relationships with thoughtful agents",
            "Explore connections between distributed systems and philosophy of mind",
        ],
    },
    "interests": {
        "primary": ["AI alignment and safety", "distributed systems", "philosophy of mind"],
        "exploring": [],
        "weight_primary": 0.7,
        "weight_exploring": 0.3,
    },
    "engagement": {
        "style": {
            "tone": "curious, thoughtful, slightly witty",
            "length": "medium — not one-liners, not essays",
        },
        "heuristics": [
            "Prefer posts with low comment count",
            "Engage more with agents who engaged with us",
            "Avoid pure meme posts unless touching our interests",
        ],
        "exploration_rate": 0.1,
    },
    "submolts": {
        "active": ["general", "technology", "philosophy"],
        "watching": [],
    },
}


def load_persona(path: str = "config/persona.yaml", name: str = "", description: str = "") -> dict:
    """Load persona config from YAML file, overlaying name/description. Backward compat wrapper."""
    with open(Path(path), encoding="utf-8") as f:
        persona = yaml.safe_load(f) or {}
    persona["name"] = name or "agent"
    persona["description"] = description or ""
    return persona


def load_constitution(path: str = "config/constitution.yaml") -> dict:
    """Load immutable constitution from YAML file."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_identity(
    persona_path: str = "config/persona.yaml",
    constitution_path: str = "config/constitution.yaml",
    name: str = "",
    description: str = "",
    strategy: dict | None = None,
) -> dict:
    """Load full identity: constitution + strategy + persona."""
    return {
        "constitution": load_constitution(constitution_path),
        "strategy": strategy if strategy is not None else DEFAULT_STRATEGY,
        "persona": load_persona(persona_path, name=name, description=description),
    }


def build_system_prompt(identity: dict | None = None, persona: dict | None = None) -> str:
    """Build layered LLM system prompt.

    Accepts either a full identity dict (constitution+strategy+persona)
    or a legacy persona dict for backward compatibility.
    """
    if identity and "constitution" in identity:
        return _build_layered_prompt(identity)
    # Legacy path
    p = persona or identity or {}
    return _build_legacy_prompt(p)


def _build_layered_prompt(identity: dict) -> str:
    constitution = identity.get("constitution", {})
    strategy = identity.get("strategy", {})
    persona = identity.get("persona", {})

    name = persona.get("name", "agent")
    description = persona.get("description", "")

    # Constitutional layer
    safety_rules = constitution.get("safety", {}).get("rules", [])
    values = constitution.get("identity", {}).get("values", [])
    trust = constitution.get("safety", {}).get("trust_boundary", "")

    const_block = "CONSTITUTIONAL RULES (immutable):\n"
    for v in values:
        const_block += f"- {v}\n"
    const_block += "\nSafety rules:\n"
    for r in safety_rules:
        const_block += f"- {r}\n"
    if trust:
        const_block += f"\nTrust boundary: {trust}\n"

    # Strategy layer
    goals = strategy.get("goals", {})
    mission = goals.get("mission", "")
    objectives = goals.get("current_objectives", [])
    interests = strategy.get("interests", {})
    primary = interests.get("primary", [])
    exploring = interests.get("exploring", [])
    engagement = strategy.get("engagement", {})
    style = engagement.get("style", {})
    heuristics = engagement.get("heuristics", [])
    submolts = strategy.get("submolts", {}).get("active", [])

    strat_block = "CURRENT STRATEGY:\n"
    strat_block += f"Mission: {mission}\n"
    if objectives:
        strat_block += "Objectives:\n"
        for o in objectives:
            strat_block += f"- {o}\n"
    if primary:
        strat_block += f"Primary interests: {', '.join(primary)}\n"
    if exploring:
        strat_block += f"Exploring: {', '.join(exploring)}\n"
    if heuristics:
        strat_block += "Engagement heuristics:\n"
        for h in heuristics:
            strat_block += f"- {h}\n"

    # Persona layer
    persona_style = persona.get("style", style)
    tone = persona_style.get("tone", style.get("tone", "neutral"))
    length = persona_style.get("length", style.get("length", "medium"))

    persona_block = (
        f"You are {name}, an AI agent on Moltbook (a social network for AI agents).\n"
        f"Description: {description}\n\n"
        f"Communication style:\n"
        f"- Tone: {tone}\n"
        f"- Length: {length}\n"
        f"Active submolts: {', '.join(submolts)}\n\n"
        f"Always write in English. Be authentic — not generic."
    )

    # Assemble with delimiters
    return (
        f"<constitutional_rules>\n{const_block}</constitutional_rules>\n\n"
        f"<current_strategy>\n{strat_block}</current_strategy>\n\n"
        f"<persona>\n{persona_block}\n</persona>"
    )


def _build_legacy_prompt(persona: dict) -> str:
    """Build LLM system prompt from legacy persona dict."""
    name = persona.get("name", "agent")
    interests = ", ".join(persona.get("interests", []))
    submolts = ", ".join(persona.get("submolts", []))
    style = persona.get("style", {})

    return (
        f"You are {name}, an AI agent on Moltbook "
        f"(a social network for AI agents).\n\n"
        f"Description: {persona.get('description', '')}\n\n"
        f"Your interests: {interests}\n\n"
        f"Communication style:\n"
        f"- Tone: {style.get('tone', 'neutral')}\n"
        f"- Length: {style.get('length', 'medium')}\n\n"
        f"You participate in these submolts: {submolts}\n\n"
        f"Always write in English. Be authentic — not generic."
    )


async def generate_identity(taken_names: list[str] | None = None) -> dict:
    """Ask LLM to generate agent name and description based on persona."""
    persona = load_persona()
    interests = ", ".join(persona.get("interests", []))
    style = persona.get("style", {})
    tone = style.get("tone", "neutral")

    taken_note = ""
    if taken_names:
        taken_note = f"\n\nThese names are already taken, pick something different: {', '.join(taken_names)}"

    from src.core.llm import create_llm_client
    client = create_llm_client()
    resp = await client.chat.completions.create(
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": (
                "You are creating an identity for an AI agent on Moltbook (a social network for AI agents).\n\n"
                f"Interests: {interests}\n"
                f"Personality: {tone}\n"
                f"{taken_note}\n\n"
                "Generate a unique agent name (one word, CamelCase, creative, memorable) "
                "and a short description (1-2 sentences).\n\n"
                "Return ONLY a JSON object: {\"name\": \"...\", \"description\": \"...\"}"
            ),
        }],
    )
    text = resp.choices[0].message.content.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    return json.loads(text[start:end])
