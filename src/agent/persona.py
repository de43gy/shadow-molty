from __future__ import annotations

from pathlib import Path

import yaml

from src.config import settings


def load_persona(path: str = "config/persona.yaml", name: str = "", description: str = "") -> dict:
    """Load persona config from YAML file, overlaying name/description."""
    with open(Path(path), encoding="utf-8") as f:
        persona = yaml.safe_load(f) or {}
    persona["name"] = name or settings.agent_name or "agent"
    persona["description"] = description or settings.agent_description
    return persona


def build_system_prompt(persona: dict) -> str:
    """Build LLM system prompt from persona dict."""
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
