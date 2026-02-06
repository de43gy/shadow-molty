from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    moltbook_api_key: str = Field(default="", alias="MOLTBOOK_API_KEY")
    telegram_bot_token: str = Field(alias="TELEGRAM_BOT_TOKEN")
    telegram_owner_id: int = Field(alias="TELEGRAM_OWNER_ID")
    anthropic_api_key: str = Field(alias="ANTHROPIC_API_KEY")

    llm_model: str = Field(default="claude-sonnet-4-5-20250929", alias="LLM_MODEL")

    moltbook_base_url: str = "https://www.moltbook.com/api/v1"

    # Rate limits
    post_cooldown_sec: int = 1800  # 30 min
    comment_cooldown_sec: int = 20
    max_comments_per_day: int = 50

    # Autonomous behavior
    heartbeat_min_sec: int = 1800  # 30 min
    heartbeat_max_sec: int = 3600  # 60 min

    # Reflection & consolidation
    reflection_every_n_heartbeats: int = 10
    consolidation_interval_min: int = 15
    episode_compression_age_hours: int = 48
    episode_compression_importance_threshold: float = 5.0


settings = Settings()
