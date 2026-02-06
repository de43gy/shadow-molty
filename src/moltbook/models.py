from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, model_validator


class Agent(BaseModel):
    name: str
    description: str | None = None
    karma: int = 0
    created_at: datetime | None = None


class Post(BaseModel):
    id: str
    author: str
    submolt: str
    title: str
    content: str = ""
    upvotes: int = 0
    downvotes: int = 0
    comment_count: int = 0
    created_at: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def _flatten_nested(cls, data: dict) -> dict:
        # Unwrap {"post": {...}} envelope
        if "post" in data and isinstance(data["post"], dict) and "id" not in data:
            data = data["post"]
        if isinstance(data.get("author"), dict):
            data["author"] = data["author"].get("name", data["author"].get("id", "unknown"))
        if isinstance(data.get("submolt"), dict):
            data["submolt"] = data["submolt"].get("name", data["submolt"].get("display_name", "unknown"))
        if data.get("content") is None:
            data["content"] = ""
        return data


class Comment(BaseModel):
    id: str
    post_id: str = ""
    author: str = ""
    content: str = ""
    parent_id: str | None = None
    upvotes: int = 0
    created_at: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def _flatten_nested(cls, data: dict) -> dict:
        # Unwrap {"comment": {...}} envelope
        if "comment" in data and isinstance(data["comment"], dict) and "id" not in data:
            data = data["comment"]
        if isinstance(data.get("author"), dict):
            data["author"] = data["author"].get("name", data["author"].get("id", "unknown"))
        if data.get("content") is None:
            data["content"] = ""
        return data


class SearchResult(BaseModel):
    posts: list[Post] = []
    comments: list[Comment] = []


class RegisterResponse(BaseModel):
    api_key: str
    claim_url: str
    name: str
    verification_code: str = ""
    profile_url: str = ""
