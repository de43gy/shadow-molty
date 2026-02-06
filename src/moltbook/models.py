from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel


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
    content: str
    upvotes: int = 0
    downvotes: int = 0
    comment_count: int = 0
    created_at: datetime | None = None


class Comment(BaseModel):
    id: str
    post_id: str
    author: str
    content: str
    parent_id: str | None = None
    upvotes: int = 0
    created_at: datetime | None = None


class SearchResult(BaseModel):
    posts: list[Post] = []
    comments: list[Comment] = []


class RegisterResponse(BaseModel):
    api_key: str
    claim_url: str
    name: str
    verification_code: str = ""
    profile_url: str = ""
