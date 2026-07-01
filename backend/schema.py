"""Pydantic models for LLM extraction output validation."""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError, field_validator

__all__ = [
    "ActionItem",
    "Decision",
    "OpenQuestion",
    "ExtractionResult",
    "ValidationError",
]


class ActionItem(BaseModel):
    owner: str = Field(..., min_length=1)
    task: str = Field(..., min_length=1)
    deadline: str | None = None
    source_message_id: str = Field(..., min_length=1)

    @field_validator("owner", "task", "source_message_id")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v.strip()


class Decision(BaseModel):
    decision: str = Field(..., min_length=1)
    made_by: str = Field(..., min_length=1)
    source_message_id: str = Field(..., min_length=1)

    @field_validator("decision", "made_by", "source_message_id")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v.strip()


class OpenQuestion(BaseModel):
    asker: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    answered: bool = False
    source_message_id: str = Field(..., min_length=1)

    @field_validator("asker", "question", "source_message_id")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v.strip()


class ExtractionResult(BaseModel):
    """Strict schema the Ollama JSON response must conform to."""

    action_items: list[ActionItem] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)
