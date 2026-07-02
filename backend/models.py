"""Pydantic models for API requests/responses and internal data structures."""
from __future__ import annotations

from pydantic import BaseModel, Field


class GradeRequest(BaseModel):
    """Incoming request to grade a student's assignment text."""

    student_name: str = Field(default="Student", description="Label for the report")
    answer_text: str = Field(..., min_length=1, description="Raw assignment text")


class Evidence(BaseModel):
    """A single retrieved book chunk used as supporting evidence."""

    chapter: str
    text: str
    similarity: float


class CriterionScore(BaseModel):
    """Score + justification for one rubric criterion."""

    criterion: str
    max_marks: int
    awarded_marks: float
    justification: str
    book_reference: str | None = None


class GradeReport(BaseModel):
    """Full structured grading report returned to the frontend."""

    student_name: str
    criteria: list[CriterionScore]
    total_score: float
    max_score: int
    overall_feedback: str
    flags: list[str] = Field(default_factory=list)
    retrieved_evidence: list[Evidence] = Field(default_factory=list)
    injection_detected: bool = False
