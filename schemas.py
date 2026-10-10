"""Схемы результатов операций."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class IndexingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunks: int = Field(ge=0)
    dimensions: int = Field(gt=0)
    inserted: int = Field(ge=0)
    updated: int = Field(ge=0)
    deleted: int = Field(ge=0)
    verified: bool


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    section_title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    distance: float = Field(allow_inf_nan=False)


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    answer: str = Field(min_length=1, max_length=3000)
    abstained: bool
    citations: list[Evidence]


class Source(Evidence):
    source_name: str
    section_title: str


class RagResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    status: Literal["answered", "insufficient_context", "invalid_output"]
    sources: list[Source]
    context: list[RetrievedChunk]
