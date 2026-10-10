"""Схемы результатов операций."""

from pydantic import BaseModel, ConfigDict, Field


class IndexingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunks: int = Field(ge=0)
    dimensions: int = Field(gt=0)
    inserted: int = Field(ge=0)
    updated: int = Field(ge=0)
    deleted: int = Field(ge=0)
    verified: bool
