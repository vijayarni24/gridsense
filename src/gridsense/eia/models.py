"""Pydantic schema for EIA v2 fuel-type generation data.

Kept separate from the client so downstream consumers (BigQuery loader, RAG,
analysis agents) can import the same typed contract.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GenerationRecord(BaseModel):
    """One row from electricity/rto/fuel-type-data.

    EIA sends hyphenated keys (``respondent-name``) and hour-only periods
    (``2025-06-01T00``); we normalize both here so the rest of the codebase
    sees clean snake_case fields and real datetimes.
    """

    model_config = ConfigDict(populate_by_name=True)

    period: datetime
    respondent: str
    respondent_name: str = Field(alias="respondent-name")
    fueltype: str
    type_name: str = Field(alias="type-name")
    value: float | None = None
    value_units: str | None = Field(default=None, alias="value-units")

    @field_validator("period", mode="before")
    @classmethod
    def _coerce_hourly_period(cls, v: object) -> object:
        # EIA hourly periods are "YYYY-MM-DDTHH" (no minutes) — pad so ISO parsing works.
        if isinstance(v, str) and len(v) == 13 and v[10] == "T":
            return f"{v}:00:00"
        return v


class EIAResponse(BaseModel):
    """The inner ``response`` object of an EIA v2 payload.

    ``total`` arrives as a string in some responses; pydantic coerces it to int.
    Unknown metadata fields are ignored.
    """

    total: int
    data: list[GenerationRecord]
