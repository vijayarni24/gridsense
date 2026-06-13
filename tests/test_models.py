"""Schema tests: alias mapping, period coercion, envelope parsing."""

from __future__ import annotations

from gridsense.eia.models import EIAResponse, GenerationRecord

RAW_ROW = {
    "period": "2025-06-01T00",
    "respondent": "CAISO",
    "respondent-name": "California Independent System Operator",
    "fueltype": "SUN",
    "type-name": "Solar",
    "value": "1234.5",
    "value-units": "megawatthours",
}


def test_generation_record_maps_aliases_and_parses_period() -> None:
    rec = GenerationRecord.model_validate(RAW_ROW)
    assert rec.respondent_name == "California Independent System Operator"
    assert rec.type_name == "Solar"
    assert rec.value == 1234.5
    assert rec.period.year == 2025 and rec.period.hour == 0


def test_record_allows_null_value() -> None:
    rec = GenerationRecord.model_validate({**RAW_ROW, "value": None})
    assert rec.value is None


def test_envelope_coerces_string_total() -> None:
    env = EIAResponse.model_validate({"total": "1", "data": [RAW_ROW]})
    assert env.total == 1
    assert len(env.data) == 1
