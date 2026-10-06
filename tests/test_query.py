from datetime import date

import pytest
from pydantic import ValidationError

from georag.query import EOQuery


def test_query_requires_place_name_and_level_together() -> None:
    with pytest.raises(ValidationError, match="supplied together"):
        EOQuery(intent="flood", place_name="Kerala", interpretation="Flood request")


def test_query_rejects_reversed_inclusive_date_range() -> None:
    with pytest.raises(ValidationError, match="on or after"):
        EOQuery(intent="flood", start_date=date(2020, 2, 1), end_date=date(2020, 1, 1),
                interpretation="Flood request")


def test_query_retains_explicit_place_resolution_and_modality() -> None:
    query = EOQuery(
        intent="flood", place_name="Kerala", place_level="admin1", place_country="India",
        resolved_place_id="admin1:IN-KL",
        requested_modality="s2_12band", interpretation="Flood scenes in Kerala", original_query="find floods",
    )
    assert query.resolved_place_id == "admin1:IN-KL"
    assert query.place_country == "India"
    assert query.requested_modality == "s2_12band"


def test_query_cannot_resolve_a_place_without_a_name() -> None:
    with pytest.raises(ValidationError, match="requires a place_name"):
        EOQuery(intent="flood", resolved_place_id="admin1:IN-KL", interpretation="Flood request")


def test_structured_parser_returns_validated_schema_without_changing_user_text(monkeypatch) -> None:
    import georag.m10_service as service_module
    from georag.m10_service import parse_natural_language

    parsed = EOQuery(intent="flood", place_name="Kerala", place_level="admin1", place_country="India",
                     interpretation="Flood scenes in Kerala")

    class FakeResponse:
        output_parsed = parsed

    class FakeResponses:
        @staticmethod
        def parse(**kwargs):
            assert kwargs["text_format"] is EOQuery
            return FakeResponse()

    class FakeClient:
        responses = FakeResponses()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(service_module, "OpenAI", lambda: FakeClient())
    result = parse_natural_language("find floods in Kerala")
    assert result.original_query == "find floods in Kerala"
    assert result.place_name == "Kerala"
    assert result.place_country == "India"
    assert result.resolved_place_id is None


def test_structured_parser_requires_server_side_api_key(monkeypatch) -> None:
    from georag.m10_service import QueryParseError, parse_natural_language

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(QueryParseError, match="not configured"):
        parse_natural_language("find floods in Kerala")
