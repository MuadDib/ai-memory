"""Privacy filter tests."""
from __future__ import annotations

from ai_memory.privacy import redact


def test_no_secrets_passes_through() -> None:
    result = redact("This is a perfectly innocuous sentence.")
    assert result.text == "This is a perfectly innocuous sentence."
    assert result.redactions == []


def test_jwt_redacted() -> None:
    jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIn0.abcDEF_xyz"
    result = redact(f"Auth header: {jwt} please")
    assert "[REDACTED:JWT]" in result.text
    assert "JWT" in result.redactions
    assert jwt not in result.text


def test_openai_key_redacted() -> None:
    text = "Use sk-abcdef0123456789ABCDEF for the OpenAI call."
    result = redact(text)
    assert "[REDACTED:OPENAI_API_KEY]" in result.text
    assert "OPENAI_API_KEY" in result.redactions


def test_anthropic_key_redacted() -> None:
    text = "Anthropic key is sk-ant-abcdef0123456789ABCDEFGHIJKLMNOPQRST."
    result = redact(text)
    assert "[REDACTED:ANTHROPIC_API_KEY]" in result.text


def test_aws_access_key_redacted() -> None:
    text = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE right here"
    result = redact(text)
    assert "[REDACTED:AWS_ACCESS_KEY]" in result.text


def test_bearer_redacted() -> None:
    text = "Authorization: Bearer abcd1234efgh5678ijkl9012mnop"
    result = redact(text)
    assert "[REDACTED:BEARER]" in result.text
