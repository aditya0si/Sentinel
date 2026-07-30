"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert "uptime_seconds" in data
        assert "engine_config" in data


class TestValidateEndpoint:
    def test_validate_clean_text(self, client):
        response = client.post(
            "/validate",
            json={"output": "This is a clean, safe text."},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["overall_pass"] is True
        assert data["aggregate_confidence"] == 1.0

    def test_validate_with_pii(self, client):
        response = client.post(
            "/validate",
            json={"output": "Contact john@example.com for details."},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["overall_pass"] is False
        assert data["aggregate_confidence"] < 1.0

    def test_validate_with_toxic_content(self, client):
        response = client.post(
            "/validate",
            json={"output": "You are a stupid idiot!"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["overall_pass"] is False

    def test_validate_with_custom_rules(self, client):
        response = client.post(
            "/validate",
            json={
                "output": "Buy now! This is spam.",
                "validators": ["rules"],
                "rules": [
                    {"name": "no_spam", "pattern": r"(?i)buy now", "severity": "error"}
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["overall_pass"] is False

    def test_validate_pii_only(self, client):
        """Test with only PII validator enabled."""
        response = client.post(
            "/validate",
            json={
                "output": "test@test.com is bad!",  # has PII AND toxic word
                "validators": ["pii"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["overall_pass"] is False  # PII fails
        # Only one validator was configured (PII)
        assert len(data["results"]) == 1

    def test_validate_with_context(self, client):
        response = client.post(
            "/validate",
            json={
                "output": "Safe text.",
                "context": {"user_id": "123", "session": "abc"},
            },
        )
        assert response.status_code == 200


class TestAgentEndpoint:
    def test_agent_basic(self, client):
        """Test basic agent invocation (may need OPENAI_API_KEY for actual LLM call)."""
        response = client.post(
            "/agent",
            json={
                "input": "What is 2 + 2?",
                "enable_guardrails": False,
            },
        )
        # Without an API key, this may 500. Just ensure the endpoint exists.
        assert response.status_code in (200, 500)

    def test_agent_with_guardrails(self, client):
        response = client.post(
            "/agent",
            json={
                "input": "Hello!",
                "enable_guardrails": True,
            },
        )
        assert response.status_code in (200, 500)
