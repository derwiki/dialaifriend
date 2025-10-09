import os
import json
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def app_module(monkeypatch):
    # Ensure the app import does not fail due to missing OPENAI_API_KEY
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # Import lazily after env var is set, and reload for determinism
    import importlib
    import main as main_mod
    importlib.reload(main_mod)
    return main_mod


def test_fastapi_index_endpoint_returns_ok_json(app_module):
    client = TestClient(app_module.app)
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["message"].startswith("Twilio Media Stream Server is running")


def test_incoming_call_returns_twiML_with_stream_url_contains_voice(app_module):
    client = TestClient(app_module.app)
    # Twilio calls this as POST typically
    response = client.post("/incoming-call")
    assert response.status_code == 200
    # Response is XML TwiML; just verify expected structure pieces
    xml_text = response.text
    assert "<Connect>" in xml_text
    assert "<Stream url=\"wss://testserver/media-stream?voice=" in xml_text


def test_create_system_message_includes_personality_and_rules(app_module):
    # Pick a specific voice for determinism
    message = app_module.create_system_message("alloy")
    # Core personality prompt parts
    assert "When you first connect, wait 2 seconds" in message
    assert "DEVELOPER MODE CONTROL:" in message
    assert "Activation pass phrase" in message
    assert "In Developer Mode" in message
    assert "ignore toddler conversation rules" in message
