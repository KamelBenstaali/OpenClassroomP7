import importlib
import json
import sys

import pytest
import requests
from streamlit.testing.v1 import AppTest


@pytest.fixture
def streamlit_module(monkeypatch):
    """Reload streamlit_app in test mode to bypass UI rendering."""
    monkeypatch.setenv("APP_TEST_MODE", "1")
    sys.modules.pop("app.FrontEnd.streamlit_app", None)
    import app.FrontEnd.streamlit_app as sa

    importlib.reload(sa)
    return sa


def test_to_data_uri_returns_base64(streamlit_module, tmp_path):
    """Encode a PNG into a data URI and ensure base64 is present."""
    img = tmp_path / "fake.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal header-like bytes

    uri = streamlit_module.to_data_uri(img)

    assert uri.startswith("data:image/png;base64,")
    assert len(uri.split(",")[1]) > 0


def test_call_api_success(monkeypatch, streamlit_module):
    """Simulate a successful API call and return decoded JSON."""
    payload = {"text": "hello"}

    class DummyResponse:
        def __init__(self, content):
            self._content = content

        def raise_for_status(self):
            return None

        def json(self):
            return self._content

    def fake_post(url, json, timeout):
        assert url == streamlit_module.API_URL
        assert json == payload
        return DummyResponse({"ok": True})

    monkeypatch.setattr(streamlit_module.requests, "post", fake_post)

    data, err = streamlit_module.call_api(payload)
    assert data == {"ok": True}
    assert err is None


def test_call_api_error(monkeypatch, streamlit_module):
    """Simulate an API error and ensure the error message is propagated."""
    class DummyResponse:
        def raise_for_status(self):
            raise requests.HTTPError("boom")

    monkeypatch.setattr(streamlit_module.requests, "post", lambda *args, **kwargs: DummyResponse())

    data, err = streamlit_module.call_api({"text": "hi"})
    assert data is None
    assert "boom" in err


def test_persist_history_appends_json(monkeypatch, streamlit_module, tmp_path):
    """Ensure history persistence appends one JSON line per entry."""
    history_path = tmp_path / "conversation_history.txt"
    monkeypatch.setattr(streamlit_module, "HISTORY_FILE", history_path)

    entry = {"text": "Hello", "label": "positive", "score": 0.99}
    streamlit_module.persist_history(entry)

    saved = history_path.read_text(encoding="utf-8").strip().splitlines()
    assert saved, "History file should contain at least one entry"
    parsed = json.loads(saved[0])
    assert parsed["text"] == "Hello"
    assert parsed["label"] == "positive"


def _find_button(app_test: AppTest, label: str):
    for btn in app_test.button:
        if getattr(btn, "label", None) == label:
            return btn
    raise AssertionError(f"Button with label '{label}' not found")


def test_analyze_shows_error_on_api_failure(monkeypatch, tmp_path):
    """User flow: submit text -> API error -> error message and no last_pred."""
    monkeypatch.delenv("APP_TEST_MODE", raising=False)
    monkeypatch.setenv("PREDICT_API_URL", "http://dummy/predict")
    monkeypatch.setenv("FEEDBACK_API_URL", "http://dummy/feedback")
    monkeypatch.setenv("HISTORY_FILE_PATH", str(tmp_path / "history.txt"))

    def failing_post(url, json=None, timeout=10):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(requests, "post", failing_post)

    at = AppTest.from_file("app/FrontEnd/streamlit_app.py").run()
    at.text_area[0].input("hello error").run()
    _find_button(at, "Analyser").click().run()

    assert "last_pred" in at.session_state
    assert at.session_state["last_pred"] is None
    assert any("Erreur d'appel API" in getattr(err, "value", "") for err in at.error)


def test_history_toggle_and_clear(monkeypatch, tmp_path):
    """User flow: analyze text, show history, then clear it."""
    monkeypatch.delenv("APP_TEST_MODE", raising=False)
    monkeypatch.setenv("PREDICT_API_URL", "http://dummy/predict")
    monkeypatch.setenv("FEEDBACK_API_URL", "http://dummy/feedback")
    monkeypatch.setenv("HISTORY_FILE_PATH", str(tmp_path / "history.txt"))

    def fake_post(url, json=None, timeout=10):
        class DummyResponse:
            def __init__(self, payload):
                self.payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        if url.endswith("/predict"):
            return DummyResponse({"label": "positive", "score": 0.73})
        return DummyResponse({"status": "ok"})

    monkeypatch.setattr(requests, "post", fake_post)

    at = AppTest.from_file("app/FrontEnd/streamlit_app.py").run()
    at.text_area[0].input("Great experience").run()
    _find_button(at, "Analyser").click().run()

    assert at.session_state["history"], "History should have one entry after prediction"
    toggle_btn = _find_button(at, "Afficher l'historique")
    toggle_btn.click().run()
    assert at.session_state["show_history"] is True

    clear_btn = _find_button(at, "Vider l'historique")
    clear_btn.click().run()
    assert at.session_state["history"] == []
    history_path = tmp_path / "history.txt"
    assert not history_path.exists() or history_path.read_text() == ""


def test_feedback_submission_succeeds(monkeypatch, tmp_path):
    """User flow: predict, submit feedback, and see success with payload sent."""
    monkeypatch.delenv("APP_TEST_MODE", raising=False)
    monkeypatch.setenv("PREDICT_API_URL", "http://dummy/predict")
    monkeypatch.setenv("FEEDBACK_API_URL", "http://dummy/feedback")
    monkeypatch.setenv("HISTORY_FILE_PATH", str(tmp_path / "history.txt"))

    feedback_calls = []

    def fake_post(url, json=None, timeout=10):
        class DummyResponse:
            def __init__(self, payload):
                self.payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        if url.endswith("/predict"):
            return DummyResponse({"label": "positive", "score": 0.88})
        feedback_calls.append(json)
        return DummyResponse({"status": "ok"})

    monkeypatch.setattr(requests, "post", fake_post)

    at = AppTest.from_file("app/FrontEnd/streamlit_app.py").run()
    at.text_area[0].input("Loved it").run()
    _find_button(at, "Analyser").click().run()

    feedback_selector = next(
        sb for sb in at.selectbox if "(inconnu)" in getattr(sb, "options", [])
    )
    feedback_selector.select("positive").run()
    at.text_area[1].input("great flight").run()
    _find_button(at, "Signaler une mauvaise prédiction").click().run()

    assert "feedback_status" in at.session_state
    assert at.session_state["feedback_status"] == "sent"
    assert feedback_calls, "Feedback API should be called"
    payload = feedback_calls[0]
    assert payload["text"] == "Loved it"
    assert payload["expected_label"] == "positive"
