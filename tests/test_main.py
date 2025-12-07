import importlib
import os
import sys

import pytest


@pytest.fixture(scope="session")
def main_module():
    """Reload main.py in test mode with the model loading."""
    os.environ.setdefault("APP_TEST_MODE", "1")
    os.environ.setdefault("APPLICATIONINSIGHTS_CONNECTION_STRING", "test")

    sys.modules.pop("app.FastApi.main", None)
    import app.FastApi.main as main

    importlib.reload(main)
    return main


def test_preprocess_normalizes_spaces(main_module):
    """Ensure preprocess trims and collapses whitespace."""
    assert main_module.preprocess("  Hello   world \n") == "Hello world"


def test_classifier_loads_and_returns_output(main_module):
    """Check the real classifier returns a list with label and score."""
    output = main_module.clf("Amazing experience")
    assert isinstance(output, list)
    assert output, "Classifier should return at least one result"
    first = output[0]
    assert "label" in first and "score" in first


def test_predict_maps_label_and_cleans_text(main_module, monkeypatch):
    """Verify predict cleans text and returns a mapped label with score."""
    item = main_module.Item(text="  bad   movie  ")
    result = main_module.predict(item)

    assert result["label"] in {"positive", "negative"}
    assert 0 <= result["score"] <= 1


def test_send_feedback_logs_custom_dimensions(main_module, monkeypatch):
    """Confirm feedback logging records text, label, and metadata."""
    calls = []

    class DummyLogger:
        def warning(self, message, extra=None):
            calls.append({"message": message, "extra": extra})

    monkeypatch.setattr(main_module, "logger", DummyLogger())

    feedback = main_module.Feedback(
        text="Great flight ",
        predicted_label="positive",
        score=0.9,
        expected_label="positive",
        comment="matched expectation",
    )
    response = main_module.send_feedback(feedback)

    assert response == {"status": "ok"}
    assert calls, "Logger should be called"
    dims = calls[0]["extra"]["custom_dimensions"]
    assert dims["text"] == "Great flight"
    assert dims["predicted_label"] == "positive"
