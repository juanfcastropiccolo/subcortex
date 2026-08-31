import pytest
from subcortex.types import Scene, prediction_error


def test_scene_key_is_stable_and_order_independent():
    a = Scene.from_features({"service": "api", "symptom": "oom"})
    b = Scene.from_features({"symptom": "oom", "service": "api"})
    assert a.key == b.key
    assert len(a.key) == 12


def test_prediction_error_signed_and_scaled_by_confidence():
    assert prediction_error("resolves", "resolves", 0.9) == 0.0
    assert prediction_error("resolves", "worsens", 1.0) == -1.0      # -3/2 clamp
    assert prediction_error("no_change", "improves", 1.0) == 0.5     # +1/2
    assert prediction_error("no_change", "improves", 0.5) == 0.25
    assert prediction_error("diagnostic", "no_change", 1.0) == 0.0
    assert prediction_error("resolves", "improves", 0.8) == pytest.approx(-0.4)
