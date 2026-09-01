import pytest

from subcortex.types import Scene, prediction_error


def test_scene_key_is_stable_and_order_independent():
    a = Scene.from_features({"service": "api", "symptom": "oom"})
    b = Scene.from_features({"symptom": "oom", "service": "api"})
    assert a.key == b.key
    assert len(a.key) == 12
    assert a.coarse_key == a.key  # sin coarse, la clase de escena es la escena exacta


def test_scene_coarse_key_ignores_other_features():
    a = Scene.from_features({"service": "api", "symptom": "oom", "hour": "night"}, coarse=("symptom",))
    b = Scene.from_features({"service": "auth", "symptom": "oom", "hour": "day"}, coarse=("symptom",))
    assert a.key != b.key and a.coarse_key == b.coarse_key


def test_success_includes_expected_no_change():
    from subcortex.types import is_success
    assert is_success("resolves", "resolves") and is_success("no_change", "improves")
    assert is_success("no_change", "no_change")          # mantener y que no pase nada: acierto
    assert not is_success("resolves", "no_change")       # esperaba resolver y no pasó nada: fallo
    assert not is_success("no_change", "worsens")


def test_prediction_error_signed_and_scaled_by_confidence():
    assert prediction_error("resolves", "resolves", 0.9) == 0.0
    assert prediction_error("resolves", "worsens", 1.0) == -1.0      # -3/2 clamp
    assert prediction_error("no_change", "improves", 1.0) == 0.5     # +1/2
    assert prediction_error("no_change", "improves", 0.5) == 0.25
    assert prediction_error("diagnostic", "no_change", 1.0) == 0.0
    assert prediction_error("resolves", "improves", 0.8) == pytest.approx(-0.4)
