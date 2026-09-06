"""Tests del protocolo confirmatorio (respuesta a la auditoría del 2026-09-06)."""
from __future__ import annotations

import pytest
from google.adk.agents import LlmAgent

from demo.arms import ARMS, CONFIG_SHA, FROZEN, build_arm_app, config_sha
from demo.stats import (
    calibration,
    compare,
    paired_bootstrap,
    pareto_front,
    trajectory_summary,
    utility,
)
from opsworld.world import COARSE_FEATURES, DIAGNOSTIC_TOOLS, RISK
from subcortex.config import SubcortexConfig
from subcortex.gate import gate_decision, gate_decision_ex
from subcortex.interoception import render_state, render_state_neutral

CFG = SubcortexConfig(risk=RISK)


def agent() -> LlmAgent:
    return LlmAgent(name="t", model="gemini-3-flash-preview", instruction="x")


def plugins_of(arm_name: str) -> list[str]:
    app, _ = build_arm_app(ARMS[arm_name], agent(), risk=RISK, diagnostic_tools=DIAGNOSTIC_TOOLS,
                           coarse_features=COARSE_FEATURES)
    return [p.name.replace("subcortex_", "") for p in app.plugins]


def test_control_arms_isolate_the_protocol_confound():
    """El brazo `protocol` debe tener SOLO el contrato de predicción: es el control que separa
    'la arquitectura ayuda' de 'obligar a declarar consecuencia esperada ayuda'."""
    assert plugins_of("vanilla") == []
    assert plugins_of("protocol") == ["prediction"]
    assert plugins_of("retrieval") == ["prediction", "memory"]
    assert plugins_of("cache") == ["prediction", "habit"]
    assert plugins_of("gate") == ["prediction", "gate"]
    assert plugins_of("telemetry") == ["prediction", "interoception"]
    assert plugins_of("full") == ["prediction", "gate", "memory", "habit", "interoception"]


def test_telemetry_arm_uses_neutral_wording():
    """El control de telemetría dice los mismos números sin encuadre corporal ni consejo."""
    _app, sc = build_arm_app(ARMS["telemetry"], agent(), risk=RISK,
                             diagnostic_tools=DIAGNOSTIC_TOOLS, coarse_features=COARSE_FEATURES)
    assert sc.config.neutral_telemetry is True
    intero = {"steps": 6, "failures": 2, "blocks": 0, "evaluated": 0, "invalid_streak": 0}
    neutral = render_state_neutral(intero, 0.2, CFG)
    rich = render_state(intero, 0.2, CFG)
    assert "Telemetría" in neutral and "6 de 8" in neutral and "Errores consecutivos: 2" in neutral
    for palabra in ("bajá la confianza", "Tono bajo", "SIN PROGRESO", "Estado interno"):
        assert palabra not in neutral
    assert "Estado interno" in rich  # la versión propuesta sí interpreta


def test_frozen_config_hash_is_stable_and_sensitive():
    assert config_sha() == CONFIG_SHA and len(CONFIG_SHA) == 16
    original = FROZEN["gate_threshold"]
    FROZEN["gate_threshold"] = 0.5
    try:
        assert config_sha() != CONFIG_SHA  # tocar un umbral cambia el hash: deja de ser la misma corrida
    finally:
        FROZEN["gate_threshold"] = original
    assert config_sha() == CONFIG_SHA


def test_gate_decision_reports_causal_code():
    """Un veto por valor, uno hiperdirecto y uno por racha no son la misma cosa."""
    assert gate_decision_ex(CFG, "restart", 0.3, 0.6, 1.0, 0)[3] == "value"
    assert gate_decision_ex(CFG, "rollback", 0.5, 0.9, 1.0, 0)[3] == "hyperdirect"
    assert gate_decision_ex(CFG, "rollback", 0.95, 0.9, 0.3, 0)[3] == "hyperdirect"
    assert gate_decision_ex(CFG, "restart", 0.99, 0.9, 1.0, 3)[3] == "blockstreak"
    assert gate_decision_ex(CFG, "restart", 0.9, 0.6, 1.0, 0)[3] == "allowed"
    # el wrapper de 3 elementos sigue siendo compatible
    assert gate_decision(CFG, "restart", 0.9, 0.6, 1.0, 0)[:2] == gate_decision_ex(CFG, "restart", 0.9, 0.6, 1.0, 0)[:2]


def test_calibration_rewards_honest_confidence():
    """Reglas de puntuación propias: acertar con confianza alta puntúa mejor que acertar dudando,
    y errar con confianza alta es lo peor. El error categórico medio no distinguía esto."""
    seguro_ok = calibration([{"expected": "resolves", "observed": "resolves", "confidence": 0.9}] * 8)
    dudoso_ok = calibration([{"expected": "resolves", "observed": "resolves", "confidence": 0.4}] * 8)
    seguro_mal = calibration([{"expected": "resolves", "observed": "worsens", "confidence": 0.9}] * 8)
    assert seguro_ok["brier"] < dudoso_ok["brier"] < seguro_mal["brier"]
    assert seguro_ok["nll"] < seguro_mal["nll"]
    assert seguro_ok["accuracy"] == 1.0 and seguro_mal["accuracy"] == 0.0
    # sobreconfianza: 90 % declarado, 0 % de acierto → ECE alto
    assert seguro_mal["ece"] > 0.8
    assert calibration([])["n"] == 0


def test_utility_prices_compute_without_double_counting_harm():
    row = {"score": 60, "llm_calls": 5}
    assert utility(row, lam=2.0) == 50.0
    assert utility(row, lam=0.0) == 60  # λ=0 recupera el score puro


def test_trajectory_summary_uses_second_half_and_splits_vetoes():
    rows = [{"score": 10, "llm_calls": 6, "resolved": False, "worsens": 1, "steps": 3, "tokens": 100,
             "seconds": 5, "vetoes": 1, "vetoes_value": 1, "arbitration_dropped": 2,
             "pred_log": [{"expected": "resolves", "observed": "worsens", "confidence": 0.8}]},
            {"score": 90, "llm_calls": 4, "resolved": True, "worsens": 0, "steps": 2, "tokens": 90,
             "seconds": 4, "vetoes": 0, "vetoes_hyperdirect": 0, "arbitration_dropped": 0,
             "pred_log": [{"expected": "resolves", "observed": "resolves", "confidence": 0.8}]}]
    t = trajectory_summary(rows)
    assert t["n_episodes"] == 2 and t["harm_total"] == 1
    assert t["utility_mean"] == pytest.approx((10 - 12 + 90 - 8) / 2)
    assert t["utility_mean_second_half"] == pytest.approx(90 - 8)  # solo el segundo episodio
    assert t["vetoes_total"] == 1 and t["vetoes_value_total"] == 1
    assert t["arbitration_dropped_total"] == 2  # no se mezcla con los vetos
    assert t["calibration"]["n"] == 2 and t["calibration"]["accuracy"] == 0.5


def test_paired_bootstrap_reports_uncertainty_and_sign_consistency():
    consistente = paired_bootstrap([5.0, 6.0, 4.0, 5.5, 5.2])
    assert consistente["favorable"] == 5 and consistente["ci95"][0] > 0  # IC no cruza cero
    mezclado = paired_bootstrap([5.0, -4.0, 6.0, -5.0, 1.0])
    assert mezclado["favorable"] == 3 and mezclado["ci95"][0] < 0 < mezclado["ci95"][1]
    assert paired_bootstrap([])["n"] == 0


def test_compare_is_paired_only_over_shared_seeds():
    traj = {"vanilla": {1: {"utility_mean": 10.0}, 2: {"utility_mean": 20.0}, 3: {"utility_mean": 30.0}},
            "full": {1: {"utility_mean": 15.0}, 2: {"utility_mean": 22.0}}}  # sin la semilla 3
    res = compare(traj, "vanilla", "utility_mean")["full"]
    assert res["seeds"] == [1, 2] and res["n_pairs"] == 2
    assert res["mean_diff"] == pytest.approx(3.5) and res["favorable"] == 2


def test_pareto_front_keeps_undominated_arms():
    # (utilidad, daño): full domina a vanilla; gate sacrifica utilidad por menos daño → ambos quedan
    puntos = {"vanilla": (10.0, 8.0), "full": (20.0, 4.0), "gate": (12.0, 1.0)}
    assert pareto_front(puntos) == ["full", "gate"]
