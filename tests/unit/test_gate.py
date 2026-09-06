from types import SimpleNamespace

import pytest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from subcortex.config import SubcortexConfig
from subcortex.gate import GatePlugin, gate_decision
from subcortex.store import EpisodicStore
from subcortex.types import K_DISCOVERED, K_FEATURES, K_INTERO, K_PENDING, K_TONE, K_VETO_LOG, Scene

CFG = SubcortexConfig(risk={"restart": "costly", "rollback": "irreversible", "resolve": "free",
                            "escalate_to_human": "free", "failover_db": "irreversible"},
                      diagnostic_tools=frozenset({"inspect_service"}),
                      coarse_features=("symptom", "finding"))


def test_decision_matrix():
    ok, v, _ = gate_decision(CFG, "restart", 0.9, 0.6, 1.0, 0)
    assert ok and v == pytest.approx(0.44)                                 # 0.54-0.10
    ok, _, why = gate_decision(CFG, "restart", 0.3, 0.6, 1.0, 0)          # 0.18-0.10 < 0.1
    assert not ok and "valor" in why
    ok, _, why = gate_decision(CFG, "rollback", 0.5, 0.9, 1.0, 0)         # hiperdirecta por confianza
    assert not ok and "irreversible" in why and "confianza" in why
    ok, _, why = gate_decision(CFG, "rollback", 0.95, 0.9, 0.3, 0)        # hiperdirecta por tono
    assert not ok
    ok, _, _ = gate_decision(CFG, "rollback", 0.95, 0.9, 0.9, 0)
    assert ok
    ok, _, why = gate_decision(CFG, "restart", 0.99, 0.9, 1.0, 3)         # bloqueo tras 3
    assert not ok and "escal" in why
    assert gate_decision(CFG, "escalate_to_human", 0.1, 0.1, 0.05, 9)[0]


def test_trust_earned_skips_hyperdirect_but_not_value():
    ok, _, _ = gate_decision(CFG, "rollback", 0.5, 0.9, 1.0, 0, trusted=True)   # 0.45-0.20 ≥ 0.10
    assert ok
    ok, _, why = gate_decision(CFG, "rollback", 0.3, 0.9, 1.0, 0, trusted=True)  # 0.27-0.20 < 0.1
    assert not ok and "valor" in why
    # dopamina baja por fracasos previos: bloquea aunque haya confianza
    ok, _, why = gate_decision(CFG, "rollback", 0.9, 0.3, 1.0, 0, trusted=False)  # 0.27-0.20 < 0.1
    assert not ok and "valor" in why


class Tool:
    def __init__(self, name): self.name = name


FEATS = {"service": "api", "symptom": "high_latency"}


def ctx(pending, tone=1.0, blocks=0, discovered=None):
    return SimpleNamespace(function_call_id="c1", state={
        K_PENDING: {"c1": pending}, K_TONE: tone, K_INTERO: {"blocks": blocks},
        K_FEATURES: FEATS, K_DISCOVERED: discovered or {}})


@pytest.mark.asyncio
async def test_before_tool_vetoes_logs_and_hints_alternative():
    store = EpisodicStore(":memory:")
    scene = Scene.from_features({**FEATS, "finding": "db_pool_exhausted"}, coarse=CFG.coarse_features)
    store.record_outcome(scene.coarse_key, "restart", True)
    g = GatePlugin(CFG, store)
    c = ctx({"tool": "failover_db", "args": {}, "expected": "resolves", "confidence": 0.5},
            discovered={"finding": "db_pool_exhausted"})
    r = await g.before_tool_callback(tool=Tool("failover_db"), tool_args={}, tool_context=c)
    assert r["status"] == "vetoed"
    assert "restart (1 éxitos, 0 fallos)" in r["hint"] and "confidence ≥ 0.6" in r["hint"]
    assert c.state["subcortex.metrics"]["vetoes"] == 1
    assert c.state["subcortex.metrics"]["vetoes_irreversible"] == 1
    assert c.state[K_VETO_LOG][0]["tool"] == "failover_db" and c.state[K_VETO_LOG][0]["confidence"] == 0.5
    c = ctx({"tool": "restart", "args": {}, "expected": "resolves", "confidence": 0.9})
    assert await g.before_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=c) is None
    assert await g.before_tool_callback(tool=Tool("inspect_service"), tool_args={}, tool_context=c) is None


@pytest.mark.asyncio
async def test_trusted_action_passes_hyperdirect_on_coarse_scene():
    store = EpisodicStore(":memory:")
    scene = Scene.from_features({**FEATS, "finding": "db_pool_exhausted"}, coarse=CFG.coarse_features)
    store.record_outcome(scene.coarse_key, "failover_db", True)
    store.record_outcome(scene.coarse_key, "failover_db", True)
    g = GatePlugin(CFG, store)
    # misma clase de escena aunque cambie el servicio
    c = ctx({"tool": "failover_db", "args": {}, "expected": "resolves", "confidence": 0.55},
            discovered={"finding": "db_pool_exhausted"})
    c.state[K_FEATURES] = {"service": "checkout", "symptom": "high_latency"}
    assert await g.before_tool_callback(tool=Tool("failover_db"), tool_args={}, tool_context=c) is None


def fc(name, conf):
    return types.Part(function_call=types.FunctionCall(
        name=name, args={"service": "api", "expected_effect": "resolves", "confidence": conf}))


@pytest.mark.asyncio
async def test_after_model_winner_take_all_and_llm_count():
    g = GatePlugin(CFG, EpisodicStore(":memory:"))
    c = SimpleNamespace(state={K_TONE: 1.0, K_FEATURES: FEATS})
    resp = LlmResponse(content=types.Content(role="model", parts=[
        fc("restart", 0.5), fc("restart", 0.9),
        types.Part(function_call=types.FunctionCall(name="inspect_service", args={}))]),
        usage_metadata=types.GenerateContentResponseUsageMetadata(total_token_count=123))
    out = await g.after_model_callback(callback_context=c, llm_response=resp)
    calls = out.get_function_calls()
    assert [x.name for x in calls] == ["restart", "inspect_service"]
    assert calls[0].args["confidence"] == 0.9
    m = c.state["subcortex.metrics"]
    # El descarte por winner-take-all es arbitraje, no veto: no impide ninguna acción que el
    # agente fuera a ejecutar igual. Contarlo como veto inflaba el "efecto de seguridad"
    # (auditoría 2026-09-06), así que va en su propia métrica.
    assert m["arbitration_dropped"] == 1 and m["vetoes"] == 0
    # El costo (llamadas, tokens) ya NO lo cuenta el gate: lo cuenta prediction, que está en
    # todas las configuraciones. Ver test_prediction.py::test_after_model_counts_cost.
    assert m.get("llm_calls", 0) == 0


def test_effective_confidence_shifts_from_model_to_history():
    from subcortex.gate import effective_confidence
    assert effective_confidence(0.9, 0.2, 0) == 0.9            # sin historial manda el modelo
    assert effective_confidence(0.9, 0.2, 2) == pytest.approx(0.55)   # a la par
    assert effective_confidence(0.9, 0.2, 6) == pytest.approx(0.375)  # historial 3 a 1
    assert effective_confidence(0.3, 0.9, 6) == pytest.approx(0.75)   # y también sube la del tímido


@pytest.mark.asyncio
async def test_gate_uses_history_confidence_when_enabled():
    store = EpisodicStore(":memory:")
    scene = Scene.from_features({**FEATS, "finding": "db_pool_exhausted"}, coarse=CFG.coarse_features)
    for _ in range(6):  # seis fracasos de restart en esta clase de escena
        store.record_outcome(scene.coarse_key, "restart", False)
    cfg_hist = SubcortexConfig(**{**CFG.__dict__, "confidence_from_history": True})
    g = GatePlugin(cfg_hist, store)
    c = ctx({"tool": "restart", "args": {}, "expected": "resolves", "confidence": 0.95},
            discovered={"finding": "db_pool_exhausted"})
    r = await g.before_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=c)
    assert r is not None and r["status"] == "vetoed"      # el modelo dice 0.95, el historial dice 0.15
    assert c.state[K_VETO_LOG][0]["declared"] == 0.95 and c.state[K_VETO_LOG][0]["confidence"] < 0.5
    cfg2 = CFG  # por defecto la confianza es la declarada por el modelo
    c2 = ctx({"tool": "restart", "args": {}, "expected": "resolves", "confidence": 0.95},
             discovered={"finding": "db_pool_exhausted"})
    # sin historial en la confianza, value = 0.95 × dopamina(0.15) − 0.10 = 0.04 < 0.1: igual veta,
    # pero por valor; la diferencia se ve en el registro
    r2 = await GatePlugin(cfg2, store).before_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=c2)
    assert r2["status"] == "vetoed" and c2.state[K_VETO_LOG][0]["confidence"] == 0.95


@pytest.mark.asyncio
async def test_cingulate_reconsider_once_per_action_when_value_is_near_threshold():
    from subcortex.gate import K_RECONSIDERED
    cfg = SubcortexConfig(**{**CFG.__dict__, "cingulate_reconsider": True})
    g = GatePlugin(cfg, EpisodicStore(":memory:"))
    # restart: value = 0.3 × 0.6 − 0.10 = 0.08 → autorizada por umbral? no: 0.08 < 0.10 → veto.
    # Con confianza 0.4: 0.24 − 0.10 = 0.14 → autorizada y a 0.04 del umbral → conflicto.
    c = ctx({"tool": "restart", "args": {}, "expected": "resolves", "confidence": 0.4})
    r = await g.before_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=c)
    assert r["status"] == "reconsider" and "conflicto" in r["reason"]
    assert c.state[K_RECONSIDERED] == ["restart"] and c.state["subcortex.metrics"]["reconsiders"] == 1
    assert c.state["subcortex.metrics"].get("vetoes", 0) == 0
    # segunda vez, misma acción: pasa
    assert await g.before_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=c) is None
    # lejos del umbral no hay conflicto
    c2 = ctx({"tool": "restart", "args": {}, "expected": "resolves", "confidence": 0.95})
    assert await g.before_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=c2) is None
