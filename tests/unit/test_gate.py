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
    assert m["llm_calls"] == 1 and m["tokens"] == 123 and m["vetoes"] == 1
