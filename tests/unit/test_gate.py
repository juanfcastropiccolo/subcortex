from types import SimpleNamespace

import pytest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from subcortex.config import SubcortexConfig
from subcortex.gate import GatePlugin, gate_decision
from subcortex.store import EpisodicStore
from subcortex.types import K_FEATURES, K_INTERO, K_PENDING, K_TONE

CFG = SubcortexConfig(risk={"restart": "costly", "rollback": "irreversible", "resolve": "free",
                            "escalate_to_human": "free"},
                      diagnostic_tools=frozenset({"inspect_service"}))


def test_decision_matrix():
    ok, v, _ = gate_decision(CFG, "restart", 0.9, 0.6, 1.0, 0)
    assert ok and v == pytest.approx(0.39)
    ok, _, why = gate_decision(CFG, "restart", 0.3, 0.6, 1.0, 0)          # 0.18-0.15 < 0.1
    assert not ok and "valor" in why
    ok, _, why = gate_decision(CFG, "rollback", 0.6, 0.9, 1.0, 0)         # hiperdirecta por confianza
    assert not ok and "irreversible" in why
    ok, _, why = gate_decision(CFG, "rollback", 0.95, 0.9, 0.3, 0)        # hiperdirecta por tono
    assert not ok
    ok, _, _ = gate_decision(CFG, "rollback", 0.95, 0.9, 0.9, 0)
    assert ok
    ok, _, why = gate_decision(CFG, "restart", 0.99, 0.9, 1.0, 3)         # bloqueo tras 3
    assert not ok and "escal" in why
    assert gate_decision(CFG, "escalate_to_human", 0.1, 0.1, 0.05, 9)[0]


class Tool:
    def __init__(self, name): self.name = name


def ctx(pending, tone=1.0, blocks=0):
    return SimpleNamespace(function_call_id="c1", state={
        K_PENDING: {"c1": pending}, K_TONE: tone, K_INTERO: {"blocks": blocks},
        K_FEATURES: {"service": "api"}})


@pytest.mark.asyncio
async def test_before_tool_vetoes_and_counts():
    g = GatePlugin(CFG, EpisodicStore(":memory:"))
    c = ctx({"tool": "rollback", "args": {}, "expected": "resolves", "confidence": 0.5})
    r = await g.before_tool_callback(tool=Tool("rollback"), tool_args={}, tool_context=c)
    assert r["status"] == "vetoed"
    assert c.state["subcortex.metrics"]["vetoes"] == 1 and c.state["subcortex.metrics"]["vetoes_irreversible"] == 1
    c = ctx({"tool": "restart", "args": {}, "expected": "resolves", "confidence": 0.9})
    assert await g.before_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=c) is None
    assert await g.before_tool_callback(tool=Tool("inspect_service"), tool_args={}, tool_context=c) is None


def fc(name, conf):
    return types.Part(function_call=types.FunctionCall(
        name=name, args={"service": "api", "expected_effect": "resolves", "confidence": conf}))


@pytest.mark.asyncio
async def test_after_model_winner_take_all_and_llm_count():
    g = GatePlugin(CFG, EpisodicStore(":memory:"))
    c = SimpleNamespace(state={K_TONE: 1.0, K_FEATURES: {"service": "api"}})
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
