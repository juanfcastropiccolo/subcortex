import inspect
from types import SimpleNamespace

import pytest

from subcortex.config import SubcortexConfig
from subcortex.prediction import PredictionPlugin, wrap_action_tool
from subcortex.types import K_LAST_ERROR, K_PENDING

CFG = SubcortexConfig(risk={"restart": "costly"}, diagnostic_tools=frozenset({"inspect_service"}))


def restart(service: str, tool_context) -> dict:
    """Reinicia el servicio."""
    return {"status": "success", "observed_effect": "resolves", "svc": service}


def test_wrap_extends_signature_and_strips_params():
    w = wrap_action_tool(restart)
    params = list(inspect.signature(w).parameters)
    assert params == ["service", "tool_context", "expected_effect", "confidence"]
    assert w.__name__ == "restart" and "expected_effect" in w.__doc__
    assert w(service="api", tool_context=None, expected_effect="resolves", confidence=0.9)["svc"] == "api"


class Tool:
    def __init__(self, name): self.name = name


def ctx(call_id="c1"):
    return SimpleNamespace(state={}, function_call_id=call_id)


@pytest.mark.asyncio
async def test_before_tool_rejects_missing_or_invalid_prediction():
    p = PredictionPlugin(CFG)
    r = await p.before_tool_callback(tool=Tool("restart"), tool_args={"service": "api"}, tool_context=ctx())
    assert r["status"] == "rejected"
    r = await p.before_tool_callback(tool=Tool("restart"), tool_context=ctx(),
                                     tool_args={"service": "api", "expected_effect": "magic", "confidence": 0.5})
    assert r["status"] == "rejected"
    assert await p.before_tool_callback(tool=Tool("inspect_service"), tool_args={}, tool_context=ctx()) is None


@pytest.mark.asyncio
async def test_after_tool_computes_error_and_skips_blocked():
    p = PredictionPlugin(CFG)
    c = ctx()
    args = {"service": "api", "expected_effect": "resolves", "confidence": 1.0}
    assert await p.before_tool_callback(tool=Tool("restart"), tool_args=args, tool_context=c) is None
    assert "c1" in c.state[K_PENDING]
    await p.after_tool_callback(tool=Tool("restart"), tool_args=args, tool_context=c,
                                result={"status": "success", "observed_effect": "worsens"})
    le = c.state[K_LAST_ERROR]
    assert le["error"] == -1.0 and le["observed"] == "worsens" and "c1" not in c.state[K_PENDING]
    c2 = ctx("c2")
    await p.before_tool_callback(tool=Tool("restart"), tool_args=args, tool_context=c2)
    await p.after_tool_callback(tool=Tool("restart"), tool_args=args, tool_context=c2,
                                result={"status": "vetoed"})
    assert c2.state.get(K_LAST_ERROR) is None


@pytest.mark.asyncio
async def test_invalid_call_is_not_an_outcome():
    """Una llamada inválida (args mal formados) no toca el mundo: no genera error de predicción."""
    p = PredictionPlugin(CFG)
    c = ctx("c9")
    args = {"service": "api", "expected_effect": "resolves", "confidence": 0.9}
    await p.before_tool_callback(tool=Tool("restart"), tool_args=args, tool_context=c)
    await p.after_tool_callback(tool=Tool("restart"), tool_args=args, tool_context=c,
                                result={"status": "invalid", "observed_effect": "no_change"})
    assert c.state.get(K_LAST_ERROR) is None
    assert c.state.get("subcortex.metrics", {}).get("error_count", 0) == 0


@pytest.mark.asyncio
async def test_after_model_counts_cost_in_every_configuration():
    """El costo lo cuenta prediction, no el gate.

    Bug encontrado durante el experimento confirmatorio (2026-09-06): llamadas y tokens se
    contaban en `GatePlugin.after_model`, así que los brazos de control sin gate reportaban
    cero llamadas — y el endpoint primario es `score − λ·llamadas`. Su utilidad habría salido
    inflada justo en las comparaciones que la auditoría pidió.
    """
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    p = PredictionPlugin(CFG)
    c = ctx("cost")
    resp = LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text="hola")]),
        usage_metadata=types.GenerateContentResponseUsageMetadata(total_token_count=321))
    assert await p.after_model_callback(callback_context=c, llm_response=resp) is None
    await p.after_model_callback(callback_context=c, llm_response=resp)
    m = c.state["subcortex.metrics"]
    assert m["llm_calls"] == 2 and m["tokens"] == 642
    # una respuesta parcial (streaming) no cuenta como llamada
    resp.partial = True
    await p.after_model_callback(callback_context=c, llm_response=resp)
    assert c.state["subcortex.metrics"]["llm_calls"] == 2
