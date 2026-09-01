from types import SimpleNamespace

import pytest

from subcortex.config import SubcortexConfig
from subcortex.habit import HabitPlugin, resolve_args, templatize_args
from subcortex.store import EpisodicStore
from subcortex.types import (
    K_ACTED,
    K_DISCOVERED,
    K_FEATURES,
    K_HABIT_HIT,
    K_LAST_ERROR,
    Habit,
    Scene,
)

CFG = SubcortexConfig(risk={"restart": "costly", "rollback": "irreversible"},
                      coarse_features=("symptom", "finding"))
FEATS = {"service": "api", "symptom": "oom"}
DISC = {"finding": "memory_high"}
KEY = Scene.from_features({**FEATS, **DISC}, coarse=CFG.coarse_features).coarse_key


class Tool:
    def __init__(self, name): self.name = name


def ctx(last_error=None, acted=False, habit_hit=False, feats=None):
    return SimpleNamespace(function_call_id="c1", state={
        K_FEATURES: feats or FEATS, K_DISCOVERED: DISC, K_LAST_ERROR: last_error,
        K_ACTED: acted, K_HABIT_HIT: habit_hit})


def le(tool, err, observed, service="api"):
    return {"tool": tool, "args": {"service": service}, "expected": "resolves", "observed": observed,
            "confidence": 0.9, "error": err, "status": "success", "call_id": "c1"}


def test_rewrite_habit_history_replaces_only_synthetic_pairs():
    from google.genai import types

    from subcortex.habit import HABIT_MARK, rewrite_habit_history
    real = types.Part(function_call=types.FunctionCall(id="r1", name="inspect_service", args={"service": "api"}),
                      thought_signature=b"real-sig")
    synth = types.Part(function_call=types.FunctionCall(id="h1", name="restart", args={"service": "api"}),
                       thought_signature=HABIT_MARK)
    contents = [
        types.Content(role="user", parts=[types.Part(text="incidente")]),
        types.Content(role="model", parts=[real]),
        types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
            id="r1", name="inspect_service", response={"finding": "memory_high"}))]),
        types.Content(role="model", parts=[synth]),
        types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
            id="h1", name="restart", response={"observed_effect": "resolves"}))]),
    ]
    assert rewrite_habit_history(contents) == 1
    assert contents[1].parts[0].function_call.name == "inspect_service"      # la real queda
    assert contents[2].parts[0].function_response is not None
    assert contents[3].parts[0].function_call is None
    assert "[hábito] Ejecuté restart(service=api)" in contents[3].parts[0].text
    assert contents[4].parts[0].function_response is None
    assert "resolves" in contents[4].parts[0].text


def test_args_template_roundtrip():
    t = templatize_args({"service": "api", "replicas": 4}, {"service": "api", "symptom": "oom"})
    assert t == {"service": "$service", "replicas": 4}
    assert resolve_args(t, {"service": "checkout"}) == {"service": "checkout", "replicas": 4}


@pytest.mark.asyncio
async def test_compiles_across_services_never_irreversible():
    store = EpisodicStore(":memory:")
    h = HabitPlugin(CFG, store)
    for svc in ("api", "auth", "search"):
        store.record_outcome(KEY, "restart", True)
        store.record_outcome(KEY, "rollback", True)
        c = ctx(le("restart", 0.0, "resolves", svc), feats={"service": svc, "symptom": "oom"})
        await h.after_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=c,
                                    result={"status": "success"})
        c = ctx(le("rollback", 0.0, "resolves", svc), feats={"service": svc, "symptom": "oom"})
        await h.after_tool_callback(tool=Tool("rollback"), tool_args={}, tool_context=c,
                                    result={"status": "success"})
    hb = store.get_habit(KEY)
    assert hb and hb.tool == "restart" and hb.strength >= 0.8 and hb.typical_effect == "resolves"
    assert hb.args == {"service": "$service"}


@pytest.mark.asyncio
async def test_bypass_resolves_args_for_current_scene_once_per_episode():
    store = EpisodicStore(":memory:")
    store.upsert_habit(Habit(scene_key=KEY, tool="restart", args={"service": "$service"},
                             typical_effect="resolves", strength=0.9, successes=3, failures=0))
    h = HabitPlugin(CFG, store)
    c = ctx(feats={"service": "checkout", "symptom": "oom"})
    resp = await h.before_model_callback(callback_context=c, llm_request=None)
    fc = resp.get_function_calls()[0]
    assert fc.name == "restart" and fc.args["service"] == "checkout"
    assert fc.args["expected_effect"] == "resolves" and fc.args["confidence"] == 0.9
    assert c.state[K_HABIT_HIT] is True and c.state["subcortex.metrics"]["habit_hits"] == 1
    assert await h.before_model_callback(callback_context=ctx(acted=True), llm_request=None) is None
    # sin el hallazgo del diagnóstico la clase de escena es otra: no hay bypass
    no_finding = ctx()
    no_finding.state[K_DISCOVERED] = {}
    assert await h.before_model_callback(callback_context=no_finding, llm_request=None) is None
    weak = EpisodicStore(":memory:")
    weak.upsert_habit(Habit(scene_key=KEY, tool="restart", args={}, typical_effect="resolves",
                            strength=0.5, successes=3, failures=1))
    assert await HabitPlugin(CFG, weak).before_model_callback(callback_context=ctx(), llm_request=None) is None


@pytest.mark.asyncio
async def test_different_args_never_compile_a_habit():
    """Tres edit_file exitosos con old/new distintos no son 'la misma acción': no hay hábito."""
    store = EpisodicStore(":memory:")
    h = HabitPlugin(SubcortexConfig(risk={"edit_file": "free"}, coarse_features=("symptom", "finding")), store)
    for i in range(3):
        store.record_outcome(KEY, "edit_file", True)
        le_ = {"tool": "edit_file", "args": {"path": "a.py", "old": f"x{i}", "new": f"y{i}"},
               "expected": "resolves", "observed": "resolves", "confidence": 0.9, "error": 0.0,
               "status": "success", "call_id": "c1"}
        await h.after_tool_callback(tool=Tool("edit_file"), tool_args={}, tool_context=ctx(le_),
                                    result={"status": "success"})
    assert store.get_habit(KEY) is None


@pytest.mark.asyncio
async def test_invalid_habit_call_weakens_and_fires_once_per_episode():
    from subcortex.habit import K_HABIT_TRIED
    store = EpisodicStore(":memory:")
    store.upsert_habit(Habit(scene_key=KEY, tool="restart", args={"service": "$service"},
                             typical_effect="resolves", strength=0.9, successes=3, failures=0))
    h = HabitPlugin(CFG, store)
    c = ctx()
    assert await h.before_model_callback(callback_context=c, llm_request=None) is not None
    assert c.state[K_HABIT_TRIED] is True
    await h.after_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=c,
                                result={"status": "invalid", "observed_effect": "no_change"})
    assert store.get_habit(KEY).strength == pytest.approx(0.45)
    assert c.state[K_HABIT_HIT] is False and c.state["subcortex.metrics"]["dehabituations"] == 1
    # mismo episodio: no vuelve a dispararse aunque no haya actuado
    assert await h.before_model_callback(callback_context=c, llm_request=None) is None


@pytest.mark.asyncio
async def test_dehabituation_on_negative_error_and_acted_flag():
    store = EpisodicStore(":memory:")
    store.upsert_habit(Habit(scene_key=KEY, tool="restart", args={"service": "$service"},
                             typical_effect="resolves", strength=0.9, successes=3, failures=0))
    h = HabitPlugin(CFG, store)
    c = ctx(le("restart", -0.8, "worsens"), habit_hit=True)
    await h.after_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=c, result={"status": "success"})
    hb = store.get_habit(KEY)
    assert hb.strength == pytest.approx(0.45) and hb.failures == 1
    assert c.state[K_ACTED] is True and c.state[K_HABIT_HIT] is False
    assert c.state["subcortex.metrics"]["dehabituations"] == 1
