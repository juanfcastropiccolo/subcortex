from types import SimpleNamespace

import pytest

from subcortex.config import SubcortexConfig
from subcortex.habit import HabitPlugin
from subcortex.store import EpisodicStore
from subcortex.types import K_ACTED, K_FEATURES, K_HABIT_HIT, K_LAST_ERROR, Habit, Scene

CFG = SubcortexConfig(risk={"restart": "costly", "rollback": "irreversible"})
FEATS = {"service": "api", "symptom": "oom"}
KEY = Scene.from_features(FEATS).key


class Tool:
    def __init__(self, name): self.name = name


def ctx(last_error=None, acted=False, habit_hit=False):
    return SimpleNamespace(function_call_id="c1", state={
        K_FEATURES: FEATS, K_LAST_ERROR: last_error, K_ACTED: acted, K_HABIT_HIT: habit_hit})


def le(tool, err, observed):
    return {"tool": tool, "args": {"service": "api"}, "expected": "resolves", "observed": observed,
            "confidence": 0.9, "error": err, "status": "success", "call_id": "c1"}


@pytest.mark.asyncio
async def test_compiles_after_three_clean_successes_never_irreversible():
    store = EpisodicStore(":memory:")
    h = HabitPlugin(CFG, store)
    for _ in range(3):
        store.record_outcome(KEY, "restart", True)
        store.record_outcome(KEY, "rollback", True)
        await h.after_tool_callback(tool=Tool("restart"), tool_args={},
                                    tool_context=ctx(le("restart", 0.0, "resolves")),
                                    result={"status": "success"})
        await h.after_tool_callback(tool=Tool("rollback"), tool_args={},
                                    tool_context=ctx(le("rollback", 0.0, "resolves")),
                                    result={"status": "success"})
    hb = store.get_habit(KEY)
    assert hb and hb.tool == "restart" and hb.strength >= 0.8 and hb.typical_effect == "resolves"


@pytest.mark.asyncio
async def test_bypass_returns_function_call_once_per_episode():
    store = EpisodicStore(":memory:")
    store.upsert_habit(Habit(scene_key=KEY, tool="restart", args={"service": "api"},
                             typical_effect="resolves", strength=0.9, successes=3, failures=0))
    h = HabitPlugin(CFG, store)
    c = ctx()
    resp = await h.before_model_callback(callback_context=c, llm_request=None)
    fc = resp.get_function_calls()[0]
    assert fc.name == "restart" and fc.args["expected_effect"] == "resolves" and fc.args["confidence"] == 0.9
    assert c.state[K_HABIT_HIT] is True and c.state["subcortex.metrics"]["habit_hits"] == 1
    assert await h.before_model_callback(callback_context=ctx(acted=True), llm_request=None) is None
    weak = EpisodicStore(":memory:")
    weak.upsert_habit(Habit(scene_key=KEY, tool="restart", args={}, typical_effect="resolves",
                            strength=0.5, successes=3, failures=1))
    assert await HabitPlugin(CFG, weak).before_model_callback(callback_context=ctx(), llm_request=None) is None


@pytest.mark.asyncio
async def test_dehabituation_on_negative_error_and_acted_flag():
    store = EpisodicStore(":memory:")
    store.upsert_habit(Habit(scene_key=KEY, tool="restart", args={"service": "api"},
                             typical_effect="resolves", strength=0.9, successes=3, failures=0))
    h = HabitPlugin(CFG, store)
    c = ctx(le("restart", -0.8, "worsens"), habit_hit=True)
    await h.after_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=c, result={"status": "success"})
    hb = store.get_habit(KEY)
    assert hb.strength == pytest.approx(0.45) and hb.failures == 1
    assert c.state[K_ACTED] is True and c.state[K_HABIT_HIT] is False
    assert c.state["subcortex.metrics"]["dehabituations"] == 1
