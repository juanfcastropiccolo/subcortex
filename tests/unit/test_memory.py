from types import SimpleNamespace

import pytest

from subcortex.config import SubcortexConfig
from subcortex.memory import MemoryPlugin, render_precedents
from subcortex.store import EpisodicStore
from subcortex.types import K_FEATURES, K_LAST_ERROR, K_TONE, Rule, Scene

CFG = SubcortexConfig(risk={"restart": "costly"})
FEATS = {"service": "api", "symptom": "high_latency"}


class Tool:
    def __init__(self, name): self.name = name


def ctx(last_error, tone=1.0):
    return SimpleNamespace(function_call_id="c1",
                           state={K_FEATURES: FEATS, K_LAST_ERROR: last_error, K_TONE: tone})


def le(err, observed="worsens", status="success"):
    return {"tool": "restart", "args": {"service": "api"}, "expected": "resolves", "observed": observed,
            "confidence": 1.0, "error": err, "status": status, "call_id": "c1"}


@pytest.mark.asyncio
async def test_writes_only_on_surprise_but_always_updates_dopamine():
    store = EpisodicStore(":memory:")
    m = MemoryPlugin(CFG, store)
    c = ctx(le(0.0, observed="resolves"))
    await m.after_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=c, result={"status": "success"})
    assert store.stats()["episodes"] == 0
    assert store.outcome_counts(Scene.from_features(FEATS).key, "restart") == (1, 0)
    c = ctx(le(-1.0))
    await m.after_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=c, result={"status": "success"})
    eps = store.all_episodes()
    assert len(eps) == 1 and eps[0].habenula and eps[0].strength == 1.0
    assert c.state["subcortex.metrics"]["episodes_written"] == 1


@pytest.mark.asyncio
async def test_skips_blocked_and_missing_error():
    store = EpisodicStore(":memory:")
    m = MemoryPlugin(CFG, store)
    await m.after_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=ctx(le(-1.0)),
                                result={"status": "vetoed"})
    await m.after_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=ctx(None),
                                result={"status": "success"})
    assert store.stats()["episodes"] == 0


@pytest.mark.asyncio
async def test_before_model_injects_precedents_and_rules_by_scene():
    store = EpisodicStore(":memory:")
    m = MemoryPlugin(CFG, store)
    c = ctx(le(-1.0))
    await m.after_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=c, result={"status": "success"})
    store.upsert_rule(Rule(scene_pattern={"symptom": "high_latency"}, tool="restart",
                           text="restart suele empeorar", support=3))
    req = SimpleNamespace(appended=[])
    req.append_instructions = lambda xs: req.appended.extend(xs)
    assert await m.before_model_callback(callback_context=ctx(None, tone=0.5), llm_request=req) is None
    text = "\n".join(req.appended)
    assert "Precedentes" in text and "[FRACASO]" in text and "restart" in text
    assert "Reglas aprendidas" in text and "suele empeorar" in text
    req2 = SimpleNamespace(appended=[])
    req2.append_instructions = lambda xs: req2.appended.extend(xs)
    other = SimpleNamespace(state={K_FEATURES: {"service": "zzz", "symptom": "oom"}, K_TONE: 1.0})
    await m.before_model_callback(callback_context=other, llm_request=req2)
    assert req2.appended == []


def test_render_precedents_format():
    txt = render_precedents([])
    assert txt == ""
