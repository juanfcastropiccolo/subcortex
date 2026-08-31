from types import SimpleNamespace

import pytest

from subcortex.config import SubcortexConfig
from subcortex.memory import MemoryPlugin, render_precedents
from subcortex.store import EpisodicStore
from subcortex.types import K_DISCOVERED, K_FEATURES, K_LAST_ERROR, K_TONE, Rule, Scene

CFG = SubcortexConfig(risk={"restart": "costly"}, diagnostic_tools=frozenset({"inspect_service"}),
                      coarse_features=("symptom", "finding"))
FEATS = {"service": "api", "symptom": "high_latency"}


class Tool:
    def __init__(self, name): self.name = name


def ctx(last_error, tone=1.0, feats=None, discovered=None):
    return SimpleNamespace(function_call_id="c1", state={
        K_FEATURES: feats or FEATS, K_LAST_ERROR: last_error, K_TONE: tone,
        K_DISCOVERED: discovered or {}})


def le(err, observed="worsens", status="success"):
    return {"tool": "restart", "args": {"service": "api"}, "expected": "resolves", "observed": observed,
            "confidence": 1.0, "error": err, "status": status, "call_id": "c1"}


@pytest.mark.asyncio
async def test_writes_only_on_surprise_but_always_updates_dopamine_on_coarse_key():
    store = EpisodicStore(":memory:")
    m = MemoryPlugin(CFG, store)
    c = ctx(le(0.0, observed="resolves"))
    await m.after_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=c, result={"status": "success"})
    assert store.stats()["episodes"] == 0
    coarse = Scene.from_features(FEATS, coarse=CFG.coarse_features).coarse_key
    assert store.outcome_counts(coarse, "restart") == (1, 0)
    c = ctx(le(-1.0))
    await m.after_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=c, result={"status": "success"})
    eps = store.all_episodes()
    assert len(eps) == 1 and eps[0].habenula and eps[0].strength == 1.0
    assert c.state["subcortex.metrics"]["episodes_written"] == 1


@pytest.mark.asyncio
async def test_diagnostic_discovery_enriches_scene():
    store = EpisodicStore(":memory:")
    m = MemoryPlugin(CFG, store)
    c = ctx(None)
    await m.after_tool_callback(tool=Tool("inspect_service"), tool_args={}, tool_context=c,
                                result={"status": "success", "finding": "db_pool_exhausted"})
    assert c.state[K_DISCOVERED] == {"finding": "db_pool_exhausted"}
    assert CFG.scene_of(c.state).features["finding"] == "db_pool_exhausted"
    await m.after_tool_callback(tool=Tool("inspect_service"), tool_args={}, tool_context=c,
                                result={"status": "success", "metrics": "sin finding"})
    assert c.state[K_DISCOVERED] == {"finding": "db_pool_exhausted"}


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
async def test_before_model_injects_precedents_with_enough_overlap_and_rules():
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
    # overlap 1 (solo symptom) < recall_min_overlap 2: no se inyecta el precedente, sí la regla
    req2 = SimpleNamespace(appended=[])
    req2.append_instructions = lambda xs: req2.appended.extend(xs)
    other = ctx(None, feats={"service": "zzz", "symptom": "high_latency"})
    await m.before_model_callback(callback_context=other, llm_request=req2)
    assert "Precedentes" not in "\n".join(req2.appended) and "Reglas" in "\n".join(req2.appended)


def test_render_precedents_format():
    txt = render_precedents([])
    assert txt == ""
