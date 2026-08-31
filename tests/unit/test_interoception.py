from types import SimpleNamespace

import pytest

from subcortex.config import SubcortexConfig
from subcortex.interoception import InteroceptionPlugin, compute_tone, render_state
from subcortex.types import K_INTERO, K_TONE

CFG = SubcortexConfig(risk={"restart": "costly"}, diagnostic_tools=frozenset({"inspect_service"}))


def test_tone_starts_high_and_drops():
    fresh = {"steps": 0, "failures": 0, "blocks": 0}
    assert compute_tone(fresh, CFG) == 1.0
    assert compute_tone({"steps": 4, "failures": 0, "blocks": 0}, CFG) == pytest.approx(0.75)
    assert compute_tone({"steps": 0, "failures": 2, "blocks": 1}, CFG) == pytest.approx(0.7)
    assert compute_tone({"steps": 0, "failures": 0, "blocks": 5}, CFG) == 1.0  # los vetos no bajan el tono
    assert compute_tone({"steps": 8, "failures": 5, "blocks": 5}, CFG) == 0.05


def test_render_translates_not_dumps():
    txt = render_state({"steps": 6, "failures": 2, "blocks": 1, "last_status": "vetoed"}, 0.3, CFG)
    assert "Estado interno" in txt and "2 fallos seguidos" in txt and "75" in txt and "vetada" in txt


class Tool:
    def __init__(self, name): self.name = name


@pytest.mark.asyncio
async def test_plugin_updates_counters_and_injects():
    p = InteroceptionPlugin(CFG)
    ctx = SimpleNamespace(state={})
    await p.after_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=ctx,
                                result={"status": "vetoed"})
    await p.after_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=ctx,
                                result={"status": "error"})
    i = ctx.state[K_INTERO]
    assert i["steps"] == 1 and i["failures"] == 1 and i["blocks"] == 0
    assert ctx.state[K_TONE] < 1.0
    req = SimpleNamespace(appended=[])
    req.append_instructions = lambda xs: req.appended.extend(xs)
    assert await p.before_model_callback(callback_context=ctx, llm_request=req) is None
    assert "Estado interno" in req.appended[0]
