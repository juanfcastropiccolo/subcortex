import pytest
from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import subcortex
from opsworld.tools import ALL_TOOLS, registry
from opsworld.world import DIAGNOSTIC_TOOLS, RISK, Incident, World, features
from subcortex.types import K_FEATURES, Habit, Scene
from tests.integration.fake_llm import ScriptedLlm, call, text


def incident(cause="memory_leak"):
    return Incident(id=1, service="api", cause=cause, recent_deploy=False, traffic="normal",
                    hour_bucket="night")


async def run_episode(llm, inc, store_path=":memory:", sc=None, app=None):
    if app is None:
        agent = LlmAgent(name="ops", model=llm, instruction="Sos operador de guardia.",
                         tools=list(ALL_TOOLS))
        app = App(name="t", root_agent=agent)
    if sc is None:
        sc = subcortex.attach(app, risk=RISK, diagnostic_tools=DIAGNOSTIC_TOOLS, store_path=store_path)
    svc = InMemorySessionService()
    session = await svc.create_session(app_name="t", user_id="u", state={K_FEATURES: features(inc)})
    world = World(inc)
    registry.register(session.id, world)
    runner = Runner(app=app, session_service=svc)
    msg = types.Content(role="user", parts=[types.Part(text="Incidente: revisá y actuá.")])
    async for _ in runner.run_async(user_id="u", session_id=session.id, new_message=msg):
        pass
    session = await svc.get_session(app_name="t", user_id="u", session_id=session.id)
    return world, session.state, sc


@pytest.mark.asyncio
async def test_authorized_action_runs_and_protocol_is_injected():
    llm = ScriptedLlm(script=[call("restart", service="api", expected_effect="resolves", confidence=0.9),
                              text("listo")])
    world, state, sc = await run_episode(llm, incident("memory_leak"))
    assert world.resolved and llm.calls == 2
    assert "Protocolo de acción" in llm.seen_instructions[0]
    assert "Estado interno" in llm.seen_instructions[0]
    assert state["subcortex.metrics"]["llm_calls"] == 2 and state["subcortex.metrics"]["steps"] == 1


@pytest.mark.asyncio
async def test_veto_makes_llm_reiterate_and_tool_not_executed():
    llm = ScriptedLlm(script=[call("rollback", service="api", expected_effect="resolves", confidence=0.4),
                              call("escalate_to_human", reason="x", expected_effect="no_change",
                                   confidence=0.9),
                              text("escalado")])
    world, state, _ = await run_episode(llm, incident("bad_deploy"))
    assert world.steps == 1 and world.log[0]["action"] == "escalate_to_human"
    assert llm.calls == 3 and state["subcortex.metrics"]["vetoes_irreversible"] == 1


@pytest.mark.asyncio
async def test_missing_prediction_is_rejected():
    llm = ScriptedLlm(script=[call("restart", service="api"), text("ok")])
    world, state, _ = await run_episode(llm, incident("memory_leak"))
    assert world.steps == 0 and state["subcortex.metrics"]["rejected"] == 1


@pytest.mark.asyncio
async def test_surprise_writes_episode_with_habenula():
    llm = ScriptedLlm(script=[call("restart", service="api", expected_effect="resolves", confidence=1.0),
                              text("uh")])
    world, state, sc = await run_episode(llm, incident("db_saturated"))
    eps = sc.store.all_episodes()
    assert len(eps) == 1 and eps[0].habenula and eps[0].observed == "worsens"
    assert state["subcortex.metrics"]["episodes_written"] == 1


@pytest.mark.asyncio
async def test_habit_bypasses_llm():
    inc = incident("memory_leak")
    llm = ScriptedLlm(script=[text("fin")])
    agent = LlmAgent(name="ops", model=llm, instruction="x", tools=list(ALL_TOOLS))
    app = App(name="t", root_agent=agent)
    sc = subcortex.attach(app, risk=RISK, diagnostic_tools=DIAGNOSTIC_TOOLS)
    key = Scene.from_features(features(inc)).key
    sc.store.upsert_habit(Habit(scene_key=key, tool="restart", args={"service": "api"},
                                typical_effect="resolves", strength=0.9, successes=3, failures=0))
    world, state, _ = await run_episode(llm, inc, sc=sc, app=app)
    assert world.resolved and world.log[0]["action"] == "restart"
    assert llm.calls == 1 and state["subcortex.metrics"]["habit_hits"] == 1
    assert state["subcortex.metrics"]["llm_calls"] == 1


@pytest.mark.asyncio
async def test_parallel_actions_reduced_to_one():
    two = LlmResponse(content=types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(
            name="restart", args={"service": "api", "expected_effect": "resolves", "confidence": 0.5})),
        types.Part(function_call=types.FunctionCall(
            name="scale", args={"service": "api", "replicas": 4, "expected_effect": "resolves",
                                "confidence": 0.9})),
    ]))
    llm = ScriptedLlm(script=[two, text("ok")])
    world, state, _ = await run_episode(llm, incident("traffic_spike"))
    assert world.steps == 1 and world.log[0]["action"] == "scale" and world.resolved
