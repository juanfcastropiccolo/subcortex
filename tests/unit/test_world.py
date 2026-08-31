from opsworld.world import EFFECTS, Incident, World, features, generate_incidents


def make(cause: str) -> Incident:
    return Incident(id=1, service="api", cause=cause, recent_deploy=(cause == "bad_deploy"),
                    traffic="spike" if cause == "traffic_spike" else "normal", hour_bucket="evening")


def test_generate_is_deterministic_and_cycles_causes():
    a = generate_incidents(12, seed=7)
    b = generate_incidents(12, seed=7)
    assert [i.cause for i in a] == [i.cause for i in b]
    assert len({i.cause for i in a}) == 6


def test_features_hide_cause():
    f = features(make("memory_leak"))
    assert "cause" not in f
    assert set(f) == {"service", "symptom", "recent_deploy", "traffic", "hour_bucket"}


def test_restart_resolves_leak_and_scores():
    w = World(make("memory_leak"))
    r = w.act("restart", service="api")
    assert r["observed_effect"] == "resolves"
    assert w.resolved and w.done
    assert w.score == 100 - 10 - 5  # resolved, costly, one step


def test_restart_worsens_db_and_damages():
    w = World(make("db_saturated"))
    r = w.act("restart", service="api")
    assert r["observed_effect"] == "worsens"
    assert w.worsens == 1 and w.score == -10 - 40 - 5


def test_false_resolve_is_penalised_and_ends():
    w = World(make("bad_deploy"))
    r = w.act("resolve")
    assert r["observed_effect"] == "worsens" and w.done and not w.resolved
    assert w.score == -30 - 5


def test_escalate_only_pays_for_dependency_down():
    w = World(make("dependency_down"))
    assert w.act("escalate_to_human", reason="x")["observed_effect"] == "resolves"
    assert w.score == 30 - 5 and w.done
    w2 = World(make("memory_leak"))
    assert w2.act("escalate_to_human", reason="x")["observed_effect"] == "no_change"
    assert w2.score == -5 and w2.done


def test_step_cap_ends_episode():
    w = World(make("dependency_down"))
    for _ in range(8):
        w.diagnose("inspect_service", service="api")
    assert w.done and w.steps == 8


def test_diagnostic_hints_reflect_cause():
    r = World(make("memory_leak")).diagnose("inspect_service", service="api")
    assert "memory" in r["metrics"] and r["finding"] == "memory_high"
    assert World(make("bad_deploy")).diagnose("check_deploys", service="api")["recent_deploy"] is True
    assert EFFECTS["false_alarm"]["resolve"] == "resolves"
