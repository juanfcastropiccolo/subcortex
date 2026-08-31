import pytest

from subcortex.store import EpisodicStore
from subcortex.types import Episode, Habit, Rule, Scene


def ep(scene: Scene, tool="restart", err=-0.6, **kw) -> Episode:
    return Episode(scene_key=scene.key, features=scene.features, tool=tool, args={"service": "api"},
                   expected="resolves", observed="worsens", prediction_error=err, valence=err,
                   habenula=err < 0, strength=abs(err), **kw)


@pytest.fixture
def store():
    s = EpisodicStore(":memory:")
    yield s
    s.close()


def test_write_and_recall_same_scene_failures_first(store):
    scene = Scene.from_features({"service": "api", "symptom": "high_latency"})
    store.write(ep(scene, err=0.3, created_at=1, last_access=1))
    store.write(ep(scene, err=-0.6, created_at=2, last_access=2))
    got = store.recall(scene, k=5, now=100)
    assert [e.prediction_error for e in got] == [-0.6, 0.3]
    assert all(e.access_count == 1 and e.last_access == 100 for e in got)


def test_recall_ranks_exact_scene_over_partial_overlap(store):
    exact = Scene.from_features({"service": "api", "symptom": "oom"})
    partial = Scene.from_features({"service": "auth", "symptom": "oom"})
    other = Scene.from_features({"service": "auth", "symptom": "errors_5xx"})
    store.write(ep(partial, err=-0.9))
    store.write(ep(exact, err=-0.3))
    store.write(ep(other, err=-0.9))
    got = store.recall(exact, k=2, now=0)
    assert [e.scene_key for e in got] == [exact.key, partial.key]


def test_dopamine_prior_and_update(store):
    assert store.dopamine("s", "restart", prior=0.6) == pytest.approx(0.6)
    store.record_outcome("s", "restart", True)
    store.record_outcome("s", "restart", True)
    store.record_outcome("s", "restart", False)
    assert store.outcome_counts("s", "restart") == (2, 1)
    # (2 + 0.6*2) / (3 + 2) = 0.64
    assert store.dopamine("s", "restart", prior=0.6) == pytest.approx(0.64)
    store.record_outcome("s", "scale", True)
    store.record_outcome("s", "rollback", False)
    assert store.best_tools("s") == [("restart", 2, 1), ("scale", 1, 0)]


def test_habit_and_rule_roundtrip(store):
    h = Habit(scene_key="s", tool="restart", args={"service": "api"}, typical_effect="resolves",
              strength=0.85, successes=3, failures=0)
    store.upsert_habit(h)
    assert store.get_habit("s") == h
    h.strength = 0.4
    store.upsert_habit(h)
    assert store.get_habit("s").strength == 0.4 and len(store.habits()) == 1
    r = Rule(scene_pattern={"symptom": "oom"}, tool="restart", text="t", support=3)
    store.upsert_rule(r)
    store.upsert_rule(Rule(scene_pattern={"symptom": "oom"}, tool="restart", text="t2", support=4))
    rules = store.rules_for(Scene.from_features({"symptom": "oom", "service": "x"}))
    assert len(rules) == 1 and rules[0].support == 4
    assert store.rules_for(Scene.from_features({"symptom": "errors_5xx"})) == []


def test_strength_update_delete_and_stats(store):
    scene = Scene.from_features({"a": "1"})
    i = store.write(ep(scene))
    store.update_strength(i, 0.01)
    assert store.all_episodes()[0].strength == 0.01
    store.delete_episodes([i])
    assert store.all_episodes() == []
    assert store.stats()["episodes"] == 0
