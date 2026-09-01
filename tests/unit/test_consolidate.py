from subcortex.consolidate import consolidate
from subcortex.store import EpisodicStore
from subcortex.types import Episode, Scene

DAY = 86400


def ep(feats, tool="restart", err=-0.8, created=0.0, access=0, last=0.0):
    s = Scene.from_features(feats)
    return Episode(scene_key=s.key, features=s.features, tool=tool, args={}, expected="resolves",
                   observed="worsens" if err < 0 else "resolves", prediction_error=err, valence=err,
                   habenula=err < 0, strength=abs(err), access_count=access, created_at=created,
                   last_access=last)


def test_decay_prune_reinforce():
    store = EpisodicStore(":memory:")
    old_unused = store.write(ep({"a": "1"}, err=-0.1, created=0, last=0))      # 0.1*0.9^30 < 0.05, >7 días
    popular = store.write(ep({"a": "2"}, err=-0.5, access=3, last=29 * DAY))   # refuerzo
    stats = consolidate(store, now=30 * DAY)
    ids = {e.id: e for e in store.all_episodes()}
    assert old_unused not in ids and stats["pruned"] == 1
    assert ids[popular].strength > 0.5 and stats["reinforced"] == 1


def test_rules_distilled_from_recurrent_pattern():
    store = EpisodicStore(":memory:")
    for svc in ("api", "auth", "search"):
        store.write(ep({"service": svc, "symptom": "high_latency", "traffic": "normal"}, last=0))
    store.write(ep({"service": "api", "symptom": "oom", "traffic": "normal"}, err=0.5, last=0))
    stats = consolidate(store, now=0)
    rules = store.rules_for(Scene.from_features({"service": "zzz", "symptom": "high_latency",
                                                 "traffic": "normal"}))
    assert stats["rules"] == 1 and len(rules) == 1
    assert rules[0].tool == "restart" and "worsens" in rules[0].text and rules[0].support == 3
    assert "service" not in rules[0].scene_pattern


def test_llm_rules_require_evidence():
    from subcortex.types import Scene
    store = EpisodicStore(":memory:")
    for svc in ("api", "auth", "search", "checkout"):
        store.write(ep({"service": svc, "symptom": "high_latency", "traffic": "normal"}, last=0))
    calls = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        return ("bla [{\"pattern\": {\"symptom\": \"high_latency\"}, \"tool\": \"restart\", "
                "\"text\": \"restart empeora con latencia alta\"}, "
                "{\"pattern\": {\"symptom\": \"oom\"}, \"tool\": \"restart\", \"text\": \"inventada\"}] bla")

    stats = consolidate(store, now=0, llm=fake_llm)
    assert stats["llm_rules"] == 1 and len(calls) == 1 and "episodios" in calls[0]
    rules = store.rules_for(Scene.from_features({"symptom": "high_latency", "service": "x"}))
    assert any("restart empeora" in r.text and "(n=4)" in r.text for r in rules)
    assert not store.rules_for(Scene.from_features({"symptom": "oom"}))  # sin evidencia, descartada
    assert consolidate(store, now=0, llm=lambda p: "no json")["llm_rules"] == 0


def test_suggest_coarse_features_ranks_by_information_gain():
    from subcortex.consolidate import suggest_coarse_features
    store = EpisodicStore(":memory:")
    # 'regime' separa perfectamente éxito/fracaso; 'hour' es ruido
    for i in range(8):
        regime = "bear" if i % 2 else "bull"
        store.write(ep({"regime": regime, "hour": str(i % 3)}, err=-0.8 if regime == "bear" else 0.5))
    ranked = suggest_coarse_features(store)
    assert ranked[0][0] == "regime" and ranked[0][1] > 0.9
    assert dict(ranked)["hour"] < 0.3
    assert suggest_coarse_features(EpisodicStore(":memory:")) == []
