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
