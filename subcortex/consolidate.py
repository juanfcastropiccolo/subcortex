"""Sueño / microglía: poda, refuerzo y destilación episódico → semántico. Sin LLM."""
from __future__ import annotations

import time
from collections import defaultdict

from .store import EpisodicStore
from .types import Episode, Rule

DAY = 86400.0


def consolidate(store: EpisodicStore, now: float | None = None, min_support: int = 3) -> dict:
    now = time.time() if now is None else now
    stats = {"decayed": 0, "pruned": 0, "reinforced": 0, "rules": 0}
    to_delete: list[int] = []
    for e in store.all_episodes():
        days_idle = max(0.0, (now - e.last_access) / DAY)
        strength = e.strength * (0.9 ** days_idle)
        if e.access_count >= 3:
            strength = min(1.0, strength * 1.2)
            stats["reinforced"] += 1
        if strength < 0.05 and e.access_count == 0 and (now - e.created_at) > 7 * DAY:
            to_delete.append(e.id)
            continue
        if strength != e.strength:
            store.update_strength(e.id, round(strength, 4))
            stats["decayed"] += 1
    store.delete_episodes(to_delete)
    stats["pruned"] = len(to_delete)
    stats["rules"] = _distill(store, min_support)
    return stats


def _distill(store: EpisodicStore, min_support: int) -> int:
    groups: dict[tuple[str, str], list[Episode]] = defaultdict(list)
    for e in store.all_episodes():
        groups[(e.tool, e.observed)].append(e)
    n = 0
    for (tool, observed), eps in groups.items():
        if len(eps) < min_support:
            continue
        common = dict(eps[0].features.items())
        for e in eps[1:]:
            common = {k: v for k, v in common.items() if e.features.get(k) == v}
        if not common:
            continue
        desc = ", ".join(f"{k}={v}" for k, v in sorted(common.items()))
        text = f"En incidentes con {desc}, {tool} tiende a {observed} (n={len(eps)})."
        store.upsert_rule(Rule(scene_pattern=common, tool=tool, text=text, support=len(eps)))
        n += 1
    return n
