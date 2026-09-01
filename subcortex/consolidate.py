"""Sueño / microglía: poda, refuerzo y destilación episódico → semántico. Sin LLM."""
from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable

from .store import EpisodicStore
from .types import Episode, Rule

DAY = 86400.0


def consolidate(store: EpisodicStore, now: float | None = None, min_support: int = 3,
                llm: Callable[[str], str] | None = None) -> dict:
    """Sueño: decae, poda, refuerza y destila. Con `llm` (texto → texto) además escribe reglas
    en lenguaje natural a partir de los episodios; sin él, solo la destilación estadística."""
    now = time.time() if now is None else now
    stats = {"decayed": 0, "pruned": 0, "reinforced": 0, "rules": 0, "llm_rules": 0}
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
    if llm is not None:
        stats["llm_rules"] = _distill_llm(store, llm, min_support)
    return stats


def _distill_llm(store: EpisodicStore, llm: Callable[[str], str], min_support: int) -> int:
    """Le pide al modelo reglas generales a partir de los episodios (una pasada, barata).

    Contrato de salida: un JSON `[{"pattern": {feature: valor}, "tool": str, "text": str}]`.
    Cada regla se guarda con `support` = episodios que matchean el patrón; se descartan las que
    matchean menos de `min_support`, así el modelo no puede inventar reglas sin evidencia."""
    eps = store.all_episodes()
    if len(eps) < min_support:
        return 0
    lines = [f"- escena {json.dumps(e.features, sort_keys=True, ensure_ascii=False)} · acción {e.tool} · "
             f"esperaba {e.expected} · resultó {e.observed}" for e in eps[-60:]]
    prompt = ("Sos la memoria de un agente. Estos son episodios (escena, acción, esperado, resultado). "
              "Escribí hasta 5 reglas generales, cada una con un patrón de features (subconjunto de la "
              "escena) y la acción a la que aplica. Solo reglas que se repitan en los episodios. Respondé "
              "ÚNICAMENTE con JSON: [{\"pattern\": {feature: valor}, \"tool\": nombre, \"text\": regla}].\n\n"
              + "\n".join(lines))
    try:
        raw = llm(prompt)
        start, end = raw.find("["), raw.rfind("]")
        rules = json.loads(raw[start:end + 1])
    except Exception:  # noqa: BLE001 — salida no parseable: no hay reglas, no hay error
        return 0
    n = 0
    for r in rules:
        pattern = {str(k): str(v) for k, v in (r.get("pattern") or {}).items()}
        tool, text = r.get("tool"), r.get("text")
        if not pattern or not tool or not text:
            continue
        support = sum(1 for e in eps if e.tool == tool and all(e.features.get(k) == v for k, v in pattern.items()))
        if support < min_support:
            continue
        store.upsert_rule(Rule(scene_pattern=pattern, tool=str(tool), text=f"{text} (n={support})", support=support))
        n += 1
    return n


def suggest_coarse_features(store: EpisodicStore, max_k: int = 2) -> list[tuple[str, float]]:
    """Escena aprendida: qué features separan mejor éxitos de fracasos (ganancia de información
    sobre `habenula`). Devuelve [(feature, IG)] ordenado; las top-k son candidatas a `coarse_features`.
    No se aplica solo: cambiar la clave a mitad de camino invalidaría dopamina y hábitos."""
    eps = store.all_episodes()
    if len(eps) < 6:
        return []

    def entropy(rows: list[Episode]) -> float:
        n = len(rows)
        if n == 0:
            return 0.0
        p = sum(1 for e in rows if e.habenula) / n
        return 0.0 if p in (0.0, 1.0) else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

    base = entropy(eps)
    feats = Counter(k for e in eps for k in e.features)
    out = []
    for f, cnt in feats.items():
        if cnt < len(eps) * 0.8:  # feature ausente en muchos episodios: no sirve de clave
            continue
        groups: dict[str, list[Episode]] = defaultdict(list)
        for e in eps:
            groups[e.features.get(f, "∅")].append(e)
        cond = sum(len(g) / len(eps) * entropy(g) for g in groups.values())
        out.append((f, round(base - cond, 4)))
    out.sort(key=lambda x: -x[1])
    return out[:max(max_k, len(out))]


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
