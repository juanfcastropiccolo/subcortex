from __future__ import annotations

from typing import Any

from .types import K_METRICS

METRIC_KEYS = (
    "llm_calls", "tokens", "steps", "vetoes", "vetoes_irreversible", "rejected",
    "episodes_written", "habit_hits", "dehabituations", "abs_error_sum", "error_count", "stalled", "reconsiders",
)


def get_metrics(state: Any) -> dict[str, float]:
    m = dict(state.get(K_METRICS) or {})
    for k in METRIC_KEYS:
        m.setdefault(k, 0)
    return m


def bump(state: Any, key: str, n: float = 1) -> None:
    m = get_metrics(state)
    m[key] = m.get(key, 0) + n
    state[K_METRICS] = m


def set_metric(state: Any, key: str, value: float) -> None:
    m = get_metrics(state)
    m[key] = value
    state[K_METRICS] = m
