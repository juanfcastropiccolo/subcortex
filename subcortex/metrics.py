from __future__ import annotations

from typing import Any

from .types import K_METRICS

METRIC_KEYS = (
    "llm_calls", "tokens", "steps", "vetoes", "vetoes_irreversible", "rejected",
    "episodes_written", "habit_hits", "dehabituations", "abs_error_sum", "error_count", "stalled", "reconsiders",
    # Desglose causal del "veto" (auditoría 2026-09-06): un veto por riesgo, un freno hiperdirecto,
    # un corte por racha de bloqueos y un descarte por arbitraje winner-take-all son cuatro cosas
    # distintas. Contarlas juntas infla el "efecto de seguridad" con decisiones que no bloquean nada.
    "vetoes_value",        # valor esperado < umbral
    "vetoes_hyperdirect",  # irreversible con poca confianza o tono bajo
    "vetoes_blockstreak",  # demasiadas acciones bloqueadas seguidas
    "arbitration_dropped",  # acciones paralelas descartadas por winner-take-all (NO bloquean nada)
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
