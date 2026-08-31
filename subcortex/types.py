"""Tipos compartidos de subcortex. Cada uno mapea a una pieza del ensayo."""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, Field

Effect = Literal["resolves", "improves", "no_change", "worsens", "diagnostic"]
EFFECTS: tuple[str, ...] = ("resolves", "improves", "no_change", "worsens", "diagnostic")
EFFECT_RANK: dict[str, int] = {"worsens": -1, "no_change": 0, "improves": 1, "resolves": 2}
SUCCESS_EFFECTS = {"improves", "resolves"}

RiskClass = Literal["free", "costly", "irreversible"]

# Claves de estado de sesión (persisten dentro de la sesión; legibles tras el run).
K_TONE = "subcortex.tone"
K_INTERO = "subcortex.intero"
K_PENDING = "subcortex.pending"
K_LAST_ERROR = "subcortex.last_error"
K_ACTED = "subcortex.acted"
K_HABIT_HIT = "subcortex.habit_hit"
K_METRICS = "subcortex.metrics"
K_FEATURES = "subcortex.features"
K_DISCOVERED = "subcortex.discovered"
K_VETO_LOG = "subcortex.veto_log"

PRED_PARAMS = ("expected_effect", "confidence")


def _hash(d: dict[str, str]) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


class Scene(BaseModel):
    """Contexto observable de la decisión (paso 8.1, giro parahipocampal).

    `key` identifica la escena exacta (todas las features): la usa el recuerdo episódico.
    `coarse_key` identifica la clase de escena (subconjunto de features): la usan la
    dopamina, el gate y los hábitos, que necesitan repetición para aprender.
    """

    key: str
    coarse_key: str
    features: dict[str, str]

    @classmethod
    def from_features(cls, features: dict[str, Any],
                      coarse: Iterable[str] | None = None) -> Scene:
        norm = {str(k): str(v) for k, v in features.items()}
        if coarse is None:
            coarse_key = _hash(norm)
        else:
            coarse_key = _hash({k: norm[k] for k in coarse if k in norm})
        return cls(key=_hash(norm), coarse_key=coarse_key, features=norm)


class Prediction(BaseModel):
    """Copia eferente (paso 13): qué espera el agente de la acción."""

    tool: str
    args: dict[str, Any]
    expected: str
    confidence: float = Field(ge=0.0, le=1.0)


class Episode(BaseModel):
    id: int | None = None
    scene_key: str
    features: dict[str, str]
    tool: str
    args: dict[str, Any]
    expected: str
    observed: str
    prediction_error: float
    valence: float
    habenula: bool
    strength: float
    access_count: int = 0
    created_at: float = Field(default_factory=time.time)
    last_access: float = Field(default_factory=time.time)


class Habit(BaseModel):
    """`args` es una plantilla: un valor "$service" se resuelve con la feature `service`
    de la escena actual, así el hábito aprendido en `api` sirve en `checkout`."""

    scene_key: str
    tool: str
    args: dict[str, Any]
    typical_effect: str
    strength: float
    successes: int
    failures: int


class Rule(BaseModel):
    scene_pattern: dict[str, str]
    tool: str
    text: str
    support: int


def prediction_error(expected: str, observed: str, confidence: float) -> float:
    """Error con signo en [-1, 1]. Positivo = mejor de lo esperado."""
    if expected == "diagnostic" or observed == "diagnostic":
        return 0.0
    diff = EFFECT_RANK[observed] - EFFECT_RANK[expected]
    scaled = max(-1.0, min(1.0, diff / 2.0))
    return scaled * confidence
