from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .types import K_FEATURES, Scene


def default_scene_fn(state: Any) -> Scene | None:
    feats = state.get(K_FEATURES) if hasattr(state, "get") else None
    return Scene.from_features(feats) if feats else None


@dataclass
class SubcortexConfig:
    risk: dict[str, str]  # tool -> RiskClass
    diagnostic_tools: frozenset[str] = frozenset()
    scene_fn: Callable[[Any], Scene | None] = default_scene_fn
    always_allowed: frozenset[str] = frozenset({"escalate_to_human", "resolve"})
    step_budget: int = 8
    surprise_threshold: float = 0.25
    gate_threshold: float = 0.10
    cost: dict[str, float] = field(
        default_factory=lambda: {"free": 0.0, "costly": 0.15, "irreversible": 0.35}
    )
    dopamine_prior: float = 0.6
    hyperdirect_confidence: float = 0.7
    hyperdirect_tone: float = 0.4
    max_consecutive_blocks: int = 3
    habit_min_successes: int = 3
    habit_min_strength: float = 0.8

    @property
    def action_tools(self) -> frozenset[str]:
        return frozenset(self.risk)

    def is_action(self, name: str) -> bool:
        return name in self.risk

    def risk_of(self, name: str) -> str:
        return self.risk.get(name, "free")
