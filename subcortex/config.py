from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .types import K_DISCOVERED, K_FEATURES, Scene


def default_scene_fn(state: Any) -> Scene | None:
    feats = state.get(K_FEATURES) if hasattr(state, "get") else None
    return Scene.from_features(feats) if feats else None


def default_discover_fn(tool: str, result: dict) -> dict[str, str]:
    """Qué aporta una tool diagnóstica a la escena: por convención, un campo `finding`."""
    finding = result.get("finding") if isinstance(result, dict) else None
    return {"finding": str(finding)} if finding else {}


@dataclass
class SubcortexConfig:
    risk: dict[str, str]  # tool -> RiskClass
    diagnostic_tools: frozenset[str] = frozenset()
    scene_fn: Callable[[Any], Scene | None] = default_scene_fn
    # Features que definen la *clase* de escena (dopamina, gate, hábitos). None = todas.
    coarse_features: tuple[str, ...] | None = None
    # Cómo una tool diagnóstica enriquece la escena durante el episodio.
    discover_fn: Callable[[str, dict], dict[str, str]] = default_discover_fn
    always_allowed: frozenset[str] = frozenset({"escalate_to_human", "resolve"})
    step_budget: int = 8
    surprise_threshold: float = 0.25
    gate_threshold: float = 0.10
    cost: dict[str, float] = field(
        default_factory=lambda: {"free": 0.0, "costly": 0.10, "irreversible": 0.20}
    )
    dopamine_prior: float = 0.6
    hyperdirect_confidence: float = 0.6
    hyperdirect_tone: float = 0.4
    trust_min_successes: int = 2  # éxitos sin fallos que eximen de la vía hiperdirecta
    max_consecutive_blocks: int = 3
    habit_min_successes: int = 3
    habit_min_strength: float = 0.8
    recall_min_overlap: int = 2
    recall_max: int = 4
    write_on_success: bool = True  # el éxito también es un episodio que vale recordar
    stall_fraction: float = 0.6    # presupuesto consumido sin resultado evaluado → "sin progreso"

    @property
    def action_tools(self) -> frozenset[str]:
        return frozenset(self.risk)

    def is_action(self, name: str) -> bool:
        return name in self.risk

    def is_diagnostic(self, name: str) -> bool:
        return name in self.diagnostic_tools

    def risk_of(self, name: str) -> str:
        return self.risk.get(name, "free")

    def scene_of(self, state: Any) -> Scene | None:
        """Escena actual = features de entrada + lo descubierto en este episodio."""
        base = self.scene_fn(state)
        if base is None:
            return None
        discovered = dict(state.get(K_DISCOVERED) or {}) if hasattr(state, "get") else {}
        return Scene.from_features({**base.features, **discovered}, coarse=self.coarse_features)
