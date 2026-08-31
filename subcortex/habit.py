"""Caudado → putamen: decisiones repetidas se compilan y saltean al LLM."""
from __future__ import annotations

import logging
from typing import Any

from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from .config import SubcortexConfig
from .metrics import bump
from .store import EpisodicStore
from .types import K_ACTED, K_HABIT_HIT, K_LAST_ERROR, SUCCESS_EFFECTS, Habit, Scene

log = logging.getLogger("subcortex")
BLOCK_STATUSES = {"vetoed", "rejected"}


def templatize_args(args: dict[str, Any], features: dict[str, str]) -> dict[str, Any]:
    """Reemplaza valores que coinciden con una feature por una referencia "$feature"."""
    out = {}
    for k, v in args.items():
        ref = next((f"${fk}" for fk, fv in features.items() if str(v) == fv), None)
        out[k] = ref if ref else v
    return out


def resolve_args(template: dict[str, Any], features: dict[str, str]) -> dict[str, Any]:
    out = {}
    for k, v in template.items():
        if isinstance(v, str) and v.startswith("$") and v[1:] in features:
            out[k] = features[v[1:]]
        else:
            out[k] = v
    return out


def habit_response(habit: Habit, scene: Scene) -> LlmResponse:
    args = {**resolve_args(habit.args, scene.features), "expected_effect": habit.typical_effect,
            "confidence": round(habit.strength, 3)}
    part = types.Part(function_call=types.FunctionCall(name=habit.tool, args=args))
    return LlmResponse(content=types.Content(role="model", parts=[part]),
                       custom_metadata={"subcortex": "habit"})


class HabitPlugin(BasePlugin):
    def __init__(self, cfg: SubcortexConfig, store: EpisodicStore) -> None:
        super().__init__(name="subcortex_habit")
        self.cfg = cfg
        self.store = store

    async def before_model_callback(self, *, callback_context, llm_request):
        try:
            state = callback_context.state
            if state.get(K_ACTED):
                return None
            scene = self.cfg.scene_of(state)
            if scene is None:
                return None
            habit = self.store.get_habit(scene.coarse_key)
            if not habit or habit.strength < self.cfg.habit_min_strength:
                return None
            if self.cfg.risk_of(habit.tool) == "irreversible":
                return None
            state[K_HABIT_HIT] = True
            bump(state, "habit_hits")
            log.info("hábito: %s en escena %s (fuerza %.2f), sin LLM",
                     habit.tool, scene.coarse_key, habit.strength)
            return habit_response(habit, scene)
        except Exception:
            log.exception("habit.before_model")
            return None

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        if not self.cfg.is_action(tool.name):
            return
        try:
            state = tool_context.state
            if (result or {}).get("status") in BLOCK_STATUSES:
                return
            state[K_ACTED] = True
            le = state.get(K_LAST_ERROR)
            if not le or le.get("tool") != tool.name:
                return
            scene = self.cfg.scene_of(state)
            if scene is None:
                return
            err = float(le["error"])
            habit = self.store.get_habit(scene.coarse_key)
            if state.get(K_HABIT_HIT) and habit and habit.tool == tool.name:
                state[K_HABIT_HIT] = False
                if err < 0:
                    habit.strength = round(habit.strength * 0.5, 4)
                    habit.failures += 1
                    self.store.upsert_habit(habit)
                    bump(state, "dehabituations")
                    log.info("des-habituación: %s cae a %.2f", habit.tool, habit.strength)
                    return
            if le["observed"] not in SUCCESS_EFFECTS or self.cfg.risk_of(tool.name) == "irreversible":
                return
            s, f = self.store.outcome_counts(scene.coarse_key, tool.name)
            if s >= self.cfg.habit_min_successes and f == 0:
                strength = min(1.0, 0.8 + 0.05 * (s - self.cfg.habit_min_successes))
                self.store.upsert_habit(Habit(
                    scene_key=scene.coarse_key, tool=tool.name,
                    args=templatize_args(le["args"], scene.features),
                    typical_effect=le["observed"], strength=strength, successes=s, failures=f))
        except Exception:
            log.exception("habit.after_tool")
