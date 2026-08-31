"""Caudado → putamen: decisiones repetidas se compilan y saltean al LLM."""
from __future__ import annotations

import logging

from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from .config import SubcortexConfig
from .metrics import bump
from .store import EpisodicStore
from .types import K_ACTED, K_HABIT_HIT, K_LAST_ERROR, SUCCESS_EFFECTS, Habit

log = logging.getLogger("subcortex")
BLOCK_STATUSES = {"vetoed", "rejected"}


def habit_response(habit: Habit) -> LlmResponse:
    args = {**habit.args, "expected_effect": habit.typical_effect, "confidence": round(habit.strength, 3)}
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
            scene = self.cfg.scene_fn(state)
            if scene is None:
                return None
            habit = self.store.get_habit(scene.key)
            if not habit or habit.strength < self.cfg.habit_min_strength:
                return None
            if self.cfg.risk_of(habit.tool) == "irreversible":
                return None
            state[K_HABIT_HIT] = True
            bump(state, "habit_hits")
            log.info("hábito: %s en escena %s (fuerza %.2f), sin LLM", habit.tool, scene.key, habit.strength)
            return habit_response(habit)
        except Exception:
            log.exception("habit.before_model")
            return None

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        if not self.cfg.is_action(tool.name):
            return None
        try:
            state = tool_context.state
            if (result or {}).get("status") in BLOCK_STATUSES:
                return None
            state[K_ACTED] = True
            le = state.get(K_LAST_ERROR)
            if not le or le.get("tool") != tool.name:
                return None
            scene = self.cfg.scene_fn(state)
            if scene is None:
                return None
            err = float(le["error"])
            habit = self.store.get_habit(scene.key)
            if state.get(K_HABIT_HIT) and habit and habit.tool == tool.name:
                state[K_HABIT_HIT] = False
                if err < 0:
                    habit.strength = round(habit.strength * 0.5, 4)
                    habit.failures += 1
                    self.store.upsert_habit(habit)
                    bump(state, "dehabituations")
                    log.info("des-habituación: %s cae a %.2f", habit.tool, habit.strength)
                    return None
            if le["observed"] not in SUCCESS_EFFECTS or self.cfg.risk_of(tool.name) == "irreversible":
                return None
            s, f = self.store.outcome_counts(scene.key, tool.name)
            if s >= self.cfg.habit_min_successes and f == 0:
                strength = min(1.0, 0.8 + 0.05 * (s - self.cfg.habit_min_successes))
                self.store.upsert_habit(Habit(scene_key=scene.key, tool=tool.name, args=le["args"],
                                              typical_effect=le["observed"], strength=strength,
                                              successes=s, failures=f))
        except Exception:
            log.exception("habit.after_tool")
        return None
