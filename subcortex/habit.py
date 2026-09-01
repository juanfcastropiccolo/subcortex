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

K_HABIT_TRIED = "subcortex.habit_tried"

log = logging.getLogger("subcortex")
BLOCK_STATUSES = {"vetoed", "rejected", "invalid"}


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


HABIT_MARK = b"subcortex:habit"


def habit_response(habit: Habit, scene: Scene) -> LlmResponse:
    args = {**resolve_args(habit.args, scene.features), "expected_effect": habit.typical_effect,
            "confidence": round(habit.strength, 3)}
    # La marca en thought_signature identifica la llamada como sintética para reescribirla
    # después: Gemini 3 rechaza (400) function calls en el historial que el modelo no generó.
    part = types.Part(function_call=types.FunctionCall(name=habit.tool, args=args),
                      thought_signature=HABIT_MARK)
    return LlmResponse(content=types.Content(role="model", parts=[part]),
                       custom_metadata={"subcortex": "habit"})


def _fmt_args(args: dict | None) -> str:
    return ", ".join(f"{k}={v}" for k, v in (args or {}).items())


def rewrite_habit_history(contents: list[types.Content]) -> int:
    """Convierte cada par (function_call sintética, function_response) en texto plano.

    Devuelve cuántas llamadas se reescribieron. Independiente del proveedor: el historial
    resultante es válido para cualquier modelo porque no contiene function calls ajenas."""
    synthetic_ids: set[str | None] = set()
    synthetic_names: set[str] = set()
    n = 0
    for content in contents:
        parts = content.parts or []
        new_parts = []
        for p in parts:
            fc = p.function_call
            fr = p.function_response
            if fc is not None and p.thought_signature == HABIT_MARK:
                synthetic_ids.add(fc.id)
                synthetic_names.add(fc.name)
                new_parts.append(types.Part(text=f"[hábito] Ejecuté {fc.name}({_fmt_args(fc.args)}) "
                                                 "sin deliberar, por experiencia previa en esta escena."))
                n += 1
            elif fr is not None and (fr.id in synthetic_ids or (fr.id is None and fr.name in synthetic_names)):
                new_parts.append(types.Part(text=f"[resultado de {fr.name}] {fr.response}"))
            else:
                new_parts.append(p)
        if len(new_parts) == len(parts):
            content.parts = new_parts
    return n


class HabitPlugin(BasePlugin):
    def __init__(self, cfg: SubcortexConfig, store: EpisodicStore) -> None:
        super().__init__(name="subcortex_habit")
        self.cfg = cfg
        self.store = store

    async def before_model_callback(self, *, callback_context, llm_request):
        try:
            state = callback_context.state
            if llm_request is not None and getattr(llm_request, "contents", None):
                rewrite_habit_history(llm_request.contents)
            if state.get(K_ACTED) or state.get(K_HABIT_TRIED):
                return None  # un hábito se intenta a lo sumo una vez por episodio
            scene = self.cfg.scene_of(state)
            if scene is None:
                return None
            habit = self.store.get_habit(scene.coarse_key)
            if not habit or habit.strength < self.cfg.habit_min_strength:
                return None
            if self.cfg.risk_of(habit.tool) == "irreversible":
                return None
            state[K_HABIT_HIT] = True
            state[K_HABIT_TRIED] = True
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
            scene = self.cfg.scene_of(state)
            habit = self.store.get_habit(scene.coarse_key) if scene else None
            fired = bool(state.get(K_HABIT_HIT)) and habit is not None and habit.tool == tool.name
            status = (result or {}).get("status")
            if status in BLOCK_STATUSES:
                if fired:  # el hábito produjo una llamada inválida o vetada: también es un fracaso
                    state[K_HABIT_HIT] = False
                    self._weaken(state, habit, f"resultado {status}")
                return
            state[K_ACTED] = True
            le = state.get(K_LAST_ERROR)
            if not le or le.get("tool") != tool.name or scene is None:
                return
            err = float(le["error"])
            if fired:
                state[K_HABIT_HIT] = False
                if err < 0:
                    self._weaken(state, habit, f"error {err:.2f}")
                    return
            if le["observed"] not in SUCCESS_EFFECTS or self.cfg.risk_of(tool.name) == "irreversible":
                return
            args = templatize_args(le["args"], scene.features)
            same = self.store.record_habit_candidate(scene.coarse_key, tool.name, args)
            s, f = self.store.outcome_counts(scene.coarse_key, tool.name)
            if same >= self.cfg.habit_min_successes and f == 0:
                strength = min(1.0, 0.8 + 0.05 * (same - self.cfg.habit_min_successes))
                self.store.upsert_habit(Habit(
                    scene_key=scene.coarse_key, tool=tool.name, args=args,
                    typical_effect=le["observed"], strength=strength, successes=same, failures=f))
        except Exception:
            log.exception("habit.after_tool")

    def _weaken(self, state, habit: Habit, why: str) -> None:
        habit.strength = round(habit.strength * 0.5, 4)
        habit.failures += 1
        self.store.upsert_habit(habit)
        bump(state, "dehabituations")
        log.info("des-habituación: %s cae a %.2f (%s)", habit.tool, habit.strength, why)
