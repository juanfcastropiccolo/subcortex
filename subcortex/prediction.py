"""Cerebelo: copia eferente antes de actuar, error de predicción después."""
from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any

from google.adk.plugins.base_plugin import BasePlugin

from .config import SubcortexConfig
from .metrics import bump
from .types import EFFECTS, K_LAST_ERROR, K_PENDING, PRED_PARAMS, Prediction, prediction_error

log = logging.getLogger("subcortex")
BLOCK_STATUSES = {"vetoed", "rejected", "invalid"}

PROTOCOL_INSTRUCTION = """## Protocolo de acción (subcortex)
Toda acción lleva `expected_effect` (resolves|improves|no_change|worsens) y `confidence` (0–1) honestos.
Si una acción vuelve "vetoed" o "rejected", leé `reason` y `hint` antes de decidir. Una acción por turno."""

_DOC_SUFFIX = """

Args extra (obligatorios):
    expected_effect: efecto esperado; uno de resolves, improves, no_change, worsens.
    confidence: confianza en ese efecto, entre 0 y 1."""


def wrap_action_tool(func: Callable[..., Any]) -> Callable[..., Any]:
    sig = inspect.signature(func)
    extra = [inspect.Parameter("expected_effect", inspect.Parameter.KEYWORD_ONLY, annotation=str),
             inspect.Parameter("confidence", inspect.Parameter.KEYWORD_ONLY, annotation=float)]
    new_params = list(sig.parameters.values()) + extra

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for p in PRED_PARAMS:
            kwargs.pop(p, None)
        return func(*args, **kwargs)

    wrapper.__signature__ = sig.replace(parameters=new_params)
    wrapper.__annotations__ = {**getattr(func, "__annotations__", {}),
                               "expected_effect": str, "confidence": float}
    wrapper.__doc__ = (inspect.cleandoc(func.__doc__ or func.__name__)) + _DOC_SUFFIX
    return wrapper


def parse_prediction(tool: str, args: dict) -> Prediction | str:
    exp = args.get("expected_effect")
    conf = args.get("confidence")
    if exp not in EFFECTS or exp == "diagnostic":
        return "falta o es inválido expected_effect (resolves|improves|no_change|worsens)"
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        return "falta o es inválido confidence (0..1)"
    if not 0.0 <= conf <= 1.0:
        return "confidence fuera de rango 0..1"
    clean = {k: v for k, v in args.items() if k not in PRED_PARAMS}
    return Prediction(tool=tool, args=clean, expected=exp, confidence=conf)


def infer_observed(result: dict) -> str:
    obs = result.get("observed_effect")
    if obs in EFFECTS:
        return obs
    return "worsens" if result.get("status") == "error" else "no_change"


class PredictionPlugin(BasePlugin):
    def __init__(self, cfg: SubcortexConfig) -> None:
        super().__init__(name="subcortex_prediction")
        self.cfg = cfg

    async def before_model_callback(self, *, callback_context, llm_request):
        try:
            llm_request.append_instructions([PROTOCOL_INSTRUCTION])
        except Exception:
            log.exception("prediction.before_model")

    async def before_tool_callback(self, *, tool, tool_args, tool_context):
        if not self.cfg.is_action(tool.name):
            return None
        try:
            parsed = parse_prediction(tool.name, tool_args)
            state = tool_context.state
            if isinstance(parsed, str):
                bump(state, "rejected")
                return {"status": "rejected", "reason": parsed,
                        "hint": "Repetí la llamada agregando expected_effect y confidence."}
            pending = dict(state.get(K_PENDING) or {})
            pending[tool_context.function_call_id or tool.name] = parsed.model_dump()
            state[K_PENDING] = pending
        except Exception:
            log.exception("prediction.before_tool")
        return None

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        if not self.cfg.is_action(tool.name):
            return
        try:
            state = tool_context.state
            call_id = tool_context.function_call_id or tool.name
            pending = dict(state.get(K_PENDING) or {})
            pred = pending.pop(call_id, None)
            state[K_PENDING] = pending
            status = (result or {}).get("status")
            if status in BLOCK_STATUSES or pred is None:
                state[K_LAST_ERROR] = None
                return
            observed = infer_observed(result or {})
            err = prediction_error(pred["expected"], observed, pred["confidence"])
            state[K_LAST_ERROR] = {"tool": tool.name, "args": pred["args"], "expected": pred["expected"],
                                   "observed": observed, "confidence": pred["confidence"], "error": err,
                                   "status": status, "call_id": call_id}
            bump(state, "abs_error_sum", abs(err))
            bump(state, "error_count")
        except Exception:
            log.exception("prediction.after_tool")
        return
