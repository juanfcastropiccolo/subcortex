"""Ínsula / hipotálamo / formación reticular: el agente sabe cómo está."""
from __future__ import annotations

import logging
import time
from typing import Any

from google.adk.plugins.base_plugin import BasePlugin

from .config import SubcortexConfig
from .metrics import bump
from .types import K_INTERO, K_TONE

log = logging.getLogger("subcortex")
BLOCK_STATUSES = {"vetoed", "rejected", "invalid", "reconsider"}


def _intero(state: Any) -> dict:
    base = {"steps": 0, "failures": 0, "blocks": 0, "evaluated": 0, "invalid_streak": 0,
            "last_status": None, "started_at": time.time()}
    return {**base, **(state.get(K_INTERO) or {})}


def is_stalled(intero: dict, cfg: SubcortexConfig) -> bool:
    """No progreso: buena parte del presupuesto consumida sin ningún resultado evaluado."""
    return (intero.get("evaluated", 0) == 0
            and intero.get("steps", 0) >= cfg.stall_fraction * cfg.step_budget)


def compute_tone(intero: dict, cfg: SubcortexConfig) -> float:
    """Tono = presupuesto restante y fallos reales. Los vetos NO bajan el tono: si lo
    hicieran, cada veto haría más probable el siguiente (bucle observado en la corrida 1)."""
    tone = (1.0 - 0.15 * intero.get("failures", 0)
            - 0.5 * intero.get("steps", 0) / max(cfg.step_budget, 1))
    return round(max(0.05, min(1.0, tone)), 4)


def render_state_neutral(intero: dict, tone: float, cfg: SubcortexConfig) -> str:
    """Control de telemetría neutra: los mismos números, sin encuadre corporal ni recomendación.

    Si el agente rinde igual con este texto que con el interoceptivo, lo que aportaba la ínsula
    era información de presupuesto, no un estado interno traducido a lenguaje (auditoría 2026-09-06).
    """
    return "\n".join([
        "## Telemetría",
        f"- Llamadas a herramientas usadas: {intero.get('steps', 0)} de {cfg.step_budget}.",
        f"- Errores consecutivos: {intero.get('failures', 0)}.",
        f"- Acciones bloqueadas consecutivas: {intero.get('blocks', 0)}.",
        f"- Llamadas inválidas consecutivas: {intero.get('invalid_streak', 0)}.",
        f"- Acciones con resultado evaluado: {intero.get('evaluated', 0)}.",
    ])


def render_state(intero: dict, tone: float, cfg: SubcortexConfig) -> str:
    lines = ["## Estado interno"]
    used = int(100 * intero.get("steps", 0) / max(cfg.step_budget, 1))
    lines.append(f"- Presupuesto de acciones consumido: {used} %.")
    f = intero.get("failures", 0)
    if f:
        lines.append(f"- Llevás {f} fallos seguidos: bajá la confianza y diagnosticá antes de actuar.")
    b = intero.get("blocks", 0)
    if b:
        lines.append(f"- Tus últimas {b} acciones fueron bloqueadas: leé el motivo y la alternativa "
                     "del bloqueo antes de insistir.")
    if intero.get("last_status") == "vetoed":
        lines.append("- La última acción fue vetada por riesgo.")
    inv = intero.get("invalid_streak", 0)
    if inv >= 2:
        lines.append(f"- Tus últimas {inv} llamadas fueron inválidas (no tocaron nada): revisá los "
                     "argumentos exactos antes de repetir; una acción inválida cuesta un paso y no enseña nada.")
    if is_stalled(intero, cfg):
        lines.append(f"- SIN PROGRESO: consumiste el {used} % del presupuesto sin ningún resultado evaluado. "
                     "Si no tenés un cambio concreto y justificado, escalá ahora (escalate_to_human) en vez "
                     "de seguir probando; el problema puede no estar donde lo buscás.")
    if tone < 0.4:
        lines.append("- Tono bajo: el sistema no está en condiciones de acciones irreversibles.")
    elif tone > 0.8:
        lines.append("- Tono alto: podés actuar con decisión si el diagnóstico es claro.")
    return "\n".join(lines)


class InteroceptionPlugin(BasePlugin):
    def __init__(self, cfg: SubcortexConfig) -> None:
        super().__init__(name="subcortex_interoception")
        self.cfg = cfg

    async def before_model_callback(self, *, callback_context, llm_request):
        try:
            state = callback_context.state
            intero = _intero(state)
            tone = float(state.get(K_TONE, compute_tone(intero, self.cfg)))
            render = render_state_neutral if self.cfg.neutral_telemetry else render_state
            llm_request.append_instructions([render(intero, tone, self.cfg)])
        except Exception:  # degradación a vanilla
            log.exception("interoception.before_model")

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        try:
            state = tool_context.state
            intero = _intero(state)
            status = (result or {}).get("status")
            if status in BLOCK_STATUSES:
                intero["blocks"] += 1
                if status == "invalid":
                    intero["invalid_streak"] += 1
                    intero["steps"] += 1  # el mundo suele cobrar la llamada aunque no la ejecute
                    bump(state, "steps")
            else:
                intero["blocks"] = 0
                intero["invalid_streak"] = 0
                intero["steps"] += 1
                bump(state, "steps")
                if self.cfg.is_action(tool.name):
                    intero["evaluated"] += 1
                if status == "error":
                    intero["failures"] += 1
                else:
                    intero["failures"] = 0
            if is_stalled(intero, self.cfg) and not intero.get("stalled_flagged"):
                intero["stalled_flagged"] = True
                bump(state, "stalled")
            intero["last_status"] = status
            state[K_INTERO] = intero
            state[K_TONE] = compute_tone(intero, self.cfg)
        except Exception:
            log.exception("interoception.after_tool")

    async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error):
        # Convertimos la excepción en resultado para que el loop siga y se aprenda de ella.
        log.warning("tool %s falló: %s", tool.name, error)
        return {"status": "error", "observed_effect": "worsens", "message": str(error)}
