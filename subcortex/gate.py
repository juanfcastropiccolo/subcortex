"""Ganglios basales + cingulado: veto por defecto, desinhibir una sola acción."""
from __future__ import annotations

import logging

from google.adk.plugins.base_plugin import BasePlugin

from .config import SubcortexConfig
from .metrics import bump
from .store import EpisodicStore
from .types import K_INTERO, K_PENDING, K_TONE

log = logging.getLogger("subcortex")


def gate_decision(cfg: SubcortexConfig, tool: str, confidence: float, dopamine: float,
                  tone: float, consecutive_blocks: int) -> tuple[bool, float, str]:
    """Devuelve (autorizada, valor, motivo)."""
    if tool in cfg.always_allowed:
        return True, 1.0, "siempre permitida"
    risk = cfg.risk_of(tool)
    value = round(confidence * dopamine * tone - cfg.cost.get(risk, 0.0), 4)
    if consecutive_blocks >= cfg.max_consecutive_blocks:
        return False, value, "demasiadas acciones bloqueadas seguidas: solo se permite escalar o cerrar"
    if risk == "irreversible":
        if confidence < cfg.hyperdirect_confidence:
            return False, value, (f"acción irreversible con confianza {confidence:.2f} < "
                                  f"{cfg.hyperdirect_confidence}: diagnosticá más o escalá")
        if tone < cfg.hyperdirect_tone:
            return False, value, f"acción irreversible con tono bajo ({tone:.2f}): frená y reconsiderá"
    if value < cfg.gate_threshold:
        return False, value, (f"valor esperado {value:.2f} < umbral {cfg.gate_threshold}: "
                              "poca confianza, historial pobre o costo alto")
    return True, value, "autorizada"


class GatePlugin(BasePlugin):
    def __init__(self, cfg: SubcortexConfig, store: EpisodicStore) -> None:
        super().__init__(name="subcortex_gate")
        self.cfg = cfg
        self.store = store

    def _scene_key(self, state) -> str:
        scene = self.cfg.scene_fn(state)
        return scene.key if scene else "-"

    def _value(self, state, tool: str, confidence: float) -> float:
        tone = float(state.get(K_TONE, 1.0))
        dop = self.store.dopamine(self._scene_key(state), tool, self.cfg.dopamine_prior)
        return gate_decision(self.cfg, tool, confidence, dop, tone, 0)[1]

    async def after_model_callback(self, *, callback_context, llm_response):
        """Cuenta llamadas reales al LLM y aplica winner-take-all sobre acciones paralelas."""
        try:
            state = callback_context.state
            if llm_response.partial:
                return None
            bump(state, "llm_calls")
            um = llm_response.usage_metadata
            if um and um.total_token_count:
                bump(state, "tokens", um.total_token_count)
            parts = (list(llm_response.content.parts)
                     if llm_response.content and llm_response.content.parts else [])
            action_idx = [i for i, p in enumerate(parts)
                          if p.function_call and self.cfg.is_action(p.function_call.name)]
            if len(action_idx) <= 1:
                return None

            def score(i: int) -> float:
                fc = parts[i].function_call
                try:
                    conf = float((fc.args or {}).get("confidence", 0.0))
                except (TypeError, ValueError):
                    conf = 0.0
                return self._value(state, fc.name, conf)

            winner = max(action_idx, key=score)
            losers = set(action_idx) - {winner}
            bump(state, "vetoes", len(losers))
            llm_response.content.parts = [p for i, p in enumerate(parts) if i not in losers]
            log.info("winner-take-all: conservo %s, veto %d acciones paralelas",
                     parts[winner].function_call.name, len(losers))
            return llm_response
        except Exception:
            log.exception("gate.after_model")
            return None

    async def before_tool_callback(self, *, tool, tool_args, tool_context):
        if not self.cfg.is_action(tool.name):
            return None
        try:
            state = tool_context.state
            pending = (state.get(K_PENDING) or {}).get(tool_context.function_call_id or tool.name)
            confidence = float(pending["confidence"]) if pending else 0.0
            tone = float(state.get(K_TONE, 1.0))
            blocks = int((state.get(K_INTERO) or {}).get("blocks", 0))
            dop = self.store.dopamine(self._scene_key(state), tool.name, self.cfg.dopamine_prior)
            ok, value, why = gate_decision(self.cfg, tool.name, confidence, dop, tone, blocks)
            if ok:
                return None
            bump(state, "vetoes")
            if self.cfg.risk_of(tool.name) == "irreversible":
                bump(state, "vetoes_irreversible")
            log.info("veto %s: %s", tool.name, why)
            return {"status": "vetoed", "tool": tool.name, "value": value, "reason": why,
                    "hint": "Reconsiderá: pedí más diagnóstico, elegí otra acción o escalá."}
        except Exception:
            log.exception("gate.before_tool")
            return None
