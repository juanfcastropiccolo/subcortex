"""Ganglios basales + cingulado: veto por defecto, desinhibir una sola acción."""
from __future__ import annotations

import logging

from google.adk.plugins.base_plugin import BasePlugin

from .config import SubcortexConfig
from .metrics import bump
from .store import EpisodicStore
from .types import K_INTERO, K_PENDING, K_TONE, K_VETO_LOG

__all__ = ["GatePlugin", "K_RECONSIDERED", "effective_confidence", "gate_decision", "gate_decision_ex"]

K_RECONSIDERED = "subcortex.reconsidered"

log = logging.getLogger("subcortex")


def gate_decision_ex(cfg: SubcortexConfig, tool: str, confidence: float, dopamine: float,
                     tone: float, consecutive_blocks: int, trusted: bool = False,
                     ) -> tuple[bool, float, str, str]:
    """Como `gate_decision`, más el código causal del bloqueo (para no contar peras con manzanas
    al reportar "vetos"): value | hyperdirect | blockstreak | allowed."""
    if tool in cfg.always_allowed:
        return True, 1.0, "siempre permitida", "allowed"
    risk = cfg.risk_of(tool)
    # El tono NO multiplica el valor: ya frena por la vía hiperdirecta. Multiplicarlo hacía
    # imposible autorizar una irreversible a mitad de episodio (corrida 1).
    value = round(confidence * dopamine - cfg.cost.get(risk, 0.0), 4)
    if consecutive_blocks >= cfg.max_consecutive_blocks:
        return (False, value, "demasiadas acciones bloqueadas seguidas: solo se permite escalar o cerrar",
                "blockstreak")
    if risk == "irreversible" and not trusted:
        if confidence < cfg.hyperdirect_confidence:
            return False, value, (f"acción irreversible con confianza {confidence:.2f} < "
                                  f"{cfg.hyperdirect_confidence}"), "hyperdirect"
        if tone < cfg.hyperdirect_tone:
            return False, value, f"acción irreversible con tono bajo ({tone:.2f})", "hyperdirect"
    if value < cfg.gate_threshold:
        return (False, value, f"valor esperado {value:.2f} < umbral {cfg.gate_threshold}: "
                "poca confianza, historial pobre o costo alto", "value")
    return True, value, "autorizada", "allowed"


def gate_decision(cfg: SubcortexConfig, tool: str, confidence: float, dopamine: float,
                  tone: float, consecutive_blocks: int, trusted: bool = False,
                  ) -> tuple[bool, float, str]:
    """Devuelve (autorizada, valor, motivo). `trusted` = la acción ya resolvió esta clase de
    escena varias veces sin fallar: la confianza ganada exime del freno hiperdirecto."""
    ok, value, why, _code = gate_decision_ex(cfg, tool, confidence, dopamine, tone,
                                             consecutive_blocks, trusted)
    return ok, value, why


def effective_confidence(declared: float, dopamine: float, n_history: int) -> float:
    """Confianza que pesa el gate: la del modelo al principio, la del historial a medida que hay.

    Los LLM están mal calibrados; la dopamina de (clase de escena, acción) es la tasa de éxito
    observada. Peso del historial = n/(n+2): con 0 observaciones manda el modelo, con 2 van a la
    par, con 6 el historial pesa 3 a 1. El modelo sigue desempatando cuando no hay datos."""
    w = n_history / (n_history + 2.0)
    return round((1 - w) * declared + w * dopamine, 4)


class GatePlugin(BasePlugin):
    def __init__(self, cfg: SubcortexConfig, store: EpisodicStore) -> None:
        super().__init__(name="subcortex_gate")
        self.cfg = cfg
        self.store = store

    def _scene_key(self, state) -> str:
        scene = self.cfg.scene_of(state)
        return scene.coarse_key if scene else "-"

    def _value(self, state, tool: str, confidence: float) -> float:
        tone = float(state.get(K_TONE, 1.0))
        dop = self.store.dopamine(self._scene_key(state), tool, self.cfg.dopamine_prior)
        return gate_decision(self.cfg, tool, confidence, dop, tone, 0)[1]

    def _hint(self, scene_key: str, tool: str, why: str) -> str:
        """Veto con alternativa: qué funcionó antes en esta clase de escena, y cómo destrabarse."""
        best = [(t, s, f) for t, s, f in self.store.best_tools(scene_key) if t != tool]
        parts = []
        if best:
            parts.append("En esta clase de escena ya funcionó: " + ", ".join(
                f"{t} ({s} éxitos, {f} fallos)" for t, s, f in best[:3]) + ".")
        if "confianza" in why:
            parts.append(f"Si tu diagnóstico respalda {tool}, repetí la llamada con confidence ≥ "
                         f"{self.cfg.hyperdirect_confidence} y justificalo; si no, diagnosticá más o escalá.")
        else:
            parts.append("Reconsiderá: pedí más diagnóstico, elegí otra acción o escalá.")
        return " ".join(parts)

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
            # Arbitraje, NO veto: elegir una de varias propuestas paralelas no impide ninguna
            # acción que el agente fuera a ejecutar igual. Se cuenta aparte (auditoría 2026-09-06).
            bump(state, "arbitration_dropped", len(losers))
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
            declared = float(pending["confidence"]) if pending else 0.0
            tone = float(state.get(K_TONE, 1.0))
            blocks = int((state.get(K_INTERO) or {}).get("blocks", 0))
            scene_key = self._scene_key(state)
            dop = self.store.dopamine(scene_key, tool.name, self.cfg.dopamine_prior)
            s, f = self.store.outcome_counts(scene_key, tool.name)
            trusted = s >= self.cfg.trust_min_successes and f == 0
            confidence = (effective_confidence(declared, dop, s + f) if self.cfg.confidence_from_history
                          else declared)
            ok, value, why, code = gate_decision_ex(self.cfg, tool.name, confidence, dop, tone,
                                                    blocks, trusted)
            if ok and self.cfg.cingulate_reconsider and tool.name not in self.cfg.always_allowed:
                # Cingulado (paso 9): conflicto = valor cerca del umbral. Compra un pase más de
                # deliberación, una sola vez por episodio y por acción, antes de desinhibir.
                near = abs(value - self.cfg.gate_threshold) < self.cfg.reconsider_margin
                done = set(state.get(K_RECONSIDERED) or [])
                if near and tool.name not in done:
                    state[K_RECONSIDERED] = sorted(done | {tool.name})
                    bump(state, "reconsiders")
                    return {"status": "reconsider", "tool": tool.name, "value": value,
                            "reason": f"valor {value:.2f} al borde del umbral {self.cfg.gate_threshold}: "
                                      "conflicto entre confianza, historial y costo",
                            "hint": (f"Pensalo un paso más. Si seguís convencido, repetí {tool.name} con la misma "
                                     "predicción y se ejecuta; si no, cambiá de acción.")}
            if ok:
                return None
            bump(state, "vetoes")
            bump(state, f"vetoes_{code}")  # value | hyperdirect | blockstreak
            if self.cfg.risk_of(tool.name) == "irreversible":
                bump(state, "vetoes_irreversible")
            veto_log = list(state.get(K_VETO_LOG) or [])
            veto_log.append({"tool": tool.name, "confidence": confidence, "declared": declared, "tone": tone,
                             "dopamine": round(dop, 3), "reason": why, "code": code})
            state[K_VETO_LOG] = veto_log
            log.info("veto %s: %s", tool.name, why)
            return {"status": "vetoed", "tool": tool.name, "value": value, "reason": why,
                    "hint": self._hint(scene_key, tool.name, why)}
        except Exception:
            log.exception("gate.before_tool")
            return None
