"""Protocolo confirmatorio congelado: brazos de control y configuración inmutable.

Respuesta a la auditoría del 2026-09-06. Dos problemas que este módulo resuelve:

1. **Confound de protocolo.** La comparación "vanilla vs subcortex" mezclaba dos cosas: la
   arquitectura (memoria, gate, hábitos) y el cambio del canal hacia el modelo (declarar
   `expected_effect` y `confidence`, más instrucciones inyectadas). El brazo `protocol` aísla
   eso: contrato de predicción SIN memoria, gate, hábitos ni interocepción.

2. **Reuso adaptativo del banco.** Las corridas previas informaron decisiones de diseño, así que
   son datos de desarrollo. Acá la configuración queda congelada y hasheada ANTES de ver las
   tareas de evaluación; cambiarla cambia el hash y obliga a declarar una corrida nueva.

Los números de FROZEN son los de la última iteración de desarrollo. No se tocan más: cualquier
cambio invalida la corrida confirmatoria y hay que reportarlo como una arquitectura distinta.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any

from google.adk.apps.app import App

import subcortex
from subcortex.config import SubcortexConfig

# --- configuración congelada -------------------------------------------------------------
# Estos valores NO se ajustan más. Si se ajustan, cambia CONFIG_SHA y la corrida deja de ser
# confirmatoria para la arquitectura anterior.
FROZEN: dict[str, Any] = {
    "step_budget": 8,
    "surprise_threshold": 0.25,
    "gate_threshold": 0.10,
    "cost": {"free": 0.0, "costly": 0.10, "irreversible": 0.20},
    "dopamine_prior": 0.6,
    "hyperdirect_confidence": 0.6,
    "hyperdirect_tone": 0.4,
    "trust_min_successes": 2,
    "max_consecutive_blocks": 3,
    "habit_min_successes": 3,
    "habit_min_strength": 0.8,
    "recall_min_overlap": 2,
    "recall_max": 4,
    "write_on_success": True,
    "confidence_from_history": False,
    "stall_fraction": 0.6,
    "cingulate_reconsider": False,
    "reconsider_margin": 0.08,
}

# --- endpoint primario congelado ---------------------------------------------------------
# El score del mundo ya cobra el daño (−40 por empeoramiento) y el paso (−5), así que la utilidad
# compuesta NO vuelve a restar daño: sería doble conteo. Lo que el score no cobra es el cómputo.
# λ_c = 2.0 puntos de score por llamada al modelo, declarado antes de correr.
LAMBDA_CALLS_PRIMARY = 2.0
LAMBDA_CALLS_SENSITIVITY = (0.0, 2.0, 5.0, 10.0)
PRIMARY_ENDPOINT = "utility_mean"   # media por trayectoria de score − λ_c · llm_calls
REPLICATE = "trajectory"            # la réplica es la trayectoria (semilla), no el episodio


@dataclass(frozen=True)
class Arm:
    name: str
    attach: bool               # False = agente desnudo, sin capa
    disable: tuple[str, ...]   # plugins apagados dentro de la capa
    neutral_telemetry: bool
    purpose: str


# Los 7 brazos del protocolo. `prediction` está siempre activa cuando hay capa: es el contrato
# que define el canal, y el brazo `protocol` existe justamente para medir su efecto por separado.
ARMS: dict[str, Arm] = {
    "vanilla": Arm(
        "vanilla", False, (), False,
        "Baseline actual: agente sin capa, sin contrato de predicción."),
    "protocol": Arm(
        "protocol", True, ("gate", "memory", "habit", "interoception"), False,
        "CONTROL CRÍTICO: solo el contrato de predicción. Mide cuánto del efecto viene de "
        "obligar al modelo a declarar consecuencia esperada y confianza."),
    "retrieval": Arm(
        "retrieval", True, ("gate", "habit", "interoception"), False,
        "Protocolo + memoria episódica. Es la explicación alternativa simple del auditor: "
        "recuperación de experiencia previa, sin gate ni hábitos."),
    "cache": Arm(
        "cache", True, ("gate", "memory", "interoception"), False,
        "Protocolo + hábitos. Mide la memoización procedural pura (bypass del LLM)."),
    "gate": Arm(
        "gate", True, ("memory", "habit", "interoception"), False,
        "Protocolo + gate. Efecto de seguridad/arbitraje aislado."),
    "telemetry": Arm(
        "telemetry", True, ("gate", "memory", "habit"), True,
        "Protocolo + presupuesto y errores en lenguaje neutro. Si iguala a la capa completa, "
        "la 'interocepción' era información, no encuadre."),
    "full": Arm(
        "full", True, (), False,
        "Arquitectura propuesta completa."),
}

STAGES = {
    "A": ("vanilla", "protocol", "full"),        # el confound crítico
    "B": ("retrieval", "cache"),                 # las alternativas simples
    "C": ("gate", "telemetry"),                  # componentes restantes
}


def config_for(arm: Arm, *, risk: dict[str, str], diagnostic_tools, coarse_features,
               scene_fn=None) -> SubcortexConfig:
    base = SubcortexConfig(risk=dict(risk), diagnostic_tools=frozenset(diagnostic_tools),
                           coarse_features=tuple(coarse_features) if coarse_features else None,
                           **FROZEN)
    if scene_fn is not None:
        base = replace(base, scene_fn=scene_fn)
    return replace(base, neutral_telemetry=arm.neutral_telemetry)


def build_arm_app(arm: Arm, agent, *, risk, diagnostic_tools, coarse_features,
                  store_path: str = ":memory:", app_name: str = "confirm", scene_fn=None):
    """Devuelve (app, sc) para un brazo. `sc` es None en vanilla."""
    app = App(name=app_name, root_agent=agent)
    if not arm.attach:
        return app, None
    cfg = config_for(arm, risk=risk, diagnostic_tools=diagnostic_tools,
                     coarse_features=coarse_features, scene_fn=scene_fn)
    sc = subcortex.attach(app, risk=risk, store_path=store_path, config=cfg,
                          disable=arm.disable)
    return app, sc


def protocol_spec() -> dict:
    """Lo que queda congelado y hasheado: config, brazos, endpoint y regla de réplica."""
    return {
        "frozen_config": FROZEN,
        "arms": {a.name: {"attach": a.attach, "disable": sorted(a.disable),
                          "neutral_telemetry": a.neutral_telemetry} for a in ARMS.values()},
        "primary_endpoint": PRIMARY_ENDPOINT,
        "lambda_calls": LAMBDA_CALLS_PRIMARY,
        "replicate": REPLICATE,
    }


def config_sha() -> str:
    return hashlib.sha256(
        json.dumps(protocol_spec(), sort_keys=True).encode()).hexdigest()[:16]


CONFIG_SHA = config_sha()

if __name__ == "__main__":
    print(json.dumps(protocol_spec(), indent=2, ensure_ascii=False))
    print("CONFIG_SHA:", CONFIG_SHA)
