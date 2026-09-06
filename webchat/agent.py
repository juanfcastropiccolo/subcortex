"""Chat interactivo con subcortex para `adk web`.

Levanta el operador de opsworld con la capa subcortical completa, memoria SQLite persistente
(webchat/subcortex-web.db: sobrevive reinicios — el agente recuerda entre charlas) y, por defecto,
Claude Sonnet 5 vía el CLI local como motor (`SUBCORTEX_WEB_MODEL` lo cambia; acepta
"claude-code[:modelo[:effort]]" o un string de modelo Gemini).

Uso:  uv run adk web .   → elegir "webchat" en el dropdown → pedirle un incidente.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.tools.tool_context import ToolContext

import subcortex
from demo.agent import resolve_model
from opsworld.tools import ALL_TOOLS, registry
from opsworld.world import (
    CAUSES,
    COARSE_FEATURES,
    DIAGNOSTIC_TOOLS,
    HOURS,
    RISK,
    SERVICES,
    Incident,
    World,
    features,
)
from subcortex.gate import K_RECONSIDERED
from subcortex.habit import K_HABIT_TRIED
from subcortex.types import (
    K_ACTED,
    K_DISCOVERED,
    K_FEATURES,
    K_HABIT_HIT,
    K_INTERO,
    K_LAST_ERROR,
    K_PENDING,
    K_TONE,
    K_VETO_LOG,
)

MODEL = os.environ.get("SUBCORTEX_WEB_MODEL", "claude-code:sonnet")
STORE = str(Path(__file__).parent / "subcortex-web.db")

_EPISODE_KEYS = (K_FEATURES, K_DISCOVERED, K_PENDING, K_LAST_ERROR, K_ACTED, K_HABIT_HIT,
                 K_INTERO, K_TONE, K_VETO_LOG, K_HABIT_TRIED, K_RECONSIDERED)


def nuevo_incidente(causa: str, tool_context: ToolContext) -> dict:
    """Abre un incidente nuevo en el simulador para esta sesión (cierra el anterior si había).

    Args:
        causa: una de memory_leak, bad_deploy, traffic_spike, db_saturated, dependency_down,
               false_alarm — o "aleatoria" para que el simulador elija sin decirlo.
    """
    rng = random.Random()
    real = rng.choice(CAUSES) if causa not in CAUSES else causa
    inc = Incident(
        id=rng.randrange(10_000), service=rng.choice(SERVICES), cause=real,
        recent_deploy=(real == "bad_deploy") or (rng.random() < 0.15),
        traffic="spike" if real == "traffic_spike" else ("spike" if rng.random() < 0.1 else "normal"),
        hour_bucket=rng.choice(HOURS))
    world = World(inc)
    registry.register(tool_context.session.id, world)
    state = tool_context.state
    for k in _EPISODE_KEYS:  # arranque limpio del episodio (la memoria SQLite persiste igual)
        state[k] = None
    f = features(inc)
    state[K_FEATURES] = f
    return {"status": "success", "observed_effect": "diagnostic",
            "message": "Incidente abierto. Solo ves lo observable; la causa hay que diagnosticarla.",
            "observables": f}


def estado_incidente(tool_context: ToolContext) -> dict:
    """Muestra el marcador del incidente activo: pasos, score, si sigue abierto y el log de acciones."""
    w = registry.get(tool_context.session.id)
    if w is None:
        return {"status": "success", "observed_effect": "diagnostic",
                "message": "No hay incidente activo: pedime uno con nuevo_incidente."}
    return {"status": "success", "observed_effect": "diagnostic",
            "abierto": not w.done, "resuelto": w.resolved, "pasos": w.steps, "score": w.score,
            "acciones": [f"{e['action']} → {e['observed_effect']}" for e in w.log]}


INSTRUCTION = """Sos el operador de guardia de una plataforma, en modo conversacional: charlás con
una persona que puede pedirte incidentes, preguntarte por qué hiciste algo o por tu memoria.

Cómo trabajar un incidente:
- Si no hay incidente activo y te piden uno, llamá a nuevo_incidente (causa "aleatoria" salvo pedido).
- Diagnóstico gratis: inspect_service, check_deploys. Acciones: restart (costosa), scale (costosa),
  rollback (irreversible), failover_db (irreversible), resolve (cierra; cerrarlo sin resolver es una
  falta grave), escalate_to_human. Máximo 8 llamadas por incidente; estado_incidente muestra el marcador.
- Diagnosticá antes de actuar si no estás seguro; no repitas acciones que no cambiaron nada; si es una
  dependencia externa, escalá.
- Si una acción vuelve "vetoed" o "rejected", leé reason y hint antes de decidir.

En la charla: HABLALE al usuario — saludá, y después de 2 o 3 herramientas frená y contale qué
hiciste, qué viste y qué vas a hacer, antes de seguir. Explicá tus decisiones con naturalidad; si
tenés precedentes o reglas aprendidas en el contexto, citalos ("la última vez que vi este
síntoma..."). Respondé siempre en español."""


def build() -> tuple[App, object]:
    agent = LlmAgent(name="operador_subcortex", model=resolve_model(MODEL, require_action=False),
                     description="Operador de guardia con capa subcortical y memoria persistente",
                     instruction=INSTRUCTION, tools=[nuevo_incidente, estado_incidente, *ALL_TOOLS])
    application = App(name="webchat", root_agent=agent)
    sc = subcortex.attach(application, risk=RISK, diagnostic_tools=DIAGNOSTIC_TOOLS,
                          store_path=STORE, coarse_features=COARSE_FEATURES)
    return application, sc


app, subcortex_layer = build()
root_agent = app.root_agent  # fallback para loaders que buscan root_agent
