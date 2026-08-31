"""subcortex: capa subcortical para agentes ADK. Punto de entrada: attach()."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.tools.function_tool import FunctionTool

from .config import SubcortexConfig, default_scene_fn
from .consolidate import consolidate as _consolidate
from .gate import GatePlugin
from .habit import HabitPlugin
from .interoception import InteroceptionPlugin
from .memory import MemoryPlugin
from .prediction import PredictionPlugin, wrap_action_tool
from .store import EpisodicStore
from .types import Scene

__all__ = ["attach", "Subcortex", "SubcortexConfig", "EpisodicStore", "Scene"]
log = logging.getLogger("subcortex")


@dataclass
class Subcortex:
    config: SubcortexConfig
    store: EpisodicStore
    plugins: list

    def consolidate(self, now: float | None = None) -> dict:
        return _consolidate(self.store, now)


def _wrap_agent_tools(agent: LlmAgent, cfg: SubcortexConfig) -> None:
    new_tools = []
    for t in agent.tools:
        if isinstance(t, FunctionTool) and cfg.is_action(t.name):
            new_tools.append(FunctionTool(func=wrap_action_tool(t.func)))
        elif callable(t) and not isinstance(t, FunctionTool) and cfg.is_action(getattr(t, "__name__", "")):
            new_tools.append(wrap_action_tool(t))
        else:
            new_tools.append(t)
    agent.tools = new_tools


def attach(app: App, *, risk: dict[str, str], diagnostic_tools: Iterable[str] = (),
           store_path: str = ":memory:", scene_fn: Callable[[Any], Scene | None] | None = None,
           config: SubcortexConfig | None = None) -> Subcortex:
    """Agrega la capa subcortical a `app` sin modificar la lógica del agente."""
    cfg = config or SubcortexConfig(risk=dict(risk), diagnostic_tools=frozenset(diagnostic_tools),
                                    scene_fn=scene_fn or default_scene_fn)
    store = EpisodicStore(store_path)
    agent = app.root_agent
    if isinstance(agent, LlmAgent):
        _wrap_agent_tools(agent, cfg)
    else:
        log.warning("root_agent no es LlmAgent; no se envuelven tools")
    # Orden importa: ADK ejecuta en orden y corta en el primer no-None.
    #   before_model: Prediction(protocolo) → Gate(-) → Memory(precedentes) → Habit(bypass) → Intero(estado)
    #   before_tool:  Prediction(valida) → Gate(veto)
    #   after_tool:   Prediction(error) → Gate(-) → Memory(escribe, dopamina) → Habit(compila) → Intero(tono)
    plugins = [PredictionPlugin(cfg), GatePlugin(cfg, store), MemoryPlugin(cfg, store),
               HabitPlugin(cfg, store), InteroceptionPlugin(cfg)]
    app.plugins.extend(plugins)
    return Subcortex(config=cfg, store=store, plugins=plugins)
