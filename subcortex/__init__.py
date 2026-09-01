"""subcortex: capa subcortical para agentes ADK. Punto de entrada: attach()."""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

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

__all__ = ["EpisodicStore", "Scene", "Subcortex", "SubcortexConfig", "attach"]
log = logging.getLogger("subcortex")


@dataclass
class Subcortex:
    config: SubcortexConfig
    store: EpisodicStore
    plugins: list

    def consolidate(self, now: float | None = None, llm=None) -> dict:
        return _consolidate(self.store, now, llm=llm)


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
           coarse_features: Iterable[str] | None = None,
           config: SubcortexConfig | None = None, disable: Iterable[str] = ()) -> Subcortex:
    """Agrega la capa subcortical a `app` sin modificar la lógica del agente.

    `coarse_features`: qué features definen la *clase* de escena para dopamina, gate y hábitos
    (None = todas; con muchas features distintas, nada se repite y nada se aprende).
    """
    cfg = config or SubcortexConfig(
        risk=dict(risk), diagnostic_tools=frozenset(diagnostic_tools),
        scene_fn=scene_fn or default_scene_fn,
        coarse_features=tuple(coarse_features) if coarse_features else None)
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
    # `disable` (ablaciones): nombres en {gate, memory, habit, interoception}. Prediction no se
    # desactiva: sin error de predicción no hay nada que aprender y la ablación sería "vanilla".
    off = set(disable)
    unknown = off - {"gate", "memory", "habit", "interoception"}
    if unknown:
        raise ValueError(f"plugins desconocidos en disable: {sorted(unknown)}")
    plugins = [PredictionPlugin(cfg)]
    if "gate" not in off:
        plugins.append(GatePlugin(cfg, store))
    if "memory" not in off:
        plugins.append(MemoryPlugin(cfg, store))
    if "habit" not in off:
        plugins.append(HabitPlugin(cfg, store))
    if "interoception" not in off:
        plugins.append(InteroceptionPlugin(cfg))
    app.plugins.extend(plugins)
    return Subcortex(config=cfg, store=store, plugins=plugins)
