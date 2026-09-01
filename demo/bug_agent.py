"""Agente mantenedor para bugworld, con y sin subcortex."""
from __future__ import annotations

from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.apps.app import App

import subcortex
from bugworld.tools import ALL_TOOLS
from bugworld.world import ALWAYS_ALLOWED, COARSE_FEATURES, DIAGNOSTIC_TOOLS, RISK
from demo.agent import MODEL
from subcortex.config import SubcortexConfig, default_scene_fn

INSTRUCTION = (Path(__file__).parent / "bug_instruction.md").read_text()


def build_agent(model: str | object = MODEL) -> LlmAgent:
    return LlmAgent(name="maintainer", model=model, description="Mantenedor que arregla bugs",
                    instruction=INSTRUCTION, tools=list(ALL_TOOLS))


def build_app(with_subcortex: bool, store_path: str = ":memory:", model: str | object = MODEL):
    app = App(name="bugworld", root_agent=build_agent(model))
    sc = None
    if with_subcortex:
        cfg = SubcortexConfig(risk=dict(RISK), diagnostic_tools=DIAGNOSTIC_TOOLS,
                              scene_fn=default_scene_fn, coarse_features=COARSE_FEATURES,
                              always_allowed=ALWAYS_ALLOWED, step_budget=10)
        sc = subcortex.attach(app, risk=RISK, store_path=store_path, config=cfg)
    return app, sc
