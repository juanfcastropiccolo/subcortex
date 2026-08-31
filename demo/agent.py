"""Agente operador. `root_agent` lleva subcortex para `adk web demo`."""
from __future__ import annotations

import os
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.apps.app import App

import subcortex
from opsworld.tools import ALL_TOOLS
from opsworld.world import DIAGNOSTIC_TOOLS, RISK

try:  # carga demo/.env o .env de la raíz si existen (adk web lo hace solo; run_ab no)
    from dotenv import load_dotenv

    for _env in (Path(__file__).parent / ".env", Path(__file__).parent.parent / ".env"):
        if _env.exists():
            load_dotenv(_env)
except ImportError:  # pragma: no cover
    pass

MODEL = os.environ.get("SUBCORTEX_MODEL", "gemini-3-flash-preview")
INSTRUCTION = (Path(__file__).parent / "instruction.md").read_text()


def build_agent(model: str | object = MODEL) -> LlmAgent:
    return LlmAgent(name="ops_operator", model=model, description="Operador de guardia",
                    instruction=INSTRUCTION, tools=list(ALL_TOOLS))


def build_app(with_subcortex: bool, store_path: str = ":memory:", model: str | object = MODEL):
    app = App(name="opsworld", root_agent=build_agent(model))
    sc = (subcortex.attach(app, risk=RISK, diagnostic_tools=DIAGNOSTIC_TOOLS, store_path=store_path)
          if with_subcortex else None)
    return app, sc


root_agent = build_agent()
