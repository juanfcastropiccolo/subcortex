"""Hipocampo / amígdala / habénula: recuerdo por escena, escritura solo ante sorpresa."""
from __future__ import annotations

import logging

from google.adk.plugins.base_plugin import BasePlugin

from .config import SubcortexConfig
from .metrics import bump
from .store import EpisodicStore
from .types import K_LAST_ERROR, K_TONE, SUCCESS_EFFECTS, Episode, Rule

log = logging.getLogger("subcortex")
BLOCK_STATUSES = {"vetoed", "rejected"}


def render_precedents(episodes: list[Episode]) -> str:
    if not episodes:
        return ""
    lines = ["## Precedentes (experiencias propias, las peores primero)"]
    for e in episodes:
        tag = "[FRACASO]" if e.habenula else "[MEJOR DE LO ESPERADO]"
        feats = ", ".join(f"{k}={v}" for k, v in e.features.items())
        args = ", ".join(f"{k}={v}" for k, v in e.args.items())
        lines.append(f"- {tag} incidente ({feats}): {e.tool}({args}) esperaba {e.expected}, "
                     f"resultó {e.observed}.")
    return "\n".join(lines)


def render_rules(rules: list[Rule]) -> str:
    if not rules:
        return ""
    return "## Reglas aprendidas\n" + "\n".join(f"- {r.text}" for r in rules)


class MemoryPlugin(BasePlugin):
    def __init__(self, cfg: SubcortexConfig, store: EpisodicStore) -> None:
        super().__init__(name="subcortex_memory")
        self.cfg = cfg
        self.store = store

    async def before_model_callback(self, *, callback_context, llm_request):
        try:
            state = callback_context.state
            scene = self.cfg.scene_fn(state)
            if scene is None:
                return
            tone = float(state.get(K_TONE, 1.0))
            k = round(2 + 4 * tone)
            blocks = [render_precedents(self.store.recall(scene, k)),
                      render_rules(self.store.rules_for(scene))]
            blocks = [b for b in blocks if b]
            if blocks:
                llm_request.append_instructions(blocks)
        except Exception:
            log.exception("memory.before_model")
        return

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        if not self.cfg.is_action(tool.name):
            return
        try:
            state = tool_context.state
            if (result or {}).get("status") in BLOCK_STATUSES:
                return
            le = state.get(K_LAST_ERROR)
            if not le or le.get("tool") != tool.name:
                return
            scene = self.cfg.scene_fn(state)
            if scene is None:
                return
            observed = le["observed"]
            self.store.record_outcome(scene.key, tool.name, observed in SUCCESS_EFFECTS)
            err = float(le["error"])
            surprised = abs(err) >= self.cfg.surprise_threshold or le.get("status") == "error"
            if not surprised:
                return
            self.store.write(Episode(
                scene_key=scene.key, features=scene.features, tool=tool.name, args=le["args"],
                expected=le["expected"], observed=observed, prediction_error=err, valence=err,
                habenula=err < 0 or le.get("status") == "error", strength=max(abs(err), 0.25)))
            bump(state, "episodes_written")
        except Exception:
            log.exception("memory.after_tool")
        return
