# subcortex POC — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Librería `subcortex` que agrega a cualquier `LlmAgent` de ADK una capa subcortical (interocepción, predicción, gate, memoria episódica, hábitos, consolidación) sin modificar el agente, más un simulador `opsworld` y una demo A/B que mide la diferencia.

**Architecture:** Cinco `BasePlugin` de ADK registrados en `App.plugins` en el orden `[Prediction, Gate, Memory, Habit, Interoception]` (ADK ejecuta plugins en orden y corta en el primer valor no-`None`). Las tools de acción se envuelven para exigir `expected_effect` y `confidence`. Un `EpisodicStore` SQLite persiste episodios, dopamina, hábitos y reglas. El loop de tool-calls de ADK es el bucle córtico-estriatal: un veto devuelve un resultado de tool y el LLM re-itera.

**Tech Stack:** Python 3.12 (uv), google-adk 2.8, pydantic 2, sqlite3 stdlib, pytest + pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-08-30-subcortex-design.md`

## Global Constraints

- Modelo para la demo: `gemini-3-flash-preview` vía `GOOGLE_API_KEY`. Nunca cambiarlo.
- Todo se corre con `uv run ...`. Tests: `uv run pytest`.
- Ningún test unitario ni de integración llama a un LLM real ni a la red.
- Los plugins atrapan sus excepciones, loguean con `logging.getLogger("subcortex")` y devuelven `None` (degradación a vanilla).
- Estado por sesión bajo claves `subcortex.*` (no `temp:`; se necesita leerlo tras el run para métricas). Nunca mutar dicts anidados in-place en `state`: leer, copiar, reasignar.
- El agente del usuario no se modifica salvo el reemplazo de sus tools de acción por versiones envueltas.
- Hechos verificados de ADK 2.8 en `.venv`: `BasePlugin` hooks son keyword-only (`before_model_callback(self, *, callback_context, llm_request)`, `after_model_callback(self, *, callback_context, llm_response)`, `before_tool_callback(self, *, tool, tool_args, tool_context)`, `after_tool_callback(self, *, tool, tool_args, tool_context, result)`, `on_tool_error_callback(self, *, tool, tool_args, tool_context, error)`); `ToolContext` y `CallbackContext` son alias de `google.adk.agents.context.Context` con `.state`, `.session`, `.function_call_id`, `.invocation_id`; si `before_tool` devuelve un dict, la tool no se ejecuta pero `after_tool` **sí** corre con ese dict como `result`; `LlmRequest.append_instructions(list[str])`; `LlmResponse.get_function_calls()`; `FunctionTool` usa `inspect.signature(func)` (respeta `__signature__`); `Runner(app=App, session_service=...)`, `runner.run_async(user_id=, session_id=, new_message=)`.

---

## File Structure

```
subcortex/__init__.py      attach(), Subcortex handle, re-exports
subcortex/types.py         Effect, RiskClass, Scene, Prediction, Episode, Habit, Rule, prediction_error(), claves de estado
subcortex/config.py        SubcortexConfig (dataclass) con todos los umbrales
subcortex/store.py         EpisodicStore (SQLite): episodes, dopamine, habits, rules
subcortex/interoception.py InteroceptionPlugin + compute_tone() + render_state()
subcortex/prediction.py    wrap_action_tool() + PredictionPlugin
subcortex/gate.py          gate_decision() + GatePlugin (before_tool veto, after_model winner-take-all, métricas LLM)
subcortex/memory.py        MemoryPlugin (recall por escena, write-on-surprise, dopamina) + render_precedents()
subcortex/habit.py         HabitPlugin (compilación, bypass, des-habituación)
subcortex/consolidate.py   consolidate()
subcortex/metrics.py       bump()/get_metrics() helpers de estado
opsworld/world.py          Incident, EFFECTS, World, generate_incidents(), features()
opsworld/tools.py          WorldRegistry + tools ADK (acción y diagnóstico)
demo/instruction.md        instrucción del operador (idéntica en ambas variantes)
demo/agent.py              build_agent(), build_app(with_subcortex, store_path), root_agent
demo/run_ab.py             corrida A/B, tabla y results.json
tests/unit/*.py            un archivo por módulo
tests/integration/fake_llm.py  ScriptedLlm(BaseLlm)
tests/integration/test_loop.py runner real de ADK con ScriptedLlm
README.md
```

---

### Task 1: Tipos, config, métricas y error de predicción

**Files:**
- Create: `subcortex/types.py`, `subcortex/config.py`, `subcortex/metrics.py`
- Test: `tests/unit/test_types.py`

**Interfaces:**
- Produces: `Effect`, `EFFECT_RANK`, `RiskClass`, `Scene.from_features(dict)->Scene`, `Prediction`, `Episode`, `Habit`, `Rule`, `prediction_error(expected, observed, confidence)->float`, claves `K_*`, `SubcortexConfig`, `bump(state, key, n=1)`, `get_metrics(state)->dict`, `set_metric(state, key, value)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_types.py
import pytest
from subcortex.types import Scene, prediction_error


def test_scene_key_is_stable_and_order_independent():
    a = Scene.from_features({"service": "api", "symptom": "oom"})
    b = Scene.from_features({"symptom": "oom", "service": "api"})
    assert a.key == b.key
    assert len(a.key) == 12


def test_prediction_error_signed_and_scaled_by_confidence():
    assert prediction_error("resolves", "resolves", 0.9) == 0.0
    assert prediction_error("resolves", "worsens", 1.0) == -1.0      # -3/2 clamp
    assert prediction_error("no_change", "improves", 1.0) == 0.5     # +1/2
    assert prediction_error("no_change", "improves", 0.5) == 0.25
    assert prediction_error("diagnostic", "no_change", 1.0) == 0.0
    assert prediction_error("resolves", "improves", 0.8) == pytest.approx(-0.4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'subcortex.types'`

- [ ] **Step 3: Write minimal implementation**

```python
# subcortex/types.py
"""Tipos compartidos de subcortex. Cada uno mapea a una pieza del ensayo."""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

Effect = Literal["resolves", "improves", "no_change", "worsens", "diagnostic"]
EFFECTS: tuple[str, ...] = ("resolves", "improves", "no_change", "worsens", "diagnostic")
EFFECT_RANK: dict[str, int] = {"worsens": -1, "no_change": 0, "improves": 1, "resolves": 2}
SUCCESS_EFFECTS = {"improves", "resolves"}

RiskClass = Literal["free", "costly", "irreversible"]

# Claves de estado de sesión (persisten dentro de la sesión; legibles tras el run).
K_TONE = "subcortex.tone"
K_INTERO = "subcortex.intero"
K_PENDING = "subcortex.pending"
K_LAST_ERROR = "subcortex.last_error"
K_ACTED = "subcortex.acted"
K_HABIT_HIT = "subcortex.habit_hit"
K_METRICS = "subcortex.metrics"
K_FEATURES = "subcortex.features"

PRED_PARAMS = ("expected_effect", "confidence")


class Scene(BaseModel):
    """Contexto observable de la decisión (paso 8.1, giro parahipocampal)."""
    key: str
    features: dict[str, str]

    @classmethod
    def from_features(cls, features: dict[str, Any]) -> "Scene":
        norm = {str(k): str(v) for k, v in features.items()}
        raw = json.dumps(norm, sort_keys=True)
        return cls(key=hashlib.sha1(raw.encode()).hexdigest()[:12], features=norm)


class Prediction(BaseModel):
    """Copia eferente (paso 13): qué espera el agente de la acción."""
    tool: str
    args: dict[str, Any]
    expected: str
    confidence: float = Field(ge=0.0, le=1.0)


class Episode(BaseModel):
    id: int | None = None
    scene_key: str
    features: dict[str, str]
    tool: str
    args: dict[str, Any]
    expected: str
    observed: str
    prediction_error: float
    valence: float
    habenula: bool
    strength: float
    access_count: int = 0
    created_at: float = Field(default_factory=time.time)
    last_access: float = Field(default_factory=time.time)


class Habit(BaseModel):
    scene_key: str
    tool: str
    args: dict[str, Any]
    typical_effect: str
    strength: float
    successes: int
    failures: int


class Rule(BaseModel):
    scene_pattern: dict[str, str]
    tool: str
    text: str
    support: int


def prediction_error(expected: str, observed: str, confidence: float) -> float:
    """Error con signo en [-1, 1]. Positivo = mejor de lo esperado."""
    if expected == "diagnostic" or observed == "diagnostic":
        return 0.0
    diff = EFFECT_RANK[observed] - EFFECT_RANK[expected]
    scaled = max(-1.0, min(1.0, diff / 2.0))
    return scaled * confidence
```

```python
# subcortex/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .types import K_FEATURES, Scene


def default_scene_fn(state: Any) -> Scene | None:
    feats = state.get(K_FEATURES) if hasattr(state, "get") else None
    return Scene.from_features(feats) if feats else None


@dataclass
class SubcortexConfig:
    risk: dict[str, str]                                  # tool -> RiskClass
    diagnostic_tools: frozenset[str] = frozenset()
    scene_fn: Callable[[Any], Scene | None] = default_scene_fn
    always_allowed: frozenset[str] = frozenset({"escalate_to_human", "resolve"})
    step_budget: int = 8
    surprise_threshold: float = 0.25
    gate_threshold: float = 0.10
    cost: dict[str, float] = field(
        default_factory=lambda: {"free": 0.0, "costly": 0.15, "irreversible": 0.35}
    )
    dopamine_prior: float = 0.6
    hyperdirect_confidence: float = 0.7
    hyperdirect_tone: float = 0.4
    max_consecutive_blocks: int = 3
    habit_min_successes: int = 3
    habit_min_strength: float = 0.8

    @property
    def action_tools(self) -> frozenset[str]:
        return frozenset(self.risk)

    def is_action(self, name: str) -> bool:
        return name in self.risk

    def risk_of(self, name: str) -> str:
        return self.risk.get(name, "free")
```

```python
# subcortex/metrics.py
from __future__ import annotations

from typing import Any

from .types import K_METRICS

METRIC_KEYS = (
    "llm_calls", "tokens", "steps", "vetoes", "vetoes_irreversible", "rejected",
    "episodes_written", "habit_hits", "dehabituations", "abs_error_sum", "error_count",
)


def get_metrics(state: Any) -> dict[str, float]:
    m = dict(state.get(K_METRICS) or {})
    for k in METRIC_KEYS:
        m.setdefault(k, 0)
    return m


def bump(state: Any, key: str, n: float = 1) -> None:
    m = get_metrics(state)
    m[key] = m.get(key, 0) + n
    state[K_METRICS] = m


def set_metric(state: Any, key: str, value: float) -> None:
    m = get_metrics(state)
    m[key] = value
    state[K_METRICS] = m
```

Crear también `.gitignore` con: `.venv/`, `__pycache__/`, `*.pyc`, `*.sqlite`, `smoke.json`, `.env`, `.pytest_cache/`, `.ruff_cache/`; y `tests/__init__.py`, `tests/unit/__init__.py` vacíos.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_types.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add .gitignore subcortex/types.py subcortex/config.py subcortex/metrics.py tests pyproject.toml uv.lock subcortex/__init__.py opsworld/__init__.py
git commit -m "feat(subcortex): tipos, config, métricas y error de predicción"
```

---

### Task 2: Simulador opsworld — dinámica y score

**Files:**
- Create: `opsworld/world.py`
- Test: `tests/unit/test_world.py`

**Interfaces:**
- Produces: `Incident` (dataclass), `CAUSES`, `EFFECTS[cause][action] -> Effect`, `features(incident)->dict[str,str]`, `generate_incidents(n, seed)->list[Incident]`, `World(incident)` con `.act(action, **args)->dict`, `.diagnose(tool, **args)->dict`, `.done`, `.score`, `.steps`, `.worsens`, `.resolved`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_world.py
from opsworld.world import EFFECTS, Incident, World, features, generate_incidents


def make(cause: str) -> Incident:
    return Incident(id=1, service="api", cause=cause, recent_deploy=(cause == "bad_deploy"),
                    traffic="spike" if cause == "traffic_spike" else "normal", hour_bucket="evening")


def test_generate_is_deterministic_and_cycles_causes():
    a = generate_incidents(12, seed=7)
    b = generate_incidents(12, seed=7)
    assert [i.cause for i in a] == [i.cause for i in b]
    assert len({i.cause for i in a}) == 6


def test_features_hide_cause():
    f = features(make("memory_leak"))
    assert "cause" not in f
    assert set(f) == {"service", "symptom", "recent_deploy", "traffic", "hour_bucket"}


def test_restart_resolves_leak_and_scores():
    w = World(make("memory_leak"))
    r = w.act("restart", service="api")
    assert r["observed_effect"] == "resolves"
    assert w.resolved and w.done
    assert w.score == 100 - 10 - 5  # resolved, costly, one step


def test_restart_worsens_db_and_damages():
    w = World(make("db_saturated"))
    r = w.act("restart", service="api")
    assert r["observed_effect"] == "worsens"
    assert w.worsens == 1 and w.score == -10 - 40 - 5


def test_false_resolve_is_penalised_and_ends():
    w = World(make("bad_deploy"))
    r = w.act("resolve")
    assert r["observed_effect"] == "worsens" and w.done and not w.resolved
    assert w.score == -30 - 5


def test_escalate_only_pays_for_dependency_down():
    w = World(make("dependency_down"))
    assert w.act("escalate_to_human", reason="x")["observed_effect"] == "resolves"
    assert w.score == 30 - 5 and w.done
    w2 = World(make("memory_leak"))
    assert w2.act("escalate_to_human", reason="x")["observed_effect"] == "no_change"
    assert w2.score == -5 and w2.done


def test_step_cap_ends_episode():
    w = World(make("dependency_down"))
    for _ in range(8):
        w.diagnose("inspect_service", service="api")
    assert w.done and w.steps == 8


def test_diagnostic_hints_reflect_cause():
    assert "memory" in World(make("memory_leak")).diagnose("inspect_service", service="api")["metrics"]
    assert World(make("bad_deploy")).diagnose("check_deploys", service="api")["recent_deploy"] is True
    assert EFFECTS["false_alarm"]["resolve"] == "resolves"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_world.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'opsworld.world'`

- [ ] **Step 3: Write minimal implementation**

```python
# opsworld/world.py
"""Operador de guardia simulado. Determinista, sin LLM."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

CAUSES = ("memory_leak", "bad_deploy", "traffic_spike", "db_saturated", "dependency_down", "false_alarm")
SERVICES = ("api", "checkout", "search", "auth")
HOURS = ("morning", "afternoon", "evening", "night")

SYMPTOM_OF = {
    "memory_leak": "oom",
    "bad_deploy": "errors_5xx",
    "traffic_spike": "high_latency",
    "db_saturated": "high_latency",
    "dependency_down": "errors_5xx",
    "false_alarm": "queue_growing",
}

# causa × acción -> efecto observado (tabla de la spec, sección 5)
EFFECTS: dict[str, dict[str, str]] = {
    "memory_leak":     {"restart": "resolves",  "rollback": "no_change", "scale": "improves",  "failover_db": "worsens",   "resolve": "worsens"},
    "bad_deploy":      {"restart": "no_change", "rollback": "resolves",  "scale": "no_change", "failover_db": "worsens",   "resolve": "worsens"},
    "traffic_spike":   {"restart": "worsens",   "rollback": "no_change", "scale": "resolves",  "failover_db": "no_change", "resolve": "worsens"},
    "db_saturated":    {"restart": "worsens",   "rollback": "no_change", "scale": "worsens",   "failover_db": "resolves",  "resolve": "worsens"},
    "dependency_down": {"restart": "no_change", "rollback": "no_change", "scale": "no_change", "failover_db": "worsens",   "resolve": "worsens"},
    "false_alarm":     {"restart": "no_change", "rollback": "worsens",   "scale": "no_change", "failover_db": "worsens",   "resolve": "resolves"},
}

RISK = {"restart": "costly", "rollback": "irreversible", "scale": "costly",
        "failover_db": "irreversible", "resolve": "free", "escalate_to_human": "free"}
DIAGNOSTIC_TOOLS = frozenset({"inspect_service", "check_deploys"})

ACTION_COST = {"free": 0, "costly": 10, "irreversible": 25}
STEP_COST = 5
WORSEN_DAMAGE = 40
RESOLVE_REWARD = 100
ESCALATE_REWARD = 30
FALSE_RESOLVE_PENALTY = 30
MAX_STEPS = 8


@dataclass
class Incident:
    id: int
    service: str
    cause: str
    recent_deploy: bool
    traffic: str
    hour_bucket: str

    @property
    def symptom(self) -> str:
        return SYMPTOM_OF[self.cause]


def features(inc: Incident) -> dict[str, str]:
    """Solo lo observable; nunca la causa."""
    return {
        "service": inc.service,
        "symptom": inc.symptom,
        "recent_deploy": str(inc.recent_deploy).lower(),
        "traffic": inc.traffic,
        "hour_bucket": inc.hour_bucket,
    }


def generate_incidents(n: int, seed: int) -> list[Incident]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        cause = CAUSES[i % len(CAUSES)]
        out.append(Incident(
            id=i,
            service=rng.choice(SERVICES),
            cause=cause,
            recent_deploy=(cause == "bad_deploy") or (rng.random() < 0.15),
            traffic="spike" if cause == "traffic_spike" else ("spike" if rng.random() < 0.1 else "normal"),
            hour_bucket=rng.choice(HOURS),
        ))
    return out


@dataclass
class World:
    incident: Incident
    steps: int = 0
    score: int = 0
    worsens: int = 0
    resolved: bool = False
    done: bool = False
    log: list[dict] = field(default_factory=list)

    def _step(self) -> None:
        self.steps += 1
        self.score -= STEP_COST
        if self.steps >= MAX_STEPS:
            self.done = True

    def act(self, action: str, **args) -> dict:
        if self.done:
            return {"status": "error", "observed_effect": "no_change",
                    "message": "El incidente ya está cerrado."}
        self._step()
        cause = self.incident.cause
        if action == "escalate_to_human":
            effect = "resolves" if cause == "dependency_down" else "no_change"
            if cause == "dependency_down":
                self.score += ESCALATE_REWARD
                self.resolved = True
            self.done = True
        elif action == "resolve":
            if self.resolved or cause == "false_alarm":
                effect = "resolves"
                if not self.resolved:
                    self.score += RESOLVE_REWARD
                self.resolved = True
            else:
                effect = "worsens"
                self.score -= FALSE_RESOLVE_PENALTY
            self.done = True
        else:
            effect = EFFECTS[cause][action]
            self.score -= ACTION_COST[RISK[action]]
            if effect == "resolves" and not self.resolved:
                self.resolved = True
                self.score += RESOLVE_REWARD
                self.done = True
            elif effect == "worsens":
                self.worsens += 1
                self.score -= WORSEN_DAMAGE
        entry = {"action": action, "args": args, "observed_effect": effect, "step": self.steps}
        self.log.append(entry)
        return {"status": "success", "observed_effect": effect,
                "message": _message(action, effect, self.incident)}

    def diagnose(self, tool: str, **args) -> dict:
        if not self.done:
            self._step()
        inc = self.incident
        if tool == "check_deploys":
            return {"status": "success", "observed_effect": "diagnostic",
                    "recent_deploy": inc.recent_deploy,
                    "last_deploy": "12 min ago" if inc.recent_deploy else "3 days ago"}
        return {"status": "success", "observed_effect": "diagnostic",
                "service": inc.service, "metrics": _HINTS[inc.cause]}


_HINTS = {
    "memory_leak": "memory 96% and climbing steadily; rps normal; db pool healthy",
    "bad_deploy": "5xx started minutes after the last deploy; memory normal; rps normal",
    "traffic_spike": "rps 5x baseline; cpu saturated on all replicas; memory normal",
    "db_saturated": "db connection pool exhausted; slow queries; app cpu low",
    "dependency_down": "upstream payment-api timing out 100%; app healthy otherwise",
    "false_alarm": "all metrics nominal; queue depth flat; alert threshold misconfigured",
}


def _message(action: str, effect: str, inc: Incident) -> str:
    return {
        "resolves": f"{action} en {inc.service}: el incidente quedó resuelto.",
        "improves": f"{action} en {inc.service}: mejora parcial, el síntoma persiste.",
        "no_change": f"{action} en {inc.service}: sin cambios.",
        "worsens": f"{action} en {inc.service}: empeoró, hay daño adicional.",
    }[effect]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_world.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add opsworld/world.py tests/unit/test_world.py
git commit -m "feat(opsworld): simulador determinista de incidentes con dinámica oculta y score"
```

---

### Task 3: opsworld — tools ADK y registro de mundos

**Files:**
- Create: `opsworld/tools.py`
- Test: `tests/unit/test_world_tools.py`

**Interfaces:**
- Consumes: `World`, `RISK`, `DIAGNOSTIC_TOOLS`.
- Produces: `registry: WorldRegistry` (`.register(session_id, world)`, `.get(session_id)`, `.clear()`), tools `restart(service, tool_context)`, `rollback(service, tool_context)`, `scale(service, replicas, tool_context)`, `failover_db(tool_context)`, `resolve(tool_context)`, `escalate_to_human(reason, tool_context)`, `inspect_service(service, tool_context)`, `check_deploys(service, tool_context)`, `ACTION_TOOLS`, `ALL_TOOLS`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_world_tools.py
from types import SimpleNamespace

from opsworld.tools import ALL_TOOLS, registry, restart, inspect_service
from opsworld.world import Incident, World


def ctx(session_id="s1"):
    return SimpleNamespace(session=SimpleNamespace(id=session_id), state={})


def test_tools_route_to_session_world():
    registry.clear()
    w = World(Incident(id=0, service="api", cause="memory_leak", recent_deploy=False,
                       traffic="normal", hour_bucket="night"))
    registry.register("s1", w)
    assert inspect_service("api", ctx())["observed_effect"] == "diagnostic"
    assert restart("api", ctx())["observed_effect"] == "resolves"
    assert w.steps == 2


def test_unknown_session_returns_error():
    registry.clear()
    assert restart("api", ctx("nope"))["status"] == "error"


def test_all_tools_have_docstrings_and_names():
    names = {t.__name__ for t in ALL_TOOLS}
    assert {"restart", "rollback", "scale", "failover_db", "resolve",
            "escalate_to_human", "inspect_service", "check_deploys"} == names
    assert all(t.__doc__ for t in ALL_TOOLS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_world_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'opsworld.tools'`

- [ ] **Step 3: Write minimal implementation**

```python
# opsworld/tools.py
"""Tools ADK del simulador. Cada sesión ADK tiene su World en el registro."""
from __future__ import annotations

from google.adk.tools.tool_context import ToolContext

from .world import World


class WorldRegistry:
    def __init__(self) -> None:
        self._worlds: dict[str, World] = {}

    def register(self, session_id: str, world: World) -> None:
        self._worlds[session_id] = world

    def get(self, session_id: str) -> World | None:
        return self._worlds.get(session_id)

    def clear(self) -> None:
        self._worlds.clear()


registry = WorldRegistry()
_NO_WORLD = {"status": "error", "observed_effect": "no_change",
             "message": "No hay incidente activo para esta sesión."}


def _world(tool_context: ToolContext) -> World | None:
    return registry.get(tool_context.session.id)


def restart(service: str, tool_context: ToolContext) -> dict:
    """Reinicia todas las réplicas del servicio. Costoso: interrumpe tráfico unos segundos."""
    w = _world(tool_context)
    return w.act("restart", service=service) if w else _NO_WORLD


def rollback(service: str, tool_context: ToolContext) -> dict:
    """Vuelve el servicio a la versión anterior. Irreversible: pierde los cambios del último deploy."""
    w = _world(tool_context)
    return w.act("rollback", service=service) if w else _NO_WORLD


def scale(service: str, replicas: int, tool_context: ToolContext) -> dict:
    """Cambia la cantidad de réplicas del servicio. Costoso: consume presupuesto de infraestructura."""
    w = _world(tool_context)
    return w.act("scale", service=service, replicas=replicas) if w else _NO_WORLD


def failover_db(tool_context: ToolContext) -> dict:
    """Promueve la réplica de base de datos a primaria. Irreversible: puede perder escrituras en vuelo."""
    w = _world(tool_context)
    return w.act("failover_db") if w else _NO_WORLD


def resolve(tool_context: ToolContext) -> dict:
    """Cierra el incidente como resuelto. Cerrar un incidente no resuelto es una falta grave."""
    w = _world(tool_context)
    return w.act("resolve") if w else _NO_WORLD


def escalate_to_human(reason: str, tool_context: ToolContext) -> dict:
    """Escala el incidente al ingeniero de guardia humano y termina tu intervención."""
    w = _world(tool_context)
    return w.act("escalate_to_human", reason=reason) if w else _NO_WORLD


def inspect_service(service: str, tool_context: ToolContext) -> dict:
    """Devuelve métricas actuales del servicio (memoria, rps, cpu, db, dependencias)."""
    w = _world(tool_context)
    return w.diagnose("inspect_service", service=service) if w else _NO_WORLD


def check_deploys(service: str, tool_context: ToolContext) -> dict:
    """Informa si hubo un deploy reciente del servicio."""
    w = _world(tool_context)
    return w.diagnose("check_deploys", service=service) if w else _NO_WORLD


ACTION_TOOLS = [restart, rollback, scale, failover_db, resolve, escalate_to_human]
DIAG_TOOLS = [inspect_service, check_deploys]
ALL_TOOLS = ACTION_TOOLS + DIAG_TOOLS
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_world_tools.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add opsworld/tools.py tests/unit/test_world_tools.py
git commit -m "feat(opsworld): tools ADK y registro de mundos por sesión"
```

---

### Task 4: EpisodicStore (SQLite)

**Files:**
- Create: `subcortex/store.py`
- Test: `tests/unit/test_store.py`

**Interfaces:**
- Consumes: `Episode`, `Habit`, `Rule`, `Scene`.
- Produces: `EpisodicStore(path=":memory:")` con `write(ep)->int`, `recall(scene, k, now)->list[Episode]`, `record_outcome(scene_key, tool, success)`, `outcome_counts(scene_key, tool)->(int,int)`, `dopamine(scene_key, tool, prior)->float`, `upsert_habit(h)`, `get_habit(scene_key)->Habit|None`, `habits()->list[Habit]`, `upsert_rule(r)`, `rules_for(scene)->list[Rule]`, `all_episodes()->list[Episode]`, `update_strength(id, strength)`, `delete_episodes(ids)`, `stats()->dict`, `close()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_store.py
import pytest

from subcortex.store import EpisodicStore
from subcortex.types import Episode, Habit, Rule, Scene


def ep(scene: Scene, tool="restart", err=-0.6, **kw) -> Episode:
    return Episode(scene_key=scene.key, features=scene.features, tool=tool, args={"service": "api"},
                   expected="resolves", observed="worsens", prediction_error=err, valence=err,
                   habenula=err < 0, strength=abs(err), **kw)


@pytest.fixture
def store():
    s = EpisodicStore(":memory:")
    yield s
    s.close()


def test_write_and_recall_same_scene_failures_first(store):
    scene = Scene.from_features({"service": "api", "symptom": "high_latency"})
    store.write(ep(scene, err=0.3, created_at=1, last_access=1))
    store.write(ep(scene, err=-0.6, created_at=2, last_access=2))
    got = store.recall(scene, k=5, now=100)
    assert [e.prediction_error for e in got] == [-0.6, 0.3]
    assert all(e.access_count == 1 and e.last_access == 100 for e in got)


def test_recall_ranks_exact_scene_over_partial_overlap(store):
    exact = Scene.from_features({"service": "api", "symptom": "oom"})
    partial = Scene.from_features({"service": "auth", "symptom": "oom"})
    other = Scene.from_features({"service": "auth", "symptom": "errors_5xx"})
    store.write(ep(partial, err=-0.9))
    store.write(ep(exact, err=-0.3))
    store.write(ep(other, err=-0.9))
    got = store.recall(exact, k=2, now=0)
    assert [e.scene_key for e in got] == [exact.key, partial.key]


def test_dopamine_prior_and_update(store):
    assert store.dopamine("s", "restart", prior=0.6) == pytest.approx(0.6)
    store.record_outcome("s", "restart", True)
    store.record_outcome("s", "restart", True)
    store.record_outcome("s", "restart", False)
    assert store.outcome_counts("s", "restart") == (2, 1)
    # (2 + 0.6*2) / (3 + 2) = 0.64
    assert store.dopamine("s", "restart", prior=0.6) == pytest.approx(0.64)


def test_habit_and_rule_roundtrip(store):
    h = Habit(scene_key="s", tool="restart", args={"service": "api"}, typical_effect="resolves",
              strength=0.85, successes=3, failures=0)
    store.upsert_habit(h)
    assert store.get_habit("s") == h
    h.strength = 0.4
    store.upsert_habit(h)
    assert store.get_habit("s").strength == 0.4 and len(store.habits()) == 1
    r = Rule(scene_pattern={"symptom": "oom"}, tool="restart", text="t", support=3)
    store.upsert_rule(r)
    store.upsert_rule(Rule(scene_pattern={"symptom": "oom"}, tool="restart", text="t2", support=4))
    rules = store.rules_for(Scene.from_features({"symptom": "oom", "service": "x"}))
    assert len(rules) == 1 and rules[0].support == 4
    assert store.rules_for(Scene.from_features({"symptom": "errors_5xx"})) == []


def test_strength_update_delete_and_stats(store):
    scene = Scene.from_features({"a": "1"})
    i = store.write(ep(scene))
    store.update_strength(i, 0.01)
    assert store.all_episodes()[0].strength == 0.01
    store.delete_episodes([i])
    assert store.all_episodes() == []
    assert store.stats()["episodes"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'subcortex.store'`

- [ ] **Step 3: Write minimal implementation**

```python
# subcortex/store.py
"""Almacén episódico (hipocampo) + contadores de dopamina (estriado) + hábitos + reglas."""
from __future__ import annotations

import json
import logging
import sqlite3
import time

from .types import Episode, Habit, Rule, Scene

log = logging.getLogger("subcortex")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
  id INTEGER PRIMARY KEY AUTOINCREMENT, scene_key TEXT, features TEXT, tool TEXT, args TEXT,
  expected TEXT, observed TEXT, prediction_error REAL, valence REAL, habenula INTEGER,
  strength REAL, access_count INTEGER, created_at REAL, last_access REAL);
CREATE INDEX IF NOT EXISTS ep_scene ON episodes(scene_key);
CREATE TABLE IF NOT EXISTS dopamine (
  scene_key TEXT, tool TEXT, successes INTEGER, failures INTEGER, PRIMARY KEY (scene_key, tool));
CREATE TABLE IF NOT EXISTS habits (
  scene_key TEXT PRIMARY KEY, tool TEXT, args TEXT, typical_effect TEXT, strength REAL,
  successes INTEGER, failures INTEGER);
CREATE TABLE IF NOT EXISTS rules (
  pattern TEXT, tool TEXT, text TEXT, support INTEGER, PRIMARY KEY (pattern, tool));
"""


class EpisodicStore:
    def __init__(self, path: str = ":memory:") -> None:
        try:
            self.conn = sqlite3.connect(path, check_same_thread=False)
        except sqlite3.Error as e:  # degradación: nunca romper al agente
            log.warning("SQLite no disponible en %s (%s); usando :memory:", path, e)
            self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # --- episodios -----------------------------------------------------------
    def write(self, ep: Episode) -> int:
        cur = self.conn.execute(
            "INSERT INTO episodes (scene_key, features, tool, args, expected, observed, prediction_error,"
            " valence, habenula, strength, access_count, created_at, last_access)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ep.scene_key, json.dumps(ep.features, sort_keys=True), ep.tool, json.dumps(ep.args, sort_keys=True),
             ep.expected, ep.observed, ep.prediction_error, ep.valence, int(ep.habenula), ep.strength,
             ep.access_count, ep.created_at, ep.last_access))
        self.conn.commit()
        return int(cur.lastrowid)

    def recall(self, scene: Scene, k: int, now: float | None = None) -> list[Episode]:
        """Misma escena primero, luego por overlap de features; dentro: habénula, luego fuerza."""
        now = time.time() if now is None else now
        rows = [self._row_to_episode(r) for r in self.conn.execute("SELECT * FROM episodes")]
        feats = scene.features.items()

        def overlap(e: Episode) -> int:
            return sum(1 for kv in feats if kv in e.features.items())

        cands = [e for e in rows if e.scene_key == scene.key or overlap(e) > 0]
        cands.sort(key=lambda e: (e.scene_key != scene.key, -overlap(e), not e.habenula, -e.strength))
        chosen = cands[:k]
        for e in chosen:
            e.access_count += 1
            e.last_access = now
            self.conn.execute("UPDATE episodes SET access_count=?, last_access=? WHERE id=?",
                              (e.access_count, now, e.id))
        self.conn.commit()
        return chosen

    def all_episodes(self) -> list[Episode]:
        return [self._row_to_episode(r) for r in self.conn.execute("SELECT * FROM episodes ORDER BY id")]

    def update_strength(self, episode_id: int, strength: float) -> None:
        self.conn.execute("UPDATE episodes SET strength=? WHERE id=?", (strength, episode_id))
        self.conn.commit()

    def delete_episodes(self, ids: list[int]) -> None:
        if ids:
            self.conn.executemany("DELETE FROM episodes WHERE id=?", [(i,) for i in ids])
            self.conn.commit()

    @staticmethod
    def _row_to_episode(r: sqlite3.Row) -> Episode:
        return Episode(id=r["id"], scene_key=r["scene_key"], features=json.loads(r["features"]),
                       tool=r["tool"], args=json.loads(r["args"]), expected=r["expected"],
                       observed=r["observed"], prediction_error=r["prediction_error"], valence=r["valence"],
                       habenula=bool(r["habenula"]), strength=r["strength"], access_count=r["access_count"],
                       created_at=r["created_at"], last_access=r["last_access"])

    # --- dopamina --------------------------------------------------------------
    def record_outcome(self, scene_key: str, tool: str, success: bool) -> None:
        self.conn.execute(
            "INSERT INTO dopamine (scene_key, tool, successes, failures) VALUES (?,?,?,?)"
            " ON CONFLICT(scene_key, tool) DO UPDATE SET successes=successes+excluded.successes,"
            " failures=failures+excluded.failures",
            (scene_key, tool, int(success), int(not success)))
        self.conn.commit()

    def outcome_counts(self, scene_key: str, tool: str) -> tuple[int, int]:
        r = self.conn.execute("SELECT successes, failures FROM dopamine WHERE scene_key=? AND tool=?",
                              (scene_key, tool)).fetchone()
        return (r["successes"], r["failures"]) if r else (0, 0)

    def dopamine(self, scene_key: str, tool: str, prior: float) -> float:
        s, f = self.outcome_counts(scene_key, tool)
        return (s + prior * 2) / (s + f + 2)

    # --- hábitos ---------------------------------------------------------------
    def upsert_habit(self, h: Habit) -> None:
        self.conn.execute(
            "INSERT INTO habits (scene_key, tool, args, typical_effect, strength, successes, failures)"
            " VALUES (?,?,?,?,?,?,?) ON CONFLICT(scene_key) DO UPDATE SET tool=excluded.tool,"
            " args=excluded.args, typical_effect=excluded.typical_effect, strength=excluded.strength,"
            " successes=excluded.successes, failures=excluded.failures",
            (h.scene_key, h.tool, json.dumps(h.args, sort_keys=True), h.typical_effect, h.strength,
             h.successes, h.failures))
        self.conn.commit()

    def get_habit(self, scene_key: str) -> Habit | None:
        r = self.conn.execute("SELECT * FROM habits WHERE scene_key=?", (scene_key,)).fetchone()
        return self._row_to_habit(r) if r else None

    def habits(self) -> list[Habit]:
        return [self._row_to_habit(r) for r in self.conn.execute("SELECT * FROM habits")]

    @staticmethod
    def _row_to_habit(r: sqlite3.Row) -> Habit:
        return Habit(scene_key=r["scene_key"], tool=r["tool"], args=json.loads(r["args"]),
                     typical_effect=r["typical_effect"], strength=r["strength"],
                     successes=r["successes"], failures=r["failures"])

    # --- reglas ----------------------------------------------------------------
    def upsert_rule(self, rule: Rule) -> None:
        self.conn.execute(
            "INSERT INTO rules (pattern, tool, text, support) VALUES (?,?,?,?)"
            " ON CONFLICT(pattern, tool) DO UPDATE SET text=excluded.text, support=excluded.support",
            (json.dumps(rule.scene_pattern, sort_keys=True), rule.tool, rule.text, rule.support))
        self.conn.commit()

    def rules_for(self, scene: Scene) -> list[Rule]:
        out = []
        for r in self.conn.execute("SELECT * FROM rules ORDER BY support DESC"):
            pattern = json.loads(r["pattern"])
            if all(scene.features.get(k) == v for k, v in pattern.items()):
                out.append(Rule(scene_pattern=pattern, tool=r["tool"], text=r["text"], support=r["support"]))
        return out

    def stats(self) -> dict:
        q = lambda t: self.conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"]  # noqa: E731
        return {"episodes": q("episodes"), "habits": q("habits"), "rules": q("rules")}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_store.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add subcortex/store.py tests/unit/test_store.py
git commit -m "feat(subcortex): EpisodicStore SQLite con episodios, dopamina, hábitos y reglas"
```

---

### Task 5: InteroceptionPlugin

**Files:**
- Create: `subcortex/interoception.py`
- Test: `tests/unit/test_interoception.py`

**Interfaces:**
- Consumes: `SubcortexConfig`, `K_INTERO`, `K_TONE`, `bump`.
- Produces: `compute_tone(intero: dict, cfg)->float`, `render_state(intero, tone, cfg)->str`, `InteroceptionPlugin(cfg)`; estado `K_INTERO = {"steps","failures","blocks","last_status","started_at"}`, `K_TONE: float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_interoception.py
from types import SimpleNamespace

import pytest

from subcortex.config import SubcortexConfig
from subcortex.interoception import InteroceptionPlugin, compute_tone, render_state
from subcortex.types import K_INTERO, K_TONE

CFG = SubcortexConfig(risk={"restart": "costly"}, diagnostic_tools=frozenset({"inspect_service"}))


def test_tone_starts_high_and_drops():
    fresh = {"steps": 0, "failures": 0, "blocks": 0}
    assert compute_tone(fresh, CFG) == 1.0
    assert compute_tone({"steps": 4, "failures": 0, "blocks": 0}, CFG) == pytest.approx(0.75)
    assert compute_tone({"steps": 0, "failures": 2, "blocks": 1}, CFG) == pytest.approx(0.55)
    assert compute_tone({"steps": 8, "failures": 5, "blocks": 5}, CFG) == 0.05


def test_render_translates_not_dumps():
    txt = render_state({"steps": 6, "failures": 2, "blocks": 1, "last_status": "vetoed"}, 0.3, CFG)
    assert "Estado interno" in txt and "2 fallos seguidos" in txt and "75" in txt and "vetada" in txt


class Tool:
    def __init__(self, name): self.name = name


@pytest.mark.asyncio
async def test_plugin_updates_counters_and_injects():
    p = InteroceptionPlugin(CFG)
    ctx = SimpleNamespace(state={})
    await p.after_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=ctx,
                                result={"status": "vetoed"})
    await p.after_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=ctx,
                                result={"status": "error"})
    i = ctx.state[K_INTERO]
    assert i["steps"] == 1 and i["failures"] == 1 and i["blocks"] == 0
    assert ctx.state[K_TONE] < 1.0
    req = SimpleNamespace(appended=[])
    req.append_instructions = lambda xs: req.appended.extend(xs)
    assert await p.before_model_callback(callback_context=ctx, llm_request=req) is None
    assert "Estado interno" in req.appended[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_interoception.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'subcortex.interoception'`

- [ ] **Step 3: Write minimal implementation**

```python
# subcortex/interoception.py
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
BLOCK_STATUSES = {"vetoed", "rejected"}


def _intero(state: Any) -> dict:
    return dict(state.get(K_INTERO) or {"steps": 0, "failures": 0, "blocks": 0,
                                        "last_status": None, "started_at": time.time()})


def compute_tone(intero: dict, cfg: SubcortexConfig) -> float:
    tone = (1.0 - 0.15 * intero.get("failures", 0) - 0.15 * intero.get("blocks", 0)
            - 0.5 * intero.get("steps", 0) / max(cfg.step_budget, 1))
    return round(max(0.05, min(1.0, tone)), 4)


def render_state(intero: dict, tone: float, cfg: SubcortexConfig) -> str:
    lines = ["## Estado interno"]
    used = int(100 * intero.get("steps", 0) / max(cfg.step_budget, 1))
    lines.append(f"- Presupuesto de acciones consumido: {used} %.")
    f = intero.get("failures", 0)
    if f:
        lines.append(f"- Llevás {f} fallos seguidos: bajá la confianza y diagnosticá antes de actuar.")
    b = intero.get("blocks", 0)
    if b:
        lines.append(f"- Tus últimas {b} acciones fueron bloqueadas: cambiá de estrategia o escalá.")
    if intero.get("last_status") == "vetoed":
        lines.append("- La última acción fue vetada por riesgo.")
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
            llm_request.append_instructions([render_state(intero, tone, self.cfg)])
        except Exception:  # degradación a vanilla
            log.exception("interoception.before_model")
        return None

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        try:
            state = tool_context.state
            intero = _intero(state)
            status = (result or {}).get("status")
            if status in BLOCK_STATUSES:
                intero["blocks"] += 1
            else:
                intero["blocks"] = 0
                intero["steps"] += 1
                bump(state, "steps")
                if status == "error":
                    intero["failures"] += 1
                else:
                    intero["failures"] = 0
            intero["last_status"] = status
            state[K_INTERO] = intero
            state[K_TONE] = compute_tone(intero, self.cfg)
        except Exception:
            log.exception("interoception.after_tool")
        return None

    async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error):
        # Convertimos la excepción en resultado para que el loop siga y se aprenda de ella.
        log.warning("tool %s falló: %s", tool.name, error)
        return {"status": "error", "observed_effect": "worsens", "message": str(error)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_interoception.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add subcortex/interoception.py tests/unit/test_interoception.py
git commit -m "feat(subcortex): InteroceptionPlugin con tono y estado interno traducido"
```

---

### Task 6: PredictionPlugin y wrapping de tools

**Files:**
- Create: `subcortex/prediction.py`
- Test: `tests/unit/test_prediction.py`

**Interfaces:**
- Consumes: `Prediction`, `prediction_error`, `PRED_PARAMS`, `EFFECTS`, `K_PENDING`, `K_LAST_ERROR`, `bump`.
- Produces: `wrap_action_tool(func)->callable` (misma `__name__`, firma extendida con `expected_effect: str, confidence: float`, docstring extendido), `PROTOCOL_INSTRUCTION: str`, `PredictionPlugin(cfg)`; estado `K_PENDING: dict[call_id -> Prediction dict]`, `K_LAST_ERROR: {"tool","args","expected","observed","confidence","error","status","call_id"}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_prediction.py
import inspect
from types import SimpleNamespace

import pytest

from subcortex.config import SubcortexConfig
from subcortex.prediction import PredictionPlugin, wrap_action_tool
from subcortex.types import K_LAST_ERROR, K_PENDING

CFG = SubcortexConfig(risk={"restart": "costly"}, diagnostic_tools=frozenset({"inspect_service"}))


def restart(service: str, tool_context) -> dict:
    """Reinicia el servicio."""
    return {"status": "success", "observed_effect": "resolves", "svc": service}


def test_wrap_extends_signature_and_strips_params():
    w = wrap_action_tool(restart)
    params = list(inspect.signature(w).parameters)
    assert params == ["service", "tool_context", "expected_effect", "confidence"]
    assert w.__name__ == "restart" and "expected_effect" in w.__doc__
    assert w(service="api", tool_context=None, expected_effect="resolves", confidence=0.9)["svc"] == "api"


class Tool:
    def __init__(self, name): self.name = name


def ctx(call_id="c1"):
    return SimpleNamespace(state={}, function_call_id=call_id)


@pytest.mark.asyncio
async def test_before_tool_rejects_missing_or_invalid_prediction():
    p = PredictionPlugin(CFG)
    r = await p.before_tool_callback(tool=Tool("restart"), tool_args={"service": "api"}, tool_context=ctx())
    assert r["status"] == "rejected"
    r = await p.before_tool_callback(tool=Tool("restart"), tool_context=ctx(),
                                     tool_args={"service": "api", "expected_effect": "magic", "confidence": 0.5})
    assert r["status"] == "rejected"
    assert await p.before_tool_callback(tool=Tool("inspect_service"), tool_args={}, tool_context=ctx()) is None


@pytest.mark.asyncio
async def test_after_tool_computes_error_and_skips_blocked():
    p = PredictionPlugin(CFG)
    c = ctx()
    args = {"service": "api", "expected_effect": "resolves", "confidence": 1.0}
    assert await p.before_tool_callback(tool=Tool("restart"), tool_args=args, tool_context=c) is None
    assert "c1" in c.state[K_PENDING]
    await p.after_tool_callback(tool=Tool("restart"), tool_args=args, tool_context=c,
                                result={"status": "success", "observed_effect": "worsens"})
    le = c.state[K_LAST_ERROR]
    assert le["error"] == -1.0 and le["observed"] == "worsens" and "c1" not in c.state[K_PENDING]
    c2 = ctx("c2")
    await p.before_tool_callback(tool=Tool("restart"), tool_args=args, tool_context=c2)
    await p.after_tool_callback(tool=Tool("restart"), tool_args=args, tool_context=c2,
                                result={"status": "vetoed"})
    assert c2.state.get(K_LAST_ERROR) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_prediction.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'subcortex.prediction'`

- [ ] **Step 3: Write minimal implementation**

```python
# subcortex/prediction.py
"""Cerebelo: copia eferente antes de actuar, error de predicción después."""
from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable

from google.adk.plugins.base_plugin import BasePlugin

from .config import SubcortexConfig
from .metrics import bump
from .types import EFFECTS, K_LAST_ERROR, K_PENDING, PRED_PARAMS, Prediction, prediction_error

log = logging.getLogger("subcortex")
BLOCK_STATUSES = {"vetoed", "rejected"}

PROTOCOL_INSTRUCTION = """## Protocolo de acción (subcortex)
Cada tool de acción exige dos parámetros extra:
- `expected_effect`: qué esperás que pase. Uno de: resolves, improves, no_change, worsens.
- `confidence`: entre 0 y 1, cuán seguro estás de ese efecto.
Las acciones sin predicción se rechazan. Las acciones irreversibles con confianza baja se vetan.
Si una acción vuelve con status "vetoed" o "rejected", no la repitas igual: diagnosticá, cambiá de acción o escalá.
Ejecutá una sola acción por turno."""

_DOC_SUFFIX = """

Args extra (obligatorios):
    expected_effect: efecto esperado; uno de resolves, improves, no_change, worsens.
    confidence: confianza en ese efecto, entre 0 y 1."""


def wrap_action_tool(func: Callable[..., Any]) -> Callable[..., Any]:
    sig = inspect.signature(func)
    extra = [inspect.Parameter("expected_effect", inspect.Parameter.KEYWORD_ONLY, annotation=str),
             inspect.Parameter("confidence", inspect.Parameter.KEYWORD_ONLY, annotation=float)]
    new_params = list(sig.parameters.values()) + extra

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for p in PRED_PARAMS:
            kwargs.pop(p, None)
        return func(*args, **kwargs)

    wrapper.__signature__ = sig.replace(parameters=new_params)
    wrapper.__annotations__ = {**getattr(func, "__annotations__", {}), "expected_effect": str, "confidence": float}
    wrapper.__doc__ = (inspect.cleandoc(func.__doc__ or func.__name__)) + _DOC_SUFFIX
    return wrapper


def parse_prediction(tool: str, args: dict) -> Prediction | str:
    exp = args.get("expected_effect")
    conf = args.get("confidence")
    if exp not in EFFECTS or exp == "diagnostic":
        return "falta o es inválido expected_effect (resolves|improves|no_change|worsens)"
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        return "falta o es inválido confidence (0..1)"
    if not 0.0 <= conf <= 1.0:
        return "confidence fuera de rango 0..1"
    clean = {k: v for k, v in args.items() if k not in PRED_PARAMS}
    return Prediction(tool=tool, args=clean, expected=exp, confidence=conf)


def infer_observed(result: dict) -> str:
    obs = result.get("observed_effect")
    if obs in EFFECTS:
        return obs
    return "worsens" if result.get("status") == "error" else "no_change"


class PredictionPlugin(BasePlugin):
    def __init__(self, cfg: SubcortexConfig) -> None:
        super().__init__(name="subcortex_prediction")
        self.cfg = cfg

    async def before_model_callback(self, *, callback_context, llm_request):
        try:
            llm_request.append_instructions([PROTOCOL_INSTRUCTION])
        except Exception:
            log.exception("prediction.before_model")
        return None

    async def before_tool_callback(self, *, tool, tool_args, tool_context):
        if not self.cfg.is_action(tool.name):
            return None
        try:
            parsed = parse_prediction(tool.name, tool_args)
            state = tool_context.state
            if isinstance(parsed, str):
                bump(state, "rejected")
                return {"status": "rejected", "reason": parsed,
                        "hint": "Repetí la llamada agregando expected_effect y confidence."}
            pending = dict(state.get(K_PENDING) or {})
            pending[tool_context.function_call_id or tool.name] = parsed.model_dump()
            state[K_PENDING] = pending
        except Exception:
            log.exception("prediction.before_tool")
        return None

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        if not self.cfg.is_action(tool.name):
            return None
        try:
            state = tool_context.state
            call_id = tool_context.function_call_id or tool.name
            pending = dict(state.get(K_PENDING) or {})
            pred = pending.pop(call_id, None)
            state[K_PENDING] = pending
            status = (result or {}).get("status")
            if status in BLOCK_STATUSES or pred is None:
                state[K_LAST_ERROR] = None
                return None
            observed = infer_observed(result or {})
            err = prediction_error(pred["expected"], observed, pred["confidence"])
            state[K_LAST_ERROR] = {"tool": tool.name, "args": pred["args"], "expected": pred["expected"],
                                   "observed": observed, "confidence": pred["confidence"], "error": err,
                                   "status": status, "call_id": call_id}
            bump(state, "abs_error_sum", abs(err))
            bump(state, "error_count")
        except Exception:
            log.exception("prediction.after_tool")
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_prediction.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add subcortex/prediction.py tests/unit/test_prediction.py
git commit -m "feat(subcortex): PredictionPlugin y wrapping de tools con expected_effect/confidence"
```

---

### Task 7: GatePlugin

**Files:**
- Create: `subcortex/gate.py`
- Test: `tests/unit/test_gate.py`

**Interfaces:**
- Consumes: `SubcortexConfig`, `EpisodicStore.dopamine`, `K_PENDING`, `K_TONE`, `K_INTERO`, `bump`, `LlmResponse.get_function_calls()`.
- Produces: `gate_decision(cfg, tool, confidence, dopamine, tone, consecutive_blocks)->(bool, float, str)`, `GatePlugin(cfg, store)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_gate.py
from types import SimpleNamespace

import pytest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from subcortex.config import SubcortexConfig
from subcortex.gate import GatePlugin, gate_decision
from subcortex.store import EpisodicStore
from subcortex.types import K_FEATURES, K_INTERO, K_PENDING, K_TONE

CFG = SubcortexConfig(risk={"restart": "costly", "rollback": "irreversible", "resolve": "free",
                            "escalate_to_human": "free"},
                      diagnostic_tools=frozenset({"inspect_service"}))


def test_decision_matrix():
    ok, v, _ = gate_decision(CFG, "restart", 0.9, 0.6, 1.0, 0)
    assert ok and v == pytest.approx(0.39)
    ok, _, why = gate_decision(CFG, "restart", 0.3, 0.6, 1.0, 0)          # 0.18-0.15 < 0.1
    assert not ok and "valor" in why
    ok, _, why = gate_decision(CFG, "rollback", 0.6, 0.9, 1.0, 0)         # hiperdirecta por confianza
    assert not ok and "irreversible" in why
    ok, _, why = gate_decision(CFG, "rollback", 0.95, 0.9, 0.3, 0)        # hiperdirecta por tono
    assert not ok
    ok, _, _ = gate_decision(CFG, "rollback", 0.95, 0.9, 0.9, 0)
    assert ok
    ok, _, why = gate_decision(CFG, "restart", 0.99, 0.9, 1.0, 3)         # bloqueo tras 3
    assert not ok and "escal" in why
    assert gate_decision(CFG, "escalate_to_human", 0.1, 0.1, 0.05, 9)[0]


class Tool:
    def __init__(self, name): self.name = name


def ctx(pending, tone=1.0, blocks=0):
    return SimpleNamespace(function_call_id="c1", state={
        K_PENDING: {"c1": pending}, K_TONE: tone, K_INTERO: {"blocks": blocks},
        K_FEATURES: {"service": "api"}})


@pytest.mark.asyncio
async def test_before_tool_vetoes_and_counts():
    g = GatePlugin(CFG, EpisodicStore(":memory:"))
    c = ctx({"tool": "rollback", "args": {}, "expected": "resolves", "confidence": 0.5})
    r = await g.before_tool_callback(tool=Tool("rollback"), tool_args={}, tool_context=c)
    assert r["status"] == "vetoed"
    assert c.state["subcortex.metrics"]["vetoes"] == 1 and c.state["subcortex.metrics"]["vetoes_irreversible"] == 1
    c = ctx({"tool": "restart", "args": {}, "expected": "resolves", "confidence": 0.9})
    assert await g.before_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=c) is None
    assert await g.before_tool_callback(tool=Tool("inspect_service"), tool_args={}, tool_context=c) is None


def fc(name, conf):
    return types.Part(function_call=types.FunctionCall(
        name=name, args={"service": "api", "expected_effect": "resolves", "confidence": conf}))


@pytest.mark.asyncio
async def test_after_model_winner_take_all_and_llm_count():
    g = GatePlugin(CFG, EpisodicStore(":memory:"))
    c = SimpleNamespace(state={K_TONE: 1.0, K_FEATURES: {"service": "api"}})
    resp = LlmResponse(content=types.Content(role="model", parts=[
        fc("restart", 0.5), fc("restart", 0.9), types.Part(function_call=types.FunctionCall(name="inspect_service", args={}))]),
        usage_metadata=types.GenerateContentResponseUsageMetadata(total_token_count=123))
    out = await g.after_model_callback(callback_context=c, llm_response=resp)
    calls = out.get_function_calls()
    assert [x.name for x in calls] == ["restart", "inspect_service"]
    assert calls[0].args["confidence"] == 0.9
    m = c.state["subcortex.metrics"]
    assert m["llm_calls"] == 1 and m["tokens"] == 123 and m["vetoes"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_gate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'subcortex.gate'`

- [ ] **Step 3: Write minimal implementation**

```python
# subcortex/gate.py
"""Ganglios basales + cingulado: veto por defecto, desinhibir una sola acción."""
from __future__ import annotations

import logging

from google.adk.plugins.base_plugin import BasePlugin

from .config import SubcortexConfig
from .metrics import bump
from .store import EpisodicStore
from .types import K_FEATURES, K_INTERO, K_PENDING, K_TONE, PRED_PARAMS, Scene

log = logging.getLogger("subcortex")


def gate_decision(cfg: SubcortexConfig, tool: str, confidence: float, dopamine: float,
                  tone: float, consecutive_blocks: int) -> tuple[bool, float, str]:
    """Devuelve (autorizada, valor, motivo)."""
    if tool in cfg.always_allowed:
        return True, 1.0, "siempre permitida"
    risk = cfg.risk_of(tool)
    value = round(confidence * dopamine * tone - cfg.cost.get(risk, 0.0), 4)
    if consecutive_blocks >= cfg.max_consecutive_blocks:
        return False, value, ("demasiadas acciones bloqueadas seguidas: solo se permite escalar o cerrar")
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
            parts = list(llm_response.content.parts) if llm_response.content and llm_response.content.parts else []
            action_idx = [i for i, p in enumerate(parts)
                          if p.function_call and self.cfg.is_action(p.function_call.name)]
            if len(action_idx) <= 1:
                return None
            def score(i):
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_gate.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add subcortex/gate.py tests/unit/test_gate.py
git commit -m "feat(subcortex): GatePlugin con veto por defecto, vía hiperdirecta y winner-take-all"
```

---

### Task 8: MemoryPlugin

**Files:**
- Create: `subcortex/memory.py`
- Test: `tests/unit/test_memory.py`

**Interfaces:**
- Consumes: `EpisodicStore`, `Episode`, `Scene`, `K_LAST_ERROR`, `K_TONE`, `bump`, `SUCCESS_EFFECTS`.
- Produces: `render_precedents(episodes)->str`, `render_rules(rules)->str`, `MemoryPlugin(cfg, store)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_memory.py
from types import SimpleNamespace

import pytest

from subcortex.config import SubcortexConfig
from subcortex.memory import MemoryPlugin, render_precedents
from subcortex.store import EpisodicStore
from subcortex.types import K_FEATURES, K_LAST_ERROR, K_TONE, Rule, Scene

CFG = SubcortexConfig(risk={"restart": "costly"})
FEATS = {"service": "api", "symptom": "high_latency"}


class Tool:
    def __init__(self, name): self.name = name


def ctx(last_error, tone=1.0):
    return SimpleNamespace(function_call_id="c1", state={K_FEATURES: FEATS, K_LAST_ERROR: last_error, K_TONE: tone})


def le(err, observed="worsens", status="success"):
    return {"tool": "restart", "args": {"service": "api"}, "expected": "resolves", "observed": observed,
            "confidence": 1.0, "error": err, "status": status, "call_id": "c1"}


@pytest.mark.asyncio
async def test_writes_only_on_surprise_but_always_updates_dopamine():
    store = EpisodicStore(":memory:")
    m = MemoryPlugin(CFG, store)
    c = ctx(le(0.0, observed="resolves"))
    await m.after_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=c, result={"status": "success"})
    assert store.stats()["episodes"] == 0
    assert store.outcome_counts(Scene.from_features(FEATS).key, "restart") == (1, 0)
    c = ctx(le(-1.0))
    await m.after_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=c, result={"status": "success"})
    eps = store.all_episodes()
    assert len(eps) == 1 and eps[0].habenula and eps[0].strength == 1.0
    assert c.state["subcortex.metrics"]["episodes_written"] == 1


@pytest.mark.asyncio
async def test_skips_blocked_and_missing_error():
    store = EpisodicStore(":memory:")
    m = MemoryPlugin(CFG, store)
    await m.after_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=ctx(le(-1.0)),
                                result={"status": "vetoed"})
    await m.after_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=ctx(None),
                                result={"status": "success"})
    assert store.stats()["episodes"] == 0


@pytest.mark.asyncio
async def test_before_model_injects_precedents_and_rules_by_scene():
    store = EpisodicStore(":memory:")
    m = MemoryPlugin(CFG, store)
    c = ctx(le(-1.0))
    await m.after_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=c, result={"status": "success"})
    store.upsert_rule(Rule(scene_pattern={"symptom": "high_latency"}, tool="restart", text="restart suele empeorar", support=3))
    req = SimpleNamespace(appended=[])
    req.append_instructions = lambda xs: req.appended.extend(xs)
    assert await m.before_model_callback(callback_context=ctx(None, tone=0.5), llm_request=req) is None
    text = "\n".join(req.appended)
    assert "Precedentes" in text and "[FRACASO]" in text and "restart" in text
    assert "Reglas aprendidas" in text and "suele empeorar" in text
    req2 = SimpleNamespace(appended=[])
    req2.append_instructions = lambda xs: req2.appended.extend(xs)
    other = SimpleNamespace(state={K_FEATURES: {"service": "zzz", "symptom": "oom"}, K_TONE: 1.0})
    await m.before_model_callback(callback_context=other, llm_request=req2)
    assert req2.appended == []


def test_render_precedents_format():
    txt = render_precedents([])
    assert txt == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_memory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'subcortex.memory'`

- [ ] **Step 3: Write minimal implementation**

```python
# subcortex/memory.py
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
        lines.append(f"- {tag} incidente ({feats}): {e.tool}({args}) esperaba {e.expected}, resultó {e.observed}.")
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
                return None
            tone = float(state.get(K_TONE, 1.0))
            k = int(round(2 + 4 * tone))
            blocks = [render_precedents(self.store.recall(scene, k)), render_rules(self.store.rules_for(scene))]
            blocks = [b for b in blocks if b]
            if blocks:
                llm_request.append_instructions(blocks)
        except Exception:
            log.exception("memory.before_model")
        return None

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        if not self.cfg.is_action(tool.name):
            return None
        try:
            state = tool_context.state
            if (result or {}).get("status") in BLOCK_STATUSES:
                return None
            le = state.get(K_LAST_ERROR)
            if not le or le.get("tool") != tool.name:
                return None
            scene = self.cfg.scene_fn(state)
            if scene is None:
                return None
            observed = le["observed"]
            self.store.record_outcome(scene.key, tool.name, observed in SUCCESS_EFFECTS)
            err = float(le["error"])
            surprised = abs(err) >= self.cfg.surprise_threshold or le.get("status") == "error"
            if not surprised:
                return None
            self.store.write(Episode(
                scene_key=scene.key, features=scene.features, tool=tool.name, args=le["args"],
                expected=le["expected"], observed=observed, prediction_error=err, valence=err,
                habenula=err < 0 or le.get("status") == "error", strength=max(abs(err), 0.25)))
            bump(state, "episodes_written")
        except Exception:
            log.exception("memory.after_tool")
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_memory.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add subcortex/memory.py tests/unit/test_memory.py
git commit -m "feat(subcortex): MemoryPlugin con recuerdo por escena y escritura por sorpresa"
```

---

### Task 9: HabitPlugin

**Files:**
- Create: `subcortex/habit.py`
- Test: `tests/unit/test_habit.py`

**Interfaces:**
- Consumes: `EpisodicStore.get_habit/upsert_habit/outcome_counts`, `Habit`, `K_LAST_ERROR`, `K_ACTED`, `K_HABIT_HIT`, `bump`, `LlmResponse`, `types.FunctionCall`.
- Produces: `HabitPlugin(cfg, store)`, `habit_response(habit)->LlmResponse`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_habit.py
from types import SimpleNamespace

import pytest

from subcortex.config import SubcortexConfig
from subcortex.habit import HabitPlugin
from subcortex.store import EpisodicStore
from subcortex.types import K_ACTED, K_FEATURES, K_HABIT_HIT, K_LAST_ERROR, Habit, Scene

CFG = SubcortexConfig(risk={"restart": "costly", "rollback": "irreversible"})
FEATS = {"service": "api", "symptom": "oom"}
KEY = Scene.from_features(FEATS).key


class Tool:
    def __init__(self, name): self.name = name


def ctx(last_error=None, acted=False, habit_hit=False):
    return SimpleNamespace(function_call_id="c1", state={
        K_FEATURES: FEATS, K_LAST_ERROR: last_error, K_ACTED: acted, K_HABIT_HIT: habit_hit})


def le(tool, err, observed):
    return {"tool": tool, "args": {"service": "api"}, "expected": "resolves", "observed": observed,
            "confidence": 0.9, "error": err, "status": "success", "call_id": "c1"}


@pytest.mark.asyncio
async def test_compiles_after_three_clean_successes_never_irreversible():
    store = EpisodicStore(":memory:")
    h = HabitPlugin(CFG, store)
    for _ in range(3):
        store.record_outcome(KEY, "restart", True)
        store.record_outcome(KEY, "rollback", True)
        await h.after_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=ctx(le("restart", 0.0, "resolves")),
                                    result={"status": "success"})
        await h.after_tool_callback(tool=Tool("rollback"), tool_args={}, tool_context=ctx(le("rollback", 0.0, "resolves")),
                                    result={"status": "success"})
    hb = store.get_habit(KEY)
    assert hb and hb.tool == "restart" and hb.strength >= 0.8 and hb.typical_effect == "resolves"


@pytest.mark.asyncio
async def test_bypass_returns_function_call_once_per_episode():
    store = EpisodicStore(":memory:")
    store.upsert_habit(Habit(scene_key=KEY, tool="restart", args={"service": "api"}, typical_effect="resolves",
                             strength=0.9, successes=3, failures=0))
    h = HabitPlugin(CFG, store)
    c = ctx()
    resp = await h.before_model_callback(callback_context=c, llm_request=None)
    fc = resp.get_function_calls()[0]
    assert fc.name == "restart" and fc.args["expected_effect"] == "resolves" and fc.args["confidence"] == 0.9
    assert c.state[K_HABIT_HIT] is True and c.state["subcortex.metrics"]["habit_hits"] == 1
    assert await h.before_model_callback(callback_context=ctx(acted=True), llm_request=None) is None
    weak = EpisodicStore(":memory:")
    weak.upsert_habit(Habit(scene_key=KEY, tool="restart", args={}, typical_effect="resolves",
                            strength=0.5, successes=3, failures=1))
    assert await HabitPlugin(CFG, weak).before_model_callback(callback_context=ctx(), llm_request=None) is None


@pytest.mark.asyncio
async def test_dehabituation_on_negative_error_and_acted_flag():
    store = EpisodicStore(":memory:")
    store.upsert_habit(Habit(scene_key=KEY, tool="restart", args={"service": "api"}, typical_effect="resolves",
                             strength=0.9, successes=3, failures=0))
    h = HabitPlugin(CFG, store)
    c = ctx(le("restart", -0.8, "worsens"), habit_hit=True)
    await h.after_tool_callback(tool=Tool("restart"), tool_args={}, tool_context=c, result={"status": "success"})
    hb = store.get_habit(KEY)
    assert hb.strength == pytest.approx(0.45) and hb.failures == 1
    assert c.state[K_ACTED] is True and c.state[K_HABIT_HIT] is False
    assert c.state["subcortex.metrics"]["dehabituations"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_habit.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'subcortex.habit'`

- [ ] **Step 3: Write minimal implementation**

```python
# subcortex/habit.py
"""Caudado → putamen: decisiones repetidas se compilan y saltean al LLM."""
from __future__ import annotations

import logging

from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from .config import SubcortexConfig
from .metrics import bump
from .store import EpisodicStore
from .types import K_ACTED, K_HABIT_HIT, K_LAST_ERROR, SUCCESS_EFFECTS, Habit

log = logging.getLogger("subcortex")
BLOCK_STATUSES = {"vetoed", "rejected"}


def habit_response(habit: Habit) -> LlmResponse:
    args = {**habit.args, "expected_effect": habit.typical_effect, "confidence": round(habit.strength, 3)}
    part = types.Part(function_call=types.FunctionCall(name=habit.tool, args=args))
    return LlmResponse(content=types.Content(role="model", parts=[part]),
                       custom_metadata={"subcortex": "habit"})


class HabitPlugin(BasePlugin):
    def __init__(self, cfg: SubcortexConfig, store: EpisodicStore) -> None:
        super().__init__(name="subcortex_habit")
        self.cfg = cfg
        self.store = store

    async def before_model_callback(self, *, callback_context, llm_request):
        try:
            state = callback_context.state
            if state.get(K_ACTED):
                return None
            scene = self.cfg.scene_fn(state)
            if scene is None:
                return None
            habit = self.store.get_habit(scene.key)
            if not habit or habit.strength < self.cfg.habit_min_strength:
                return None
            if self.cfg.risk_of(habit.tool) == "irreversible":
                return None
            state[K_HABIT_HIT] = True
            bump(state, "habit_hits")
            log.info("hábito: %s en escena %s (fuerza %.2f), sin LLM", habit.tool, scene.key, habit.strength)
            return habit_response(habit)
        except Exception:
            log.exception("habit.before_model")
            return None

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        if not self.cfg.is_action(tool.name):
            return None
        try:
            state = tool_context.state
            if (result or {}).get("status") in BLOCK_STATUSES:
                return None
            state[K_ACTED] = True
            le = state.get(K_LAST_ERROR)
            if not le or le.get("tool") != tool.name:
                return None
            scene = self.cfg.scene_fn(state)
            if scene is None:
                return None
            err = float(le["error"])
            habit = self.store.get_habit(scene.key)
            if state.get(K_HABIT_HIT) and habit and habit.tool == tool.name:
                state[K_HABIT_HIT] = False
                if err < 0:
                    habit.strength = round(habit.strength * 0.5, 4)
                    habit.failures += 1
                    self.store.upsert_habit(habit)
                    bump(state, "dehabituations")
                    log.info("des-habituación: %s cae a %.2f", habit.tool, habit.strength)
                    return None
            if le["observed"] not in SUCCESS_EFFECTS or self.cfg.risk_of(tool.name) == "irreversible":
                return None
            s, f = self.store.outcome_counts(scene.key, tool.name)
            if s >= self.cfg.habit_min_successes and f == 0:
                strength = min(1.0, 0.8 + 0.05 * (s - self.cfg.habit_min_successes))
                self.store.upsert_habit(Habit(scene_key=scene.key, tool=tool.name, args=le["args"],
                                              typical_effect=le["observed"], strength=strength,
                                              successes=s, failures=f))
        except Exception:
            log.exception("habit.after_tool")
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_habit.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add subcortex/habit.py tests/unit/test_habit.py
git commit -m "feat(subcortex): HabitPlugin con compilación, bypass del LLM y des-habituación"
```

---

### Task 10: consolidate()

**Files:**
- Create: `subcortex/consolidate.py`
- Test: `tests/unit/test_consolidate.py`

**Interfaces:**
- Consumes: `EpisodicStore`, `Episode`, `Rule`.
- Produces: `consolidate(store, now=None, min_support=3)->dict` con claves `decayed, pruned, reinforced, rules`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_consolidate.py
from subcortex.consolidate import consolidate
from subcortex.store import EpisodicStore
from subcortex.types import Episode, Scene

DAY = 86400


def ep(feats, tool="restart", err=-0.8, created=0.0, access=0, last=0.0):
    s = Scene.from_features(feats)
    return Episode(scene_key=s.key, features=s.features, tool=tool, args={}, expected="resolves",
                   observed="worsens" if err < 0 else "resolves", prediction_error=err, valence=err,
                   habenula=err < 0, strength=abs(err), access_count=access, created_at=created, last_access=last)


def test_decay_prune_reinforce():
    store = EpisodicStore(":memory:")
    old_unused = store.write(ep({"a": "1"}, err=-0.1, created=0, last=0))          # 0.1*0.9^30 < 0.05 y >7 días
    popular = store.write(ep({"a": "2"}, err=-0.5, access=3, last=29 * DAY))       # refuerzo
    stats = consolidate(store, now=30 * DAY)
    ids = {e.id: e for e in store.all_episodes()}
    assert old_unused not in ids and stats["pruned"] == 1
    assert ids[popular].strength > 0.5 and stats["reinforced"] == 1


def test_rules_distilled_from_recurrent_pattern():
    store = EpisodicStore(":memory:")
    for svc in ("api", "auth", "search"):
        store.write(ep({"service": svc, "symptom": "high_latency", "traffic": "normal"}, last=0))
    store.write(ep({"service": "api", "symptom": "oom", "traffic": "normal"}, err=0.5, last=0))
    stats = consolidate(store, now=0)
    rules = store.rules_for(Scene.from_features({"service": "zzz", "symptom": "high_latency", "traffic": "normal"}))
    assert stats["rules"] == 1 and len(rules) == 1
    assert rules[0].tool == "restart" and "worsens" in rules[0].text and rules[0].support == 3
    assert "service" not in rules[0].scene_pattern
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_consolidate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'subcortex.consolidate'`

- [ ] **Step 3: Write minimal implementation**

```python
# subcortex/consolidate.py
"""Sueño / microglía: poda, refuerzo y destilación episódico → semántico. Sin LLM."""
from __future__ import annotations

import time
from collections import defaultdict

from .store import EpisodicStore
from .types import Episode, Rule

DAY = 86400.0


def consolidate(store: EpisodicStore, now: float | None = None, min_support: int = 3) -> dict:
    now = time.time() if now is None else now
    stats = {"decayed": 0, "pruned": 0, "reinforced": 0, "rules": 0}
    to_delete: list[int] = []
    for e in store.all_episodes():
        days_idle = max(0.0, (now - e.last_access) / DAY)
        strength = e.strength * (0.9 ** days_idle)
        if e.access_count >= 3:
            strength = min(1.0, strength * 1.2)
            stats["reinforced"] += 1
        if strength < 0.05 and e.access_count == 0 and (now - e.created_at) > 7 * DAY:
            to_delete.append(e.id)
            continue
        if strength != e.strength:
            store.update_strength(e.id, round(strength, 4))
            stats["decayed"] += 1
    store.delete_episodes(to_delete)
    stats["pruned"] = len(to_delete)
    stats["rules"] = _distill(store, min_support)
    return stats


def _distill(store: EpisodicStore, min_support: int) -> int:
    groups: dict[tuple[str, str], list[Episode]] = defaultdict(list)
    for e in store.all_episodes():
        groups[(e.tool, e.observed)].append(e)
    n = 0
    for (tool, observed), eps in groups.items():
        if len(eps) < min_support:
            continue
        common = dict(eps[0].features.items())
        for e in eps[1:]:
            common = {k: v for k, v in common.items() if e.features.get(k) == v}
        if not common:
            continue
        desc = ", ".join(f"{k}={v}" for k, v in sorted(common.items()))
        text = f"En incidentes con {desc}, {tool} tiende a {observed} (n={len(eps)})."
        store.upsert_rule(Rule(scene_pattern=common, tool=tool, text=text, support=len(eps)))
        n += 1
    return n
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_consolidate.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add subcortex/consolidate.py tests/unit/test_consolidate.py
git commit -m "feat(subcortex): consolidate() con decaimiento, poda, refuerzo y reglas"
```

---

### Task 11: attach() e integración con el Runner real de ADK

**Files:**
- Modify: `subcortex/__init__.py`
- Create: `tests/integration/fake_llm.py`, `tests/integration/test_loop.py`

**Interfaces:**
- Consumes: todos los plugins, `EpisodicStore`, `wrap_action_tool`, `SubcortexConfig`.
- Produces: `attach(app, *, risk, diagnostic_tools=(), store_path=":memory:", scene_fn=None, config=None)->Subcortex`; `Subcortex` con `.store`, `.config`, `.plugins`, `.consolidate(now=None)`. `ScriptedLlm(BaseLlm)` con `script: list[LlmResponse]`, `.calls`.

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/fake_llm.py
from __future__ import annotations

from typing import AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types


def call(name: str, **args) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(name=name, args=args))]))


def text(msg: str) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=msg)]))


class ScriptedLlm(BaseLlm):
    """LLM falso: devuelve respuestas programadas en orden y registra las instrucciones recibidas."""
    model: str = "scripted"
    script: list[LlmResponse] = []
    calls: int = 0
    seen_instructions: list[str] = []

    @classmethod
    def supported_models(cls) -> list[str]:
        return ["scripted"]

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False
                                     ) -> AsyncGenerator[LlmResponse, None]:
        self.calls += 1
        si = llm_request.config.system_instruction if llm_request.config else None
        self.seen_instructions.append(si if isinstance(si, str) else str(si))
        idx = min(self.calls - 1, len(self.script) - 1)
        yield self.script[idx]
```

```python
# tests/integration/test_loop.py
import pytest
from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import subcortex
from opsworld.tools import ALL_TOOLS, registry
from opsworld.world import DIAGNOSTIC_TOOLS, RISK, Incident, World, features
from subcortex.types import K_FEATURES, Habit, Scene
from tests.integration.fake_llm import ScriptedLlm, call, text


def incident(cause="memory_leak"):
    return Incident(id=1, service="api", cause=cause, recent_deploy=False, traffic="normal", hour_bucket="night")


async def run_episode(llm, inc, store_path=":memory:", sc=None):
    agent = LlmAgent(name="ops", model=llm, instruction="Sos operador de guardia.", tools=list(ALL_TOOLS))
    app = App(name="t", root_agent=agent)
    sc = sc or subcortex.attach(app, risk=RISK, diagnostic_tools=DIAGNOSTIC_TOOLS, store_path=store_path)
    svc = InMemorySessionService()
    session = await svc.create_session(app_name="t", user_id="u", state={K_FEATURES: features(inc)})
    world = World(inc)
    registry.register(session.id, world)
    runner = Runner(app=app, session_service=svc)
    msg = types.Content(role="user", parts=[types.Part(text="Incidente: revisá y actuá.")])
    async for _ in runner.run_async(user_id="u", session_id=session.id, new_message=msg):
        pass
    session = await svc.get_session(app_name="t", user_id="u", session_id=session.id)
    return world, session.state, sc


@pytest.mark.asyncio
async def test_authorized_action_runs_and_protocol_is_injected():
    llm = ScriptedLlm(script=[call("restart", service="api", expected_effect="resolves", confidence=0.9), text("listo")])
    world, state, sc = await run_episode(llm, incident("memory_leak"))
    assert world.resolved and llm.calls == 2
    assert "Protocolo de acción" in llm.seen_instructions[0] and "Estado interno" in llm.seen_instructions[0]
    assert state["subcortex.metrics"]["llm_calls"] == 2 and state["subcortex.metrics"]["steps"] == 1


@pytest.mark.asyncio
async def test_veto_makes_llm_reiterate_and_tool_not_executed():
    llm = ScriptedLlm(script=[call("rollback", service="api", expected_effect="resolves", confidence=0.4),
                              call("escalate_to_human", reason="x", expected_effect="no_change", confidence=0.9),
                              text("escalado")])
    world, state, _ = await run_episode(llm, incident("bad_deploy"))
    assert world.steps == 1 and world.log[0]["action"] == "escalate_to_human"
    assert llm.calls == 3 and state["subcortex.metrics"]["vetoes_irreversible"] == 1


@pytest.mark.asyncio
async def test_missing_prediction_is_rejected():
    llm = ScriptedLlm(script=[call("restart", service="api"), text("ok")])
    world, state, _ = await run_episode(llm, incident("memory_leak"))
    assert world.steps == 0 and state["subcortex.metrics"]["rejected"] == 1


@pytest.mark.asyncio
async def test_surprise_writes_episode_with_habenula():
    llm = ScriptedLlm(script=[call("restart", service="api", expected_effect="resolves", confidence=1.0), text("uh")])
    world, state, sc = await run_episode(llm, incident("db_saturated"))
    eps = sc.store.all_episodes()
    assert len(eps) == 1 and eps[0].habenula and eps[0].observed == "worsens"
    assert state["subcortex.metrics"]["episodes_written"] == 1


@pytest.mark.asyncio
async def test_habit_bypasses_llm():
    inc = incident("memory_leak")
    llm = ScriptedLlm(script=[text("fin")])
    agent = LlmAgent(name="ops", model=llm, instruction="x", tools=list(ALL_TOOLS))
    app = App(name="t", root_agent=agent)
    sc = subcortex.attach(app, risk=RISK, diagnostic_tools=DIAGNOSTIC_TOOLS)
    key = Scene.from_features(features(inc)).key
    sc.store.upsert_habit(Habit(scene_key=key, tool="restart", args={"service": "api"}, typical_effect="resolves",
                                strength=0.9, successes=3, failures=0))
    world, state, _ = await run_episode(llm, inc, sc=sc)
    assert world.resolved and world.log[0]["action"] == "restart"
    assert llm.calls == 1 and state["subcortex.metrics"]["habit_hits"] == 1
    assert state["subcortex.metrics"]["llm_calls"] == 1


@pytest.mark.asyncio
async def test_parallel_actions_reduced_to_one():
    from google.adk.models.llm_response import LlmResponse
    two = LlmResponse(content=types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(name="restart", args={"service": "api", "expected_effect": "resolves", "confidence": 0.5})),
        types.Part(function_call=types.FunctionCall(name="scale", args={"service": "api", "replicas": 4, "expected_effect": "resolves", "confidence": 0.9})),
    ]))
    llm = ScriptedLlm(script=[two, text("ok")])
    world, state, _ = await run_episode(llm, incident("traffic_spike"))
    assert world.steps == 1 and world.log[0]["action"] == "scale" and world.resolved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration -v`
Expected: FAIL with `AttributeError: module 'subcortex' has no attribute 'attach'`

- [ ] **Step 3: Write minimal implementation**

```python
# subcortex/__init__.py
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
```

Crear también `tests/__init__.py` y `tests/integration/__init__.py` vacíos para que `from tests.integration.fake_llm import ...` resuelva.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests -v`
Expected: PASS (todos). Si `agent.tools = new_tools` falla por validación pydantic, usar `object.__setattr__(agent, "tools", new_tools)`. Si `App.plugins` no acepta `extend` tras construcción, reconstruir: `app.plugins = list(app.plugins) + plugins`. Si el flow no ejecuta la function call devuelta por `before_model` (test `test_habit_bypasses_llm`), verificar en `base_llm_flow._run_one_step_async` cómo se postprocesa el `LlmResponse` y ajustar `habit_response` (por ejemplo, `turn_complete=True`).

- [ ] **Step 5: Commit**

```bash
git add subcortex/__init__.py tests/__init__.py tests/integration
git commit -m "feat(subcortex): attach() y tests de integración con el Runner real de ADK"
```

---

### Task 12: Demo — agente, corrida A/B y README

**Files:**
- Create: `demo/__init__.py`, `demo/instruction.md`, `demo/agent.py`, `demo/run_ab.py`, `README.md`, `.env.example`
- Test: `tests/unit/test_demo.py`

**Interfaces:**
- Consumes: `attach`, `ALL_TOOLS`, `RISK`, `DIAGNOSTIC_TOOLS`, `registry`, `World`, `generate_incidents`, `features`, `get_metrics`.
- Produces: `build_agent(model)->LlmAgent`, `build_app(with_subcortex, store_path, model)->(App, Subcortex|None)`, `root_agent` (para `adk web demo`), `run_variant(name, incidents, with_subcortex, model, consolidate_every)->list[dict]`, `summarize(rows)->dict`, `main()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_demo.py
from demo.run_ab import summarize


def test_summarize_by_thirds():
    rows = [{"score": s, "llm_calls": c, "steps": 2, "vetoes": 0, "vetoes_irreversible": 0, "worsens": 0,
             "abs_error_sum": 0.5, "error_count": 1, "episodes_written": 1, "habit_hits": 0,
             "dehabituations": 0, "tokens": 10, "resolved": True}
            for s, c in zip([10, 20, 30, 40, 50, 60], [3, 3, 3, 2, 2, 1])]
    s = summarize(rows)
    assert s["overall"]["score_mean"] == 35 and s["thirds"][0]["score_mean"] == 15
    assert s["thirds"][2]["llm_calls_mean"] == 1.5 and s["overall"]["mean_abs_error"] == 0.5
    assert s["overall"]["resolved_rate"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_demo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'demo'`

- [ ] **Step 3: Write minimal implementation**

```markdown
<!-- demo/instruction.md -->
Sos el operador de guardia de una plataforma. Te llega un incidente con sus síntomas observables.
Tu objetivo es resolverlo con el menor costo y sin causar daño.

Herramientas de diagnóstico (gratis): inspect_service, check_deploys.
Herramientas de acción: restart (costosa), scale (costosa), rollback (irreversible), failover_db (irreversible),
resolve (cierra el incidente; cerrarlo sin resolverlo es una falta grave), escalate_to_human (termina tu intervención).

Reglas:
- Diagnosticá antes de actuar si no estás seguro.
- No repitas una acción que no cambió nada.
- Si el problema es una dependencia externa, escalá.
- Cuando el incidente esté resuelto, o si es una falsa alarma, llamá a resolve.
Tenés como máximo 8 llamadas a herramientas.
```

```python
# demo/agent.py
"""Agente operador. `root_agent` lleva subcortex para `adk web demo`."""
from __future__ import annotations

import os
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.apps.app import App

import subcortex
from opsworld.tools import ALL_TOOLS
from opsworld.world import DIAGNOSTIC_TOOLS, RISK

MODEL = os.environ.get("SUBCORTEX_MODEL", "gemini-3-flash-preview")
INSTRUCTION = (Path(__file__).parent / "instruction.md").read_text()


def build_agent(model: str | object = MODEL) -> LlmAgent:
    return LlmAgent(name="ops_operator", model=model, description="Operador de guardia",
                    instruction=INSTRUCTION, tools=list(ALL_TOOLS))


def build_app(with_subcortex: bool, store_path: str = ":memory:", model: str | object = MODEL):
    app = App(name="opsworld", root_agent=build_agent(model))
    sc = subcortex.attach(app, risk=RISK, diagnostic_tools=DIAGNOSTIC_TOOLS, store_path=store_path) if with_subcortex else None
    return app, sc


root_agent = build_agent()
```

```python
# demo/run_ab.py
"""Corre el mismo agente con y sin subcortex sobre la misma secuencia de incidentes."""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from demo.agent import MODEL, build_app
from opsworld.tools import registry
from opsworld.world import World, features, generate_incidents
from subcortex.metrics import get_metrics
from subcortex.types import K_FEATURES

NUM_KEYS = ("score", "llm_calls", "tokens", "steps", "vetoes", "vetoes_irreversible", "worsens",
            "episodes_written", "habit_hits", "dehabituations")


async def run_variant(name: str, incidents, with_subcortex: bool, model=MODEL, consolidate_every: int = 0):
    app, sc = build_app(with_subcortex, model=model)
    svc = InMemorySessionService()
    runner = Runner(app=app, session_service=svc)
    rows = []
    for i, inc in enumerate(incidents):
        session = await svc.create_session(app_name="opsworld", user_id="demo", state={K_FEATURES: features(inc)})
        world = World(inc)
        registry.register(session.id, world)
        f = features(inc)
        prompt = (f"Incidente #{inc.id}: servicio={f['service']}, síntoma={f['symptom']}, "
                  f"deploy_reciente={f['recent_deploy']}, tráfico={f['traffic']}, franja={f['hour_bucket']}. Actuá.")
        msg = types.Content(role="user", parts=[types.Part(text=prompt)])
        t0 = time.time()
        llm_calls_fallback = 0
        async for ev in runner.run_async(user_id="demo", session_id=session.id, new_message=msg):
            if ev.author != "user" and ev.get_function_calls() and not ev.partial:
                llm_calls_fallback += 1
        session = await svc.get_session(app_name="opsworld", user_id="demo", session_id=session.id)
        m = get_metrics(session.state) if with_subcortex else {}
        row = {"variant": name, "i": i, "cause": inc.cause, "score": world.score, "resolved": world.resolved,
               "steps": world.steps, "worsens": world.worsens, "seconds": round(time.time() - t0, 1),
               "llm_calls": m.get("llm_calls", llm_calls_fallback + 1), "tokens": m.get("tokens", 0),
               "vetoes": m.get("vetoes", 0), "vetoes_irreversible": m.get("vetoes_irreversible", 0),
               "episodes_written": m.get("episodes_written", 0), "habit_hits": m.get("habit_hits", 0),
               "dehabituations": m.get("dehabituations", 0), "abs_error_sum": m.get("abs_error_sum", 0.0),
               "error_count": m.get("error_count", 0), "actions": [e["action"] for e in world.log]}
        rows.append(row)
        print(f"[{name}] #{i:02d} {inc.cause:15s} score={world.score:5d} steps={world.steps} "
              f"llm={row['llm_calls']} vetoes={row['vetoes']} habit={row['habit_hits']} acciones={row['actions']}")
        if sc and consolidate_every and (i + 1) % consolidate_every == 0:
            print(f"[{name}] consolidate → {sc.consolidate()}")
    if sc:
        print(f"[{name}] store: {sc.store.stats()}")
    return rows


def _mean(rows, key):
    return round(statistics.mean(r[key] for r in rows), 2) if rows else 0.0


def _block(rows):
    err_n = sum(r["error_count"] for r in rows)
    return {
        "n": len(rows),
        "score_mean": _mean(rows, "score"),
        "resolved_rate": round(sum(1 for r in rows if r["resolved"]) / len(rows), 2) if rows else 0.0,
        "llm_calls_mean": _mean(rows, "llm_calls"),
        "tokens_mean": _mean(rows, "tokens"),
        "steps_mean": _mean(rows, "steps"),
        "worsens_total": sum(r["worsens"] for r in rows),
        "vetoes_total": sum(r["vetoes"] for r in rows),
        "vetoes_irreversible_total": sum(r["vetoes_irreversible"] for r in rows),
        "episodes_written_total": sum(r["episodes_written"] for r in rows),
        "habit_hits_total": sum(r["habit_hits"] for r in rows),
        "dehabituations_total": sum(r["dehabituations"] for r in rows),
        "mean_abs_error": round(sum(r["abs_error_sum"] for r in rows) / err_n, 3) if err_n else 0.0,
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    cut = max(1, n // 3)
    thirds = [rows[:cut], rows[cut:2 * cut], rows[2 * cut:]]
    return {"overall": _block(rows), "thirds": [_block(t) for t in thirds]}


def print_table(summaries: dict[str, dict]) -> None:
    keys = ["score_mean", "resolved_rate", "llm_calls_mean", "tokens_mean", "steps_mean", "worsens_total",
            "vetoes_total", "vetoes_irreversible_total", "episodes_written_total", "habit_hits_total",
            "dehabituations_total", "mean_abs_error"]
    names = list(summaries)
    print("\n== Resumen general ==")
    print(f"{'métrica':28s}" + "".join(f"{n:>14s}" for n in names))
    for k in keys:
        print(f"{k:28s}" + "".join(f"{summaries[n]['overall'][k]:>14}" for n in names))
    print("\n== Por tercios (primero → último) ==")
    for k in ("score_mean", "llm_calls_mean", "mean_abs_error", "habit_hits_total", "worsens_total"):
        for n in names:
            vals = [t[k] for t in summaries[n]["thirds"]]
            print(f"{k:28s}{n:>12s}: {vals}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--consolidate-every", type=int, default=10)
    ap.add_argument("--only", choices=["baseline", "subcortex"], default=None)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()
    incidents = generate_incidents(args.n, args.seed)
    results, summaries = {}, {}
    for name, flag in (("baseline", False), ("subcortex", True)):
        if args.only and args.only != name:
            continue
        registry.clear()
        rows = await run_variant(name, incidents, flag, consolidate_every=args.consolidate_every if flag else 0)
        results[name] = rows
        summaries[name] = summarize(rows)
    print_table(summaries)
    Path(args.out).write_text(json.dumps({"rows": results, "summary": summaries}, indent=2, ensure_ascii=False))
    print(f"\nGuardado en {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
```

```
# .env.example
GOOGLE_API_KEY=tu_api_key
GOOGLE_GENAI_USE_VERTEXAI=False
```

README con: qué es, mapeo componente ↔ paso del ensayo (tabla de la spec §1), instalación (`uv sync`), tests (`uv run pytest`), demo (`uv run python -m demo.run_ab --n 40`), `adk web demo`, y cómo usar `subcortex.attach()` en un agente propio (snippet de §4.1 de la spec).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests -v && uv run python -c "import demo.agent"`
Expected: PASS y el import no falla.

- [ ] **Step 5: Commit**

```bash
git add demo README.md .env.example tests/unit/test_demo.py
git commit -m "feat(demo): agente operador, corrida A/B y README"
```

---

### Task 13: Corrida real con Gemini y reporte

**Files:**
- Create: `results.json` (salida), `docs/superpowers/results/2026-08-30-ab-run.md`

- [ ] **Step 1: Verificar credenciales**

Run: `test -n "$GOOGLE_API_KEY" && echo ok || echo "falta GOOGLE_API_KEY"`
Si falta: pedirla al usuario (o `.env` en `demo/`) y no inventar resultados.

- [ ] **Step 2: Corrida corta de humo**

Run: `uv run python -m demo.run_ab --n 6 --only subcortex --out smoke.json`
Expected: 6 líneas de episodio sin excepciones; al menos un `rejected`/`vetoed` o acción autorizada visible.
Si Gemini devuelve 404: es `GOOGLE_CLOUD_LOCATION`/proveedor, no el nombre del modelo — no cambiar el modelo.

- [ ] **Step 3: Corrida A/B completa**

Run: `uv run python -m demo.run_ab --n 40 --seed 42 --consolidate-every 10`
Expected: tabla comparativa y `results.json`.

- [ ] **Step 4: Reporte**

Escribir `docs/superpowers/results/2026-08-30-ab-run.md` con la tabla tal cual salió, la lectura honesta (qué mejoró, qué no, qué hipótesis quedan) y los siguientes pasos. No maquillar: si el baseline gana en alguna métrica, decirlo.

- [ ] **Step 5: Commit**

```bash
git add results.json docs/superpowers/results
git commit -m "docs: resultados de la corrida A/B baseline vs subcortex"
```
