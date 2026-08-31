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
