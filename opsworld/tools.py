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
