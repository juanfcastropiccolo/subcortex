"""Tools ADK de bugworld. Cada sesión tiene su BugWorld en el registro."""
from __future__ import annotations

from google.adk.tools.tool_context import ToolContext

from opsworld.tools import WorldRegistry

from .world import BugWorld

registry = WorldRegistry()
_NO_WORLD = {"status": "error", "observed_effect": "no_change",
             "message": "No hay episodio activo para esta sesión."}


def _world(tool_context: ToolContext) -> BugWorld | None:
    return registry.get(tool_context.session.id)


def run_tests(pattern: str, tool_context: ToolContext) -> dict:
    """Corre la suite (o un subconjunto con -k pattern; usá "" para toda la suite). Devuelve
    cuántos pasan/fallan, los ids que fallan, el primer fallo y un diagnóstico (finding)."""
    w = _world(tool_context)
    return w.diagnose("run_tests", pattern=pattern) if w else _NO_WORLD


def read_file(path: str, start: int, end: int, tool_context: ToolContext) -> dict:
    """Lee líneas [start, end] (1-based, máx. 120) de un archivo .py relativo al repo, con números de línea."""
    w = _world(tool_context)
    return w.diagnose("read_file", path=path, start=start, end=end) if w else _NO_WORLD


def search(pattern: str, tool_context: ToolContext) -> dict:
    """Busca una regex en todos los .py del repo (código y tests). Hasta 20 coincidencias."""
    w = _world(tool_context)
    return w.diagnose("search", pattern=pattern) if w else _NO_WORLD


def edit_file(path: str, old: str, new: str, tool_context: ToolContext) -> dict:
    """Reemplaza `old` por `new` en el archivo (old debe aparecer exactamente una vez). Costosa:
    después de aplicar se corre la suite completa y se informa el efecto."""
    w = _world(tool_context)
    return w.act("edit_file", path=path, old=old, new=new) if w else _NO_WORLD


def rewrite_file(path: str, content: str, tool_context: ToolContext) -> dict:
    """Sobrescribe el archivo completo. Irreversible y cara: podés destruir el módulo."""
    w = _world(tool_context)
    return w.act("rewrite_file", path=path, content=content) if w else _NO_WORLD


def revert_file(path: str, tool_context: ToolContext) -> dict:
    """Descarta tus cambios en ese archivo y lo deja como estaba al empezar."""
    w = _world(tool_context)
    return w.act("revert_file", path=path) if w else _NO_WORLD


def finish(tool_context: ToolContext) -> dict:
    """Cierra el episodio declarando el bug arreglado. Cerrar con tests fallando es una falta grave."""
    w = _world(tool_context)
    return w.act("finish") if w else _NO_WORLD


def escalate_to_human(reason: str, tool_context: ToolContext) -> dict:
    """Escala a un humano y termina tu intervención. Correcto si el problema está en el test, no en el código."""
    w = _world(tool_context)
    return w.act("escalate_to_human", reason=reason) if w else _NO_WORLD


ACTION_TOOLS = [edit_file, rewrite_file, revert_file, finish, escalate_to_human]
DIAG_TOOLS = [run_tests, read_file, search]
ALL_TOOLS = ACTION_TOOLS + DIAG_TOOLS
