"""Tools ADK de marketworld. Cada sesión tiene su MarketWorld (una semana) en el registro."""
from __future__ import annotations

from google.adk.tools.tool_context import ToolContext

from opsworld.tools import WorldRegistry

from .world import MarketWorld

registry = WorldRegistry()
_NO_WORLD = {"status": "error", "observed_effect": "no_change",
             "message": "No hay semana activa para esta sesión."}


def _world(tool_context: ToolContext) -> MarketWorld | None:
    return registry.get(tool_context.session.id)


def market_snapshot(tool_context: ToolContext) -> dict:
    """Ranking de los 10 majors por retorno 30d, retorno 7d, si cotizan sobre su SMA-100, el
    régimen actual (tendencia de BTC, amplitud, volatilidad, dispersión) y el target de la regla momentum."""
    w = _world(tool_context)
    return w.diagnose("market_snapshot") if w else _NO_WORLD


def asset_detail(symbol: str, tool_context: ToolContext) -> dict:
    """Detalle de un símbolo (p. ej. "SOL/USDT"): retornos 7/30/90d, volatilidad anualizada 30d
    y distancia a su SMA-100."""
    w = _world(tool_context)
    return w.diagnose("asset_detail", symbol=symbol) if w else _NO_WORLD


def portfolio(tool_context: ToolContext) -> dict:
    """Cartera actual: cash, posiciones en USD, equity y costos acumulados."""
    w = _world(tool_context)
    return w.diagnose("portfolio") if w else _NO_WORLD


def follow_momentum(tool_context: ToolContext) -> dict:
    """Aplica la regla de la casa: top-2 por retorno 30d entre los que cotizan sobre su SMA-100,
    pesos iguales (mitad en cash si califica uno solo; todo cash si ninguno). Costo 0.15 % por lado."""
    w = _world(tool_context)
    return w.act("follow_momentum") if w else _NO_WORLD


def hold(tool_context: ToolContext) -> dict:
    """Mantiene la cartera tal como está esta semana, sin operar ni pagar costos."""
    w = _world(tool_context)
    return w.act("hold") if w else _NO_WORLD


def go_cash(tool_context: ToolContext) -> dict:
    """Vende todo y queda en cash esta semana. Costo 0.15 % de lo vendido."""
    w = _world(tool_context)
    return w.act("go_cash") if w else _NO_WORLD


def rotate(symbols: list[str], tool_context: ToolContext) -> dict:
    """Lleva la cartera a 1 o 2 símbolos del universo, pesos iguales (p. ej. ["SOL/USDT", "LINK/USDT"]).
    Costo 0.15 % por lado sobre lo que cambia."""
    w = _world(tool_context)
    return w.act("rotate", symbols=symbols) if w else _NO_WORLD


ACTION_TOOLS = [follow_momentum, hold, go_cash, rotate]
DIAG_TOOLS = [market_snapshot, asset_detail, portfolio]
ALL_TOOLS = ACTION_TOOLS + DIAG_TOOLS
