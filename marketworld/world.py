"""Replay histórico del Proyecto Momentum: 10 majors, velas diarias reales, decisiones semanales.

Cada episodio es un día de decisión `t`. El agente ve el régimen (no la fecha), decide una
acción sobre la cartera y el resultado es el retorno de la cartera resultante 7 días después,
costos incluidos. La cartera persiste entre episodios: es un replay secuencial, no 40 muestras.
"""
from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).parent / "data"
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
           "DOGE/USDT", "ADA/USDT", "LINK/USDT", "AVAX/USDT", "LTC/USDT"]
TOP_K = 2
LOOKBACK = 30
TREND_W = 100
COST = 0.0015
HORIZON = 21  # = SPACING: la consecuencia de una decisión se mide hasta la decisión siguiente
MAX_STEPS = 5
VOL_HIST = 180  # días de historia para la mediana de volatilidad del régimen

RISK = {"follow_momentum": "costly", "rotate": "costly", "go_cash": "costly", "hold": "free"}
DIAGNOSTIC_TOOLS = frozenset({"market_snapshot", "asset_detail", "portfolio"})
ALWAYS_ALLOWED = frozenset({"hold"})
COARSE_FEATURES = ("finding", "holding")

RESOLVE_T = 0.06   # retorno a 21 días > +6 % → resolves
WORSEN_T = -0.06   # < −6 % → worsens


def _sym_file(sym: str) -> Path:
    return DATA / f"binance_{sym.replace('/', '-')}_1d_1095d.csv"


class Market:
    """Cierres diarios alineados por fecha para los 10 símbolos."""

    def __init__(self) -> None:
        per: dict[str, dict[str, float]] = {}
        for s in SYMBOLS:
            with open(_sym_file(s)) as f:
                per[s] = {row["timestamp"]: float(row["close"]) for row in csv.DictReader(f)}
        common = sorted(set.intersection(*(set(d) for d in per.values())))
        self.dates = common
        self.closes = {s: [per[s][d] for d in common] for s in SYMBOLS}
        self.n = len(common)

    def price(self, s: str, t: int) -> float:
        return self.closes[s][t]

    def ret(self, s: str, t: int, n: int) -> float:
        return self.closes[s][t] / self.closes[s][t - n] - 1.0

    def sma(self, s: str, t: int, w: int = TREND_W) -> float:
        xs = self.closes[s][t - w + 1:t + 1]
        return sum(xs) / len(xs)

    def vol(self, s: str, t: int, n: int = LOOKBACK) -> float:
        xs = self.closes[s]
        lr = [math.log(xs[i] / xs[i - 1]) for i in range(t - n + 1, t + 1)]
        return statistics.pstdev(lr) * math.sqrt(365)

    def above_sma(self, s: str, t: int) -> bool:
        return self.closes[s][t] > self.sma(s, t)

    def momentum_target(self, t: int) -> list[str]:
        """La regla del proyecto: top-2 por retorno 30d, solo los que cotizan sobre su SMA-100."""
        ranked = sorted(SYMBOLS, key=lambda s: self.ret(s, t, LOOKBACK), reverse=True)
        return [s for s in ranked if self.above_sma(s, t)][:TOP_K]

    def regime(self, t: int) -> dict[str, str]:
        """Features observables del día: nunca la fecha ni el futuro."""
        btc_trend = "up" if self.above_sma("BTC/USDT", t) else "down"
        breadth_n = sum(self.above_sma(s, t) for s in SYMBOLS)
        breadth = "low" if breadth_n <= 3 else ("mid" if breadth_n <= 6 else "high")
        assert t >= VOL_HIST + LOOKBACK, f"t={t} sin historia suficiente para el régimen de volatilidad"
        v = self.vol("BTC/USDT", t)
        hist = [self.vol("BTC/USDT", u) for u in range(t - VOL_HIST, t, 7)]
        vol = "high" if v > statistics.median(hist) else "low"
        rets = sorted((self.ret(s, t, LOOKBACK) for s in SYMBOLS), reverse=True)
        dispersion = "wide" if rets[0] - statistics.median(rets) > 0.15 else "narrow"
        return {"btc_trend": btc_trend, "breadth": breadth, "vol": vol, "dispersion": dispersion}

    def finding(self, t: int) -> str:
        r = self.regime(t)
        return f"{r['btc_trend']}_{r['breadth']}_{r['vol']}"


@dataclass
class Portfolio:
    cash: float = 100.0
    holdings: dict[str, float] = field(default_factory=dict)
    costs_paid: float = 0.0

    def equity(self, m: Market, t: int) -> float:
        return self.cash + sum(q * m.price(s, t) for s, q in self.holdings.items())

    def rebalance(self, target: list[str], m: Market, t: int) -> float:
        """Pesos iguales 1/TOP_K por pick (mitad en cash si hay uno solo). Devuelve el costo."""
        if set(target) == set(self.holdings):
            return 0.0
        eq = self.equity(m, t)
        slot = eq / TOP_K
        cost = 0.0
        for s in list(self.holdings):
            px = m.price(s, t)
            have = self.holdings[s] * px
            sell = have if s not in target else max(0.0, have - slot)
            if sell <= 0.5:
                continue
            self.cash += sell * (1 - COST)
            cost += sell * COST
            self.holdings[s] -= sell / px
            if self.holdings[s] * px < 0.5:
                self.holdings.pop(s)
        for s in target:
            px = m.price(s, t)
            have = self.holdings.get(s, 0.0) * px
            spend = min(slot - have, self.cash)
            if spend <= 0.5:
                continue
            self.holdings[s] = self.holdings.get(s, 0.0) + spend * (1 - COST) / px
            self.cash -= spend
            cost += spend * COST
        self.costs_paid += cost
        return cost

    def holding_bucket(self) -> str:
        n = len(self.holdings)
        return "cash" if n == 0 else ("one" if n == 1 else "two")


def band_for(horizon: int) -> float:
    """Umbral de 'resolves'/'worsens' escalado con la raíz del horizonte (±6 % a 21 días, ±3.5 % a 7)."""
    return round(RESOLVE_T * (horizon / 21) ** 0.5, 3)


def effect_of(r: float, band: float = RESOLVE_T) -> str:
    if r > band:
        return "resolves"
    if r > 0:
        return "improves"
    if r >= -band:
        return "no_change"
    return "worsens"


@dataclass
class MarketWorld:
    market: Market
    portfolio: Portfolio
    t: int
    horizon: int = HORIZON
    steps: int = 0
    score: int = 0
    worsens: int = 0
    resolved: bool = False
    done: bool = False
    ret: float = 0.0
    cost: float = 0.0
    action: str | None = None
    target: list[str] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)

    def features(self) -> dict[str, str]:
        return {**self.market.regime(self.t), "holding": self.portfolio.holding_bucket()}

    def intro(self) -> str:
        m, t = self.market, self.t
        held = ", ".join(f"{s} {q * m.price(s, t):.0f} USD" for s, q in self.portfolio.holdings.items()) or "cash"
        return (f"Día de decisión (la próxima es en {self.horizon} días). Cartera: {held}; cash "
                f"{self.portfolio.cash:.0f} USD; equity {self.portfolio.equity(m, t):.2f} USD. Decidí qué hacer.")

    def _step(self) -> None:
        self.steps += 1

    # --- diagnósticas -----------------------------------------------------------
    def diagnose(self, tool: str, **args) -> dict:
        if self.done:
            return {"status": "invalid", "observed_effect": "diagnostic",
                    "message": "FIN: la decisión de esta semana ya está tomada."}
        self._step()
        m, t = self.market, self.t
        if tool == "market_snapshot":
            rows = []
            for s in sorted(SYMBOLS, key=lambda s: m.ret(s, t, LOOKBACK), reverse=True):
                rows.append({"symbol": s, "ret_30d_pct": round(100 * m.ret(s, t, LOOKBACK), 1),
                             "ret_7d_pct": round(100 * m.ret(s, t, 7), 1),
                             "above_sma100": m.above_sma(s, t)})
            reg = m.regime(t)
            out = {"status": "success", "observed_effect": "diagnostic", "ranking": rows,
                   "regime": reg, "finding": m.finding(t), "momentum_rule_target": m.momentum_target(t)}
            if self.steps >= MAX_STEPS:
                return self._force_hold(out)
            return out
        if tool == "asset_detail":
            s = args.get("symbol")
            if s not in SYMBOLS:
                return {"status": "invalid", "observed_effect": "diagnostic",
                        "message": f"símbolo desconocido; universo: {SYMBOLS}"}
            out = {"status": "success", "observed_effect": "diagnostic", "symbol": s,
                   "ret_7d_pct": round(100 * m.ret(s, t, 7), 1), "ret_30d_pct": round(100 * m.ret(s, t, 30), 1),
                   "ret_90d_pct": round(100 * m.ret(s, t, 90), 1), "vol_30d_ann_pct": round(100 * m.vol(s, t), 0),
                   "dist_to_sma100_pct": round(100 * (m.price(s, t) / m.sma(s, t) - 1), 1)}
            if self.steps >= MAX_STEPS:
                return self._force_hold(out)
            return out
        if tool == "portfolio":
            out = {"status": "success", "observed_effect": "diagnostic", "cash": round(self.portfolio.cash, 2),
                   "holdings": {s: round(q * m.price(s, t), 2) for s, q in self.portfolio.holdings.items()},
                   "equity": round(self.portfolio.equity(m, t), 2),
                   "costs_paid_total": round(self.portfolio.costs_paid, 2)}
            if self.steps >= MAX_STEPS:
                return self._force_hold(out)
            return out
        return {"status": "invalid", "observed_effect": "diagnostic", "message": f"tool desconocida {tool}"}

    def _force_hold(self, out: dict) -> dict:
        res = self.act("hold", forced=True)
        out["message"] = f"Presupuesto agotado sin decidir: se mantuvo la cartera. {res['message']}"
        return out

    # --- acciones ---------------------------------------------------------------
    def act(self, action: str, forced: bool = False, **args) -> dict:
        if self.done:
            return {"status": "invalid", "observed_effect": "no_change",
                    "message": "FIN: la decisión de esta semana ya está tomada."}
        m, t = self.market, self.t
        if action == "follow_momentum":
            target = m.momentum_target(t)
        elif action == "hold":
            target = list(self.portfolio.holdings)
        elif action == "go_cash":
            target = []
        elif action == "rotate":
            syms = args.get("symbols") or []
            if isinstance(syms, str):
                syms = [x.strip() for x in syms.split(",") if x.strip()]
            if not (1 <= len(syms) <= TOP_K) or any(s not in SYMBOLS for s in syms):
                return {"status": "invalid", "observed_effect": "no_change",
                        "message": f"rotate necesita 1–{TOP_K} símbolos del universo {SYMBOLS}"}
            target = list(dict.fromkeys(syms))
        else:
            return {"status": "invalid", "observed_effect": "no_change", "message": f"acción desconocida {action}"}
        if not forced:
            self._step()
        eq0 = self.portfolio.equity(m, t)
        self.cost = self.portfolio.rebalance(target, m, t)
        eq7 = self.portfolio.equity(m, t + self.horizon)
        self.ret = eq7 / eq0 - 1.0
        effect = effect_of(self.ret, band_for(self.horizon))
        self.score = round(1000 * self.ret)
        self.resolved = self.ret > 0
        self.worsens = int(effect == "worsens")
        self.action, self.target, self.done = action, target, True
        self.log.append({"action": action, "args": {"symbols": target}, "observed_effect": effect,
                         "ret_pct": round(100 * self.ret, 2), "forced": forced, "step": self.steps})
        return {"status": "success", "observed_effect": effect,
                "message": (f"Cartera → {target or ['cash']}; costo {self.cost:.2f} USD. "
                            f"{self.horizon} días después: {100 * self.ret:+.2f} %."),
                "ret_pct": round(100 * self.ret, 2)}


def benchmark_week(m: Market, t: int, rule: Portfolio, horizon: int = HORIZON) -> tuple[float, float]:
    """(retorno de la regla pura, retorno de BTC) a `horizon` días desde el día t."""
    eq0 = rule.equity(m, t)
    rule.rebalance(m.momentum_target(t), m, t)
    rule_ret = rule.equity(m, t + horizon) / eq0 - 1.0
    btc_ret = m.price("BTC/USDT", t + horizon) / m.price("BTC/USDT", t) - 1.0
    return rule_ret, btc_ret


SPACING = 21  # días entre decisiones: 40 decisiones cubren abr-2024 → jul-2026 (alcista y bajista)


def decision_days(m: Market, n: int, spacing: int = SPACING, horizon: int = HORIZON) -> list[int]:
    """Los últimos `n` días de decisión, cada `spacing` días, con `horizon` días de futuro y 1 año de historia.
    El resultado de cada decisión se mide a `horizon` días; la cartera sigue viva hasta la decisión siguiente."""
    last = m.n - 1 - horizon
    days = [last - spacing * k for k in range(n)][::-1]
    assert days[0] >= max(VOL_HIST + LOOKBACK, TREND_W), "datos insuficientes para el warmup"
    return days
