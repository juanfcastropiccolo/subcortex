import pytest

from marketworld.world import (
    COST,
    HORIZON,
    SPACING,
    SYMBOLS,
    TOP_K,
    Market,
    MarketWorld,
    Portfolio,
    benchmark_week,
    decision_days,
    effect_of,
)


@pytest.fixture(scope="module")
def market() -> Market:
    return Market()


def test_data_aligned_and_long_enough(market):
    assert market.n > 900 and len(market.closes) == 10
    assert all(len(v) == market.n for v in market.closes.values())
    days = decision_days(market, 40)
    assert len(days) == 40 and days[1] - days[0] == SPACING and days[-1] + HORIZON == market.n - 1


def test_regime_and_rule_are_consistent(market):
    t = decision_days(market, 40)[0]
    r = market.regime(t)
    assert set(r) == {"btc_trend", "breadth", "vol", "dispersion"}
    assert r["btc_trend"] in ("up", "down") and r["breadth"] in ("low", "mid", "high")
    target = market.momentum_target(t)
    assert len(target) <= TOP_K and all(market.above_sma(s, t) for s in target)
    ranked = sorted((s for s in SYMBOLS if market.above_sma(s, t)),
                    key=lambda s: market.ret(s, t, 30), reverse=True)
    assert target == ranked[:TOP_K]
    assert market.finding(t).count("_") == 2


def test_rebalance_costs_and_weights(market):
    t = decision_days(market, 40)[0]
    p = Portfolio()
    cost = p.rebalance(["BTC/USDT", "ETH/USDT"], market, t)
    assert cost == pytest.approx(100 * COST, rel=1e-6)
    eq = p.equity(market, t)
    assert eq == pytest.approx(100 - cost, rel=1e-6) and p.cash < 1
    assert p.rebalance(["ETH/USDT", "BTC/USDT"], market, t) == 0.0  # mismo conjunto: sin operar
    cost2 = p.rebalance(["BTC/USDT"], market, t)  # vende ETH, mitad queda en cash
    assert cost2 > 0 and p.cash > 40 and set(p.holdings) == {"BTC/USDT"}


def test_world_decision_and_forced_hold(market):
    days = decision_days(market, 40)
    w = MarketWorld(market, Portfolio(), days[5])
    f = w.features()
    assert f["holding"] == "cash" and "btc_trend" in f
    snap = w.diagnose("market_snapshot")
    assert snap["finding"] == market.finding(days[5]) and len(snap["ranking"]) == 10
    assert w.diagnose("asset_detail", symbol="NOPE")["status"] == "invalid"
    r = w.act("rotate", symbols=["SOL/USDT", "LINK/USDT", "BTC/USDT"])
    assert r["status"] == "invalid" and not w.done
    r = w.act("follow_momentum")
    assert w.done and r["observed_effect"] == effect_of(w.ret) and w.score == round(1000 * w.ret)
    assert w.act("hold")["status"] == "invalid"
    # presupuesto agotado sin decidir → hold forzado
    w2 = MarketWorld(market, Portfolio(), days[6])
    for _ in range(4):
        w2.diagnose("portfolio")
    out = w2.diagnose("portfolio")
    assert w2.done and w2.action == "hold" and "agotado" in out["message"] and w2.log[0]["forced"]


def test_effect_buckets_and_benchmarks(market):
    assert effect_of(0.08) == "resolves" and effect_of(0.01) == "improves"
    assert effect_of(-0.01) == "no_change" and effect_of(-0.08) == "worsens"
    assert HORIZON == SPACING  # la consecuencia se mide hasta la decisión siguiente
    t = decision_days(market, 40)[0]
    rule = Portfolio()
    rr, br = benchmark_week(market, t, rule)
    assert -0.5 < rr < 0.5 and -0.5 < br < 0.5
