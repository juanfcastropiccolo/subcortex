"""A/B en marketworld: el mismo gestor con y sin subcortex sobre las mismas 40 semanas,
más dos referencias sin LLM: la regla momentum pura y BTC buy&hold."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from demo.ab import load_baseline, load_partial, print_table, run_episodes, save, summarize
from demo.market_agent import build_app
from marketworld.tools import registry
from marketworld.world import HORIZON, Market, MarketWorld, Portfolio, benchmark_week, decision_days

RUN_ROOT = Path(".marketworld")


def references(m: Market, days: list[int]) -> dict:
    rule, eq_rule, eq_btc = Portfolio(), 100.0, 100.0
    rows = []
    for t in days:
        rr, br = benchmark_week(m, t, rule)
        eq_rule *= 1 + rr
        eq_btc *= 1 + br
        rows.append({"t": t, "rule_ret_pct": round(100 * rr, 2), "btc_ret_pct": round(100 * br, 2),
                     "finding": m.finding(t)})
    # La regla de la casa real rebalancea a diario (momentum_paper.py); la de arriba solo puede
    # operar en los días de decisión del agente. Las dos son referencias legítimas y distintas.
    daily = Portfolio()
    for t in range(days[0], days[-1] + HORIZON):
        daily.rebalance(m.momentum_target(t), m, t)
    eq_daily = daily.equity(m, days[-1] + HORIZON)
    return {"rows": rows, "rule_equity": round(eq_rule, 2), "rule_daily_equity": round(eq_daily, 2),
            "btc_equity": round(eq_btc, 2)}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--consolidate-every", type=int, default=10)
    ap.add_argument("--only", choices=["baseline", "subcortex"], default=None)
    ap.add_argument("--out", default="results-market.json")
    ap.add_argument("--baseline-from", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    m = Market()
    days = decision_days(m, args.n)
    refs = references(m, days)
    print(f"referencias sobre {args.n} decisiones: regla al ritmo del agente → {refs['rule_equity']}  "
          f"regla diaria → {refs['rule_daily_equity']}  BTC → {refs['btc_equity']}")
    results = load_partial(args.out) if args.resume else {}
    results = {k: v for k, v in results.items() if v and k in ("baseline", "subcortex")}
    if args.baseline_from:
        results["baseline"] = load_baseline(args.baseline_from, len(days))
    for name, flag in (("baseline", False), ("subcortex", True)):
        if args.only and args.only != name:
            continue
        if len(results.get(name, [])) >= len(days):
            continue
        registry.clear()
        RUN_ROOT.mkdir(exist_ok=True)
        store_path = str(RUN_ROOT / "subcortex.sqlite")
        if flag and not (args.resume and results.get(name)):
            Path(store_path).unlink(missing_ok=True)
        app, sc = build_app(flag, store_path=store_path if flag else ":memory:")
        # La cartera persiste entre semanas: replay secuencial. Con --resume se reconstruye
        # re-aplicando las decisiones ya guardadas.
        portfolio = Portfolio()
        for r in results.get(name, []):
            portfolio.rebalance(r["target"], m, r["t"])
        state = {"last": None}

        def make_world(t, i, portfolio=portfolio, state=state):
            w = MarketWorld(m, portfolio, t)
            state["last"] = w
            return w

        rows = await run_episodes(
            name, days, app=app, sc=sc, registry=registry, app_name="marketworld",
            make_world=make_world, features=lambda w: w.features(), prompt=lambda w: w.intro(),
            label=lambda t: m.finding(t),
            extra=lambda w: {"t": w.t, "action": w.action, "target": w.target, "ret_pct": round(100 * w.ret, 2),
                             "cost": round(w.cost, 2), "equity_after": round(w.portfolio.equity(m, w.t + HORIZON), 2)},
            with_subcortex=flag, consolidate_every=args.consolidate_every if flag else 0,
            resume_rows=results.get(name), on_row=lambda rs, name=name: save(args.out, {**results, name: rs}))
        results[name] = rows
        save(args.out, results)
    summaries = {k: summarize(v) for k, v in results.items() if v}
    print_table(summaries)
    for k, v in results.items():
        print(f"{k:10s} equity final {v[-1]['equity_after']:.2f}  acciones: "
              + ", ".join(f"{a}×{sum(1 for r in v if r['action'] == a)}"
                          for a in ("follow_momentum", "hold", "go_cash", "rotate")))
    print(f"regla (ritmo agente) equity final {refs['rule_equity']:.2f}")
    print(f"regla diaria         equity final {refs['rule_daily_equity']:.2f}")
    print(f"BTC                  equity final {refs['btc_equity']:.2f}")
    save(args.out, results)
    Path(args.out.replace(".json", "-refs.json")).write_text(__import__("json").dumps(refs, indent=2))
    print(f"\nGuardado en {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
