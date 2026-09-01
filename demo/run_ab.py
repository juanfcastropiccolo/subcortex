"""A/B en opsworld: el mismo agente operador con y sin subcortex sobre los mismos incidentes."""
from __future__ import annotations

import argparse
import asyncio

from demo.ab import load_baseline, print_table, run_episodes, save, summarize
from demo.agent import build_app
from opsworld.tools import registry
from opsworld.world import World, features, generate_incidents


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--consolidate-every", type=int, default=10)
    ap.add_argument("--only", choices=["baseline", "subcortex"], default=None)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--baseline-from", default=None,
                    help="reusar las filas de baseline de un results.json previo (mismo n y seed)")
    args = ap.parse_args()
    incidents = generate_incidents(args.n, args.seed)
    results, summaries = {}, {}
    if args.baseline_from:
        results["baseline"] = load_baseline(args.baseline_from, args.n)
        summaries["baseline"] = summarize(results["baseline"])
    for name, flag in (("baseline", False), ("subcortex", True)):
        if (args.only and args.only != name) or name in results:
            continue
        registry.clear()
        app, sc = build_app(flag)
        rows = await run_episodes(
            name, incidents, app=app, sc=sc, registry=registry, app_name="opsworld",
            make_world=lambda inc, i: World(inc),
            features=lambda w: features(w.incident),
            prompt=lambda w: _prompt(features(w.incident), w.incident.id),
            label=lambda inc: inc.cause,
            with_subcortex=flag, consolidate_every=args.consolidate_every if flag else 0)
        results[name] = rows
        summaries[name] = summarize(rows)
        save(args.out, results, summaries)
    print_table(summaries)
    print(f"\nGuardado en {args.out}")


def _prompt(f: dict, incident_id: int) -> str:
    return (f"Incidente #{incident_id}: servicio={f['service']}, síntoma={f['symptom']}, "
            f"deploy_reciente={f['recent_deploy']}, tráfico={f['traffic']}, "
            f"franja={f['hour_bucket']}. Actuá.")


if __name__ == "__main__":
    asyncio.run(main())
