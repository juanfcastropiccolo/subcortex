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
from subcortex.types import K_FEATURES, K_VETO_LOG

MAX_ATTEMPTS = 4


async def run_variant(name: str, incidents, with_subcortex: bool, model=MODEL,
                      consolidate_every: int = 0, store_path: str = ":memory:"):
    app, sc = build_app(with_subcortex, model=model, store_path=store_path)
    svc = InMemorySessionService()
    runner = Runner(app=app, session_service=svc)
    rows = []
    for i, inc in enumerate(incidents):
        f = features(inc)
        prompt = (f"Incidente #{inc.id}: servicio={f['service']}, síntoma={f['symptom']}, "
                  f"deploy_reciente={f['recent_deploy']}, tráfico={f['traffic']}, "
                  f"franja={f['hour_bucket']}. Actuá.")
        msg = types.Content(role="user", parts=[types.Part(text=prompt)])
        t0 = time.time()
        # Un 503/429 transitorio de Gemini no debe tirar la corrida: reintento con sesión y mundo nuevos.
        for attempt in range(1, MAX_ATTEMPTS + 1):
            session = await svc.create_session(app_name="opsworld", user_id="demo",
                                               state={K_FEATURES: f})
            world = World(inc)
            registry.register(session.id, world)
            model_turns = 0
            tokens_fallback = 0
            try:
                async for ev in runner.run_async(user_id="demo", session_id=session.id,
                                                 new_message=msg):
                    if (ev.author != "user" and not ev.partial and ev.content
                            and not ev.get_function_responses()):
                        model_turns += 1
                    if ev.usage_metadata and ev.usage_metadata.total_token_count:
                        tokens_fallback += ev.usage_metadata.total_token_count
                break
            except Exception as e:
                if attempt == MAX_ATTEMPTS:
                    raise
                wait = 5 * attempt
                print(f"[{name}] #{i:02d} fallo transitorio ({type(e).__name__}); "
                      f"reintento {attempt}/{MAX_ATTEMPTS - 1} en {wait}s", flush=True)
                await asyncio.sleep(wait)
        session = await svc.get_session(app_name="opsworld", user_id="demo", session_id=session.id)
        m = get_metrics(session.state) if with_subcortex else {}
        row = {"variant": name, "i": i, "cause": inc.cause, "score": world.score,
               "resolved": world.resolved, "steps": world.steps, "worsens": world.worsens,
               "seconds": round(time.time() - t0, 1),
               "llm_calls": m.get("llm_calls", model_turns), "tokens": m.get("tokens", tokens_fallback),
               "vetoes": m.get("vetoes", 0), "vetoes_irreversible": m.get("vetoes_irreversible", 0),
               "rejected": m.get("rejected", 0),
               "episodes_written": m.get("episodes_written", 0), "habit_hits": m.get("habit_hits", 0),
               "dehabituations": m.get("dehabituations", 0),
               "abs_error_sum": m.get("abs_error_sum", 0.0), "error_count": m.get("error_count", 0),
               "actions": [e["action"] for e in world.log],
               "veto_log": list(session.state.get(K_VETO_LOG) or []) if with_subcortex else []}
        rows.append(row)
        print(f"[{name}] #{i:02d} {inc.cause:15s} score={world.score:5d} steps={world.steps} "
              f"llm={row['llm_calls']} vetoes={row['vetoes']} rej={row['rejected']} "
              f"habit={row['habit_hits']} acciones={row['actions']}", flush=True)
        if sc and consolidate_every and (i + 1) % consolidate_every == 0:
            print(f"[{name}] consolidate → {sc.consolidate()}", flush=True)
    if sc:
        print(f"[{name}] store: {sc.store.stats()}", flush=True)
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
    keys = ["score_mean", "resolved_rate", "llm_calls_mean", "tokens_mean", "steps_mean",
            "worsens_total", "vetoes_total", "vetoes_irreversible_total", "episodes_written_total",
            "habit_hits_total", "dehabituations_total", "mean_abs_error"]
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
        rows = await run_variant(name, incidents, flag,
                                 consolidate_every=args.consolidate_every if flag else 0)
        results[name] = rows
        summaries[name] = summarize(rows)
        # Guardado parcial: si la segunda variante cae, la primera no se pierde.
        Path(args.out).write_text(json.dumps({"rows": results, "summary": summaries}, indent=2,
                                             ensure_ascii=False))
    print_table(summaries)
    print(f"\nGuardado en {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
