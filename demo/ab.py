"""Corrida A/B genérica: mismo agente, misma secuencia de episodios, con y sin subcortex.

Lo específico de cada mundo (cómo se crea, qué features expone, qué prompt recibe el agente)
se pasa como callables; lo común (sesiones ADK, reintentos, métricas, resumen) vive acá.
"""
from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from subcortex.metrics import get_metrics
from subcortex.types import K_FEATURES, K_PRED_LOG, K_VETO_LOG

MAX_ATTEMPTS = 6
BACKOFF = (10, 30, 60, 120, 180)  # segundos entre reintentos: un 429 de cuota por minuto necesita esperar
MAX_LLM_CALLS = 40  # tope por episodio: si el modelo no cierra, el episodio termina igual
EPISODE_TIMEOUT = 480  # segundos de pared por episodio: un bucle sin LLM no puede colgar la corrida


async def run_episodes(name: str, items: list, *, app, sc, registry, app_name: str,
                       make_world: Callable[[Any, int], Any], features: Callable[[Any], dict],
                       prompt: Callable[[Any], str], label: Callable[[Any], str],
                       extra: Callable[[Any], dict] | None = None, with_subcortex: bool,
                       consolidate_every: int = 0, resume_rows: list[dict] | None = None,
                       on_row: Callable[[list[dict]], None] | None = None,
                       consolidate_llm: Callable[[str], str] | None = None) -> list[dict]:
    """`resume_rows`: filas ya corridas de esta variante (se saltean esos episodios).
    `on_row`: callback tras cada episodio (guardado incremental)."""
    svc = InMemorySessionService()
    runner = Runner(app=app, session_service=svc)
    rows = list(resume_rows or [])
    done_idx = {r["i"] for r in rows}
    for i, item in enumerate(items):
        if i in done_idx:
            continue
        t0 = time.time()
        # Un 503/429 transitorio no debe tirar la corrida: reintento con sesión y mundo nuevos.
        for attempt in range(1, MAX_ATTEMPTS + 1):
            world = make_world(item, i)
            f = features(world)
            session = await svc.create_session(app_name=app_name, user_id="demo", state={K_FEATURES: f})
            registry.register(session.id, world)
            msg = types.Content(role="user", parts=[types.Part(text=prompt(world))])
            model_turns = 0
            tokens_fallback = 0
            try:
                async with asyncio.timeout(EPISODE_TIMEOUT):
                    async for ev in runner.run_async(user_id="demo", session_id=session.id, new_message=msg,
                                                     run_config=RunConfig(max_llm_calls=MAX_LLM_CALLS)):
                        if (ev.author != "user" and not ev.partial and ev.content
                                and not ev.get_function_responses()):
                            model_turns += 1
                        if ev.usage_metadata and ev.usage_metadata.total_token_count:
                            tokens_fallback += ev.usage_metadata.total_token_count
                break
            except LlmCallsLimitExceededError:
                print(f"[{name}] #{i:02d} tope de {MAX_LLM_CALLS} llamadas al LLM: episodio cortado", flush=True)
                break
            except TimeoutError:
                print(f"[{name}] #{i:02d} timeout de {EPISODE_TIMEOUT}s: episodio cortado", flush=True)
                break
            except Exception as e:
                if attempt == MAX_ATTEMPTS:
                    raise
                wait = BACKOFF[min(attempt - 1, len(BACKOFF) - 1)]
                print(f"[{name}] #{i:02d} fallo transitorio ({type(e).__name__}); "
                      f"reintento {attempt}/{MAX_ATTEMPTS - 1} en {wait}s", flush=True)
                await asyncio.sleep(wait)
        session = await svc.get_session(app_name=app_name, user_id="demo", session_id=session.id)
        m = get_metrics(session.state) if with_subcortex else {}
        row = {"variant": name, "i": i, "cause": label(item), "score": world.score,
               "resolved": world.resolved, "steps": world.steps, "worsens": world.worsens,
               "seconds": round(time.time() - t0, 1),
               # `or` y no `get(..., default)`: si la métrica quedó en 0 por una configuración sin
               # el plugin que la cuenta, el costo real igual se registra desde los eventos.
               "llm_calls": m.get("llm_calls") or model_turns,
               "tokens": m.get("tokens") or tokens_fallback,
               "vetoes": m.get("vetoes", 0), "vetoes_irreversible": m.get("vetoes_irreversible", 0),
               "rejected": m.get("rejected", 0),
               "episodes_written": m.get("episodes_written", 0), "habit_hits": m.get("habit_hits", 0),
               "dehabituations": m.get("dehabituations", 0), "stalled": m.get("stalled", 0),
               "abs_error_sum": m.get("abs_error_sum", 0.0), "error_count": m.get("error_count", 0),
               # Desglose causal del veto y registro de calibración (auditoría 2026-09-06)
               "vetoes_value": m.get("vetoes_value", 0),
               "vetoes_hyperdirect": m.get("vetoes_hyperdirect", 0),
               "vetoes_blockstreak": m.get("vetoes_blockstreak", 0),
               "arbitration_dropped": m.get("arbitration_dropped", 0),
               "reconsiders": m.get("reconsiders", 0),
               "pred_log": list(session.state.get(K_PRED_LOG) or []) if with_subcortex else [],
               "actions": [e["action"] for e in world.log],
               "veto_log": list(session.state.get(K_VETO_LOG) or []) if with_subcortex else []}
        if extra:
            row.update(extra(world))
        rows.append(row)
        if on_row:
            on_row(rows)
        print(f"[{name}] #{i:02d} {label(item):16s} score={world.score:5d} steps={world.steps} "
              f"llm={row['llm_calls']} vetoes={row['vetoes']} rej={row['rejected']} "
              f"habit={row['habit_hits']} acciones={row['actions']}", flush=True)
        if sc and consolidate_every and (i + 1) % consolidate_every == 0:
            print(f"[{name}] consolidate → {sc.consolidate(llm=consolidate_llm)}", flush=True)
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


def load_baseline(path: str, n: int) -> list[dict]:
    prev = json.loads(Path(path).read_text())["rows"]["baseline"]
    assert len(prev) == n, f"baseline previo con {len(prev)} filas, se esperaban {n}"
    return prev


def load_partial(path: str) -> dict[str, list[dict]]:
    """Filas ya corridas por variante (para --resume). {} si el archivo no existe."""
    p = Path(path)
    return json.loads(p.read_text()).get("rows", {}) if p.exists() else {}


def save(out: str, results: dict, summaries: dict | None = None) -> None:
    summaries = summaries or {k: summarize(v) for k, v in results.items() if v}
    Path(out).write_text(json.dumps({"rows": results, "summary": summaries}, indent=2, ensure_ascii=False))
