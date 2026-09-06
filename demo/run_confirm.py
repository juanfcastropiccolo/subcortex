"""Corrida confirmatoria en opsworld: arquitectura congelada, semillas nunca vistas, 7 brazos.

Diferencias con `run_ab.py` (que fue el runner de desarrollo):
- La unidad de réplica es la TRAYECTORIA (una semilla completa con su memoria), no el episodio.
- Cada brazo corre la MISMA secuencia de incidentes por semilla: comparación pareada.
- Las semillas son held-out: 42 (y 7) quedaron contaminadas por el desarrollo y no se usan.
- El orden de ejecución de los brazos se baraja por semilla, para que un brazo no cargue
  sistemáticamente con la degradación del proveedor a lo largo de la corrida.
- Reanudable a nivel (brazo, semilla) y a nivel episodio: un corte por cuota no pierde trabajo.

Uso:
    SUBCORTEX_MODEL=claude-code:sonnet uv run python -m demo.run_confirm --stage A
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

from demo.ab import run_episodes
from demo.agent import MODEL, build_agent
from demo.arms import ARMS, CONFIG_SHA, STAGES, build_arm_app, protocol_spec
from demo.stats import render_report, trajectory_summary
from opsworld.tools import registry
from opsworld.world import (
    COARSE_FEATURES,
    DIAGNOSTIC_TOOLS,
    RISK,
    World,
    features,
    generate_incidents,
)

# Semillas de evaluación: elegidas de antemano y nunca usadas en desarrollo (42 y 7 sí lo fueron).
HELD_OUT_SEEDS = (101, 202, 303, 404, 505)
ORDER_SEED = 20260906  # baraja el orden de brazos de forma determinista y auditable


def _prompt(f: dict, incident_id: int) -> str:
    return (f"Incidente #{incident_id}: servicio={f['service']}, síntoma={f['symptom']}, "
            f"deploy_reciente={f['recent_deploy']}, tráfico={f['traffic']}, "
            f"franja={f['hour_bucket']}. Actuá.")


def load(path: str) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def save(path: str, data: dict) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=sorted(STAGES), default="A",
                    help="A: vanilla/protocol/full · B: retrieval/cache · C: gate/telemetry")
    ap.add_argument("--arms", default=None, help="lista separada por comas (pisa --stage)")
    ap.add_argument("--n", type=int, default=30, help="episodios por trayectoria")
    ap.add_argument("--seeds", default=None, help="semillas separadas por comas")
    ap.add_argument("--consolidate-every", type=int, default=10)
    ap.add_argument("--out", default="results-confirm.json")
    args = ap.parse_args()

    arm_names = ([a.strip() for a in args.arms.split(",")] if args.arms else list(STAGES[args.stage]))
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds else list(HELD_OUT_SEEDS))
    for a in arm_names:
        if a not in ARMS:
            raise SystemExit(f"brazo desconocido: {a} (disponibles: {sorted(ARMS)})")

    data = load(args.out)
    if data and data.get("config_sha") != CONFIG_SHA:
        raise SystemExit(f"El archivo {args.out} se generó con config_sha={data.get('config_sha')} "
                         f"y la config actual es {CONFIG_SHA}: son arquitecturas distintas, usá otro --out.")
    data.setdefault("config_sha", CONFIG_SHA)
    data.setdefault("protocol", protocol_spec())
    data.setdefault("model", MODEL)
    data.setdefault("n_episodes", args.n)
    data.setdefault("rows", {})
    rows_all: dict = data["rows"]

    plan = [(seed, arm) for seed in seeds for arm in arm_names]
    rng = random.Random(ORDER_SEED)
    by_seed: dict[int, list[str]] = {}
    for seed in seeds:
        shuffled = list(arm_names)
        rng.shuffle(shuffled)
        by_seed[seed] = shuffled
    plan = [(seed, arm) for seed in seeds for arm in by_seed[seed]]

    print(f"Protocolo congelado {CONFIG_SHA} · modelo {MODEL} · {args.n} episodios × "
          f"{len(seeds)} semillas × {len(arm_names)} brazos = {args.n * len(plan)} episodios", flush=True)

    for seed, arm_name in plan:
        key = f"{arm_name}|{seed}"
        done = rows_all.get(key, [])
        if len(done) >= args.n:
            print(f"[{key}] ya completo ({len(done)} episodios), salteo", flush=True)
            continue
        arm = ARMS[arm_name]
        incidents = generate_incidents(args.n, seed)
        registry.clear()
        agent = build_agent()
        app, sc = build_arm_app(arm, agent, risk=RISK, diagnostic_tools=DIAGNOSTIC_TOOLS,
                                coarse_features=COARSE_FEATURES, app_name="confirm")
        print(f"\n=== {key} · {arm.purpose[:70]} ===", flush=True)

        def _on_row(rows, key=key):
            rows_all[key] = rows
            save(args.out, data)

        rows = await run_episodes(
            key, incidents, app=app, sc=sc, registry=registry, app_name="confirm",
            make_world=lambda inc, i: World(inc),
            features=lambda w: features(w.incident),
            prompt=lambda w: _prompt(features(w.incident), w.incident.id),
            label=lambda inc: inc.cause,
            with_subcortex=arm.attach,
            consolidate_every=args.consolidate_every if sc else 0,
            resume_rows=done, on_row=_on_row)
        rows_all[key] = rows
        save(args.out, data)

    # Resumen por trayectoria y comparación pareada
    trajectories: dict[str, dict[int, dict]] = {}
    for key, rows in rows_all.items():
        arm_name, seed = key.split("|")
        if len(rows) >= args.n:
            trajectories.setdefault(arm_name, {})[int(seed)] = trajectory_summary(rows)
    data["trajectories"] = {a: {str(s): t for s, t in by.items()} for a, by in trajectories.items()}
    save(args.out, data)
    print("\n" + render_report(trajectories))
    print(f"\nGuardado en {args.out} (config_sha {CONFIG_SHA})")


if __name__ == "__main__":
    asyncio.run(main())
