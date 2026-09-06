"""Estadística del protocolo confirmatorio: la réplica es la trayectoria, no el episodio.

La auditoría del 2026-09-06 marcó tres errores estadísticos de las corridas de desarrollo:
episodios tratados como observaciones independientes cuando la memoria los acopla en serie,
ablaciones de una sola trayectoria contra una varianza de ±19.5, y ausencia de pruebas o
intervalos. Este módulo corrige los tres: agrega a nivel trayectoria, compara pareado por
semilla (misma secuencia de tareas en todos los brazos) y reporta bootstrap con consistencia
de signo en vez de un p-valor sobre n chico.
"""
from __future__ import annotations

import math
import random
import statistics

from demo.arms import LAMBDA_CALLS_PRIMARY, LAMBDA_CALLS_SENSITIVITY

EFFECTS4 = ("worsens", "no_change", "improves", "resolves")


# --- calibración -------------------------------------------------------------------------
def calibration(pred_rows: list[dict], bins: int = 5) -> dict:
    """Brier, log-loss y ECE a partir del contrato (etiqueta esperada + confianza).

    El contrato actual no pide una distribución completa, así que la reconstruimos como
    p(esperado) = confianza y el resto uniforme sobre las otras tres etiquetas. Es una
    aproximación declarada: un contrato distribucional propio queda como trabajo futuro, pero
    esto ya distingue "predice mejor" de "bajó el error categórico medio cambiando de acciones".
    """
    rows = [r for r in pred_rows if r.get("observed") in EFFECTS4 and r.get("expected") in EFFECTS4]
    if not rows:
        return {"n": 0}
    briers, nlls, hits, confs = [], [], [], []
    for r in rows:
        c = min(max(float(r["confidence"]), 0.0), 1.0)
        rest = (1.0 - c) / 3.0
        p = {e: (c if e == r["expected"] else rest) for e in EFFECTS4}
        briers.append(sum((p[e] - (1.0 if e == r["observed"] else 0.0)) ** 2 for e in EFFECTS4))
        nlls.append(-math.log(max(p[r["observed"]], 1e-9)))
        hits.append(1.0 if r["expected"] == r["observed"] else 0.0)
        confs.append(c)
    ece, n = 0.0, len(rows)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if (lo < c <= hi or (b == 0 and c == 0))]
        if idx:
            ece += (len(idx) / n) * abs(statistics.mean(confs[i] for i in idx)
                                        - statistics.mean(hits[i] for i in idx))
    return {"n": n, "brier": round(statistics.mean(briers), 4), "nll": round(statistics.mean(nlls), 4),
            "ece": round(ece, 4), "accuracy": round(statistics.mean(hits), 4),
            "mean_confidence": round(statistics.mean(confs), 4)}


# --- agregación por trayectoria ----------------------------------------------------------
def utility(row: dict, lam: float = LAMBDA_CALLS_PRIMARY) -> float:
    """Endpoint primario por episodio. El score del mundo ya cobra daño y pasos; λ cobra cómputo."""
    return row["score"] - lam * row.get("llm_calls", 0)


def trajectory_summary(rows: list[dict]) -> dict:
    """Una trayectoria = una (semilla, brazo). Devuelve sus escalares, incluidos los de la
    segunda mitad, donde la capa ya tuvo tiempo de aprender."""
    if not rows:
        return {}
    n = len(rows)
    half = rows[n // 2:]
    pred_log = [p for r in rows for p in (r.get("pred_log") or [])]

    def mean(key, src=rows):
        return round(statistics.mean(r.get(key, 0) for r in src), 3)

    out = {
        "n_episodes": n,
        "utility_mean": round(statistics.mean(utility(r) for r in rows), 3),
        "utility_mean_second_half": round(statistics.mean(utility(r) for r in half), 3),
        "score_mean": mean("score"),
        "resolved_rate": round(sum(1 for r in rows if r.get("resolved")) / n, 4),
        "harm_total": sum(r.get("worsens", 0) for r in rows),
        "llm_calls_mean": mean("llm_calls"),
        "llm_calls_mean_second_half": mean("llm_calls", half),
        "tokens_mean": mean("tokens"),
        "seconds_mean": mean("seconds"),
        "steps_mean": mean("steps"),
        "vetoes_total": sum(r.get("vetoes", 0) for r in rows),
        "vetoes_value_total": sum(r.get("vetoes_value", 0) for r in rows),
        "vetoes_hyperdirect_total": sum(r.get("vetoes_hyperdirect", 0) for r in rows),
        "vetoes_blockstreak_total": sum(r.get("vetoes_blockstreak", 0) for r in rows),
        "arbitration_dropped_total": sum(r.get("arbitration_dropped", 0) for r in rows),
        "rejected_total": sum(r.get("rejected", 0) for r in rows),
        "habit_hits_total": sum(r.get("habit_hits", 0) for r in rows),
        "dehabituations_total": sum(r.get("dehabituations", 0) for r in rows),
        "episodes_written_total": sum(r.get("episodes_written", 0) for r in rows),
    }
    for lam in LAMBDA_CALLS_SENSITIVITY:
        out[f"utility_lambda_{lam:g}"] = round(statistics.mean(utility(r, lam) for r in rows), 3)
    out["calibration"] = calibration(pred_log)
    return out


# --- comparación pareada -----------------------------------------------------------------
def paired_bootstrap(diffs: list[float], iters: int = 20000, seed: int = 20260906) -> dict:
    """IC percentil sobre la media de diferencias pareadas, remuestreando SEMILLAS.

    Con pocas semillas el intervalo es ancho a propósito: es la incertidumbre real, no un
    p-valor cosmético. Se reporta junto a la consistencia de signo, que con n chico informa más.
    """
    if not diffs:
        return {"n": 0}
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(iters):
        means.append(statistics.mean(rng.choice(diffs) for _ in range(n)))
    means.sort()
    return {
        "n_pairs": n,
        "mean_diff": round(statistics.mean(diffs), 3),
        "ci95": [round(means[int(0.025 * iters)], 3), round(means[int(0.975 * iters)], 3)],
        "favorable": sum(1 for d in diffs if d > 0),
        "unfavorable": sum(1 for d in diffs if d < 0),
        "per_seed": [round(d, 3) for d in diffs],
    }


def compare(trajectories: dict[str, dict[int, dict]], reference: str, metric: str,
            higher_is_better: bool = True) -> dict[str, dict]:
    """`trajectories[arm][seed] = trajectory_summary`. Compara cada brazo contra `reference`
    sobre las semillas que ambos tengan completas."""
    ref = trajectories.get(reference, {})
    out = {}
    for arm, by_seed in trajectories.items():
        if arm == reference:
            continue
        seeds = sorted(set(by_seed) & set(ref))
        diffs = [(by_seed[s][metric] - ref[s][metric]) * (1 if higher_is_better else -1)
                 for s in seeds if metric in by_seed[s] and metric in ref[s]]
        out[arm] = paired_bootstrap(diffs) | {"seeds": seeds, "metric": metric,
                                              "direction": "mayor mejor" if higher_is_better else "menor mejor"}
    return out


def pareto_front(points: dict[str, tuple[float, float]]) -> list[str]:
    """Frontera seguridad-utilidad: (utilidad ↑, daño ↓). Devuelve los brazos no dominados."""
    front = []
    for a, (ua, ha) in points.items():
        if not any(ub >= ua and hb <= ha and (ub > ua or hb < ha) for b, (ub, hb) in points.items() if b != a):
            front.append(a)
    return sorted(front)


def render_report(trajectories: dict[str, dict[int, dict]], reference: str = "vanilla") -> str:
    """Tabla de texto lista para pegar en el reporte de resultados."""
    lines = []
    arms = list(trajectories)
    lines.append(f"{'brazo':12s} {'semillas':>9s} {'U':>9s} {'U 2ª mit':>9s} {'score':>8s} "
                 f"{'resol':>7s} {'daño':>6s} {'calls':>7s} {'vetos':>6s} {'hábitos':>8s}")
    for arm in arms:
        by_seed = trajectories[arm]
        if not by_seed:
            continue

        def m(key, by_seed=by_seed):
            vals = [t[key] for t in by_seed.values() if key in t]
            return statistics.mean(vals) if vals else 0.0

        lines.append(f"{arm:12s} {len(by_seed):>9d} {m('utility_mean'):>9.2f} "
                     f"{m('utility_mean_second_half'):>9.2f} {m('score_mean'):>8.2f} "
                     f"{m('resolved_rate'):>7.2f} {m('harm_total'):>6.1f} {m('llm_calls_mean'):>7.2f} "
                     f"{m('vetoes_total'):>6.1f} {m('habit_hits_total'):>8.1f}")
    lines.append("")
    lines.append(f"Comparación pareada por semilla contra '{reference}' (endpoint primario U):")
    for arm, res in compare(trajectories, reference, "utility_mean").items():
        if res.get("n_pairs"):
            lines.append(f"  {arm:12s} ΔU = {res['mean_diff']:+7.2f}  IC95 {res['ci95']}  "
                         f"semillas a favor: {res['favorable']}/{res['n_pairs']}  {res['per_seed']}")
    return "\n".join(lines)
