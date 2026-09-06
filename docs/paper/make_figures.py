"""Genera las figuras del paper como SVG embebible, en español e inglés (figures/{es,en}/).

Colores por rol vía CSS vars definidas en la hoja del paper (--viz-s1 subcortex,
--viz-neutral baseline, --viz-ref referencias, tinta en currentColor): las figuras
se adaptan al tema claro/oscuro sin duplicarse.
"""
from __future__ import annotations

from pathlib import Path

BASE = Path(__file__).parent / "figures"

S1 = "var(--viz-s1)"        # subcortex
NEU = "var(--viz-neutral)"  # baseline
REF = "var(--viz-ref)"      # referencias sin LLM
INK2 = "var(--ink-2)"
INK3 = "var(--ink-3)"
LINE = "var(--line)"
FS = 'font-family="IBM Plex Sans, sans-serif" font-size="12"'

T = {
    "es": {
        "f1cap": ("Figura 1. Un turno del bucle: los cinco plugins se cuelgan de los hooks del ciclo de "
                  "tool-calls. Un veto no rompe el bucle (vuelve como resultado y el modelo re-itera); un "
                  "hábito compilado ejecuta la acción sin invocar al modelo."),
        "f1aria": ("Diagrama del flujo de un turno: before_model con memoria, hábito e interocepción; LLM; "
                   "after_model con winner-take-all; before_tool con validación y veto; herramienta; after_tool con aprendizaje."),
        "f1bm": "memoria: precedentes y reglas · hábito: ¿bypass? · interocepción: estado",
        "f1llm": "propone acción + predicción", "f1am": "gate: una sola acción (winner-take-all)",
        "f1bt": "predicción válida · veto por defecto", "f1tool": "herramienta", "f1toolsub": "mundo real → observed_effect",
        "f1at": "error de predicción → memoria (sorpresa/éxito) → dopamina → hábitos → tono",
        "f1habit": "hábito compilado", "f1nollm": "sin llamar al modelo", "f1auth": "autorizada",
        "f1veto": "vetada / rechazada: el modelo re-itera", "f1learn": "lo aprendido alimenta la próxima decisión",
        "f2cap": ("Figura 2. opsworld (40 incidentes): score medio y llamadas al modelo por tercio de la "
                  "secuencia. El baseline no cambia con la experiencia; subcortex baja un tercio sus llamadas "
                  "cuando los hábitos ya están compilados."),
        "f2aria": ("Barras: score medio 46.3 baseline contra 56.3 subcortex; llamadas al modelo por tercio: "
                   "baseline 6.2, 6.7, 7.2; subcortex 5.9, 4.5, 4.6."),
        "f2t1": "Score medio", "f2t2": "Llamadas al LLM por episodio, por tercio", "third": "T",
        "f3cap": ("Figura 3. marketworld: equity final por corredor. Arriba, 40 decisiones cada 21 días (los "
                  "aros son las cinco trayectorias de subcortex; el baseline es determinista). Abajo, la "
                  "variante semanal, donde subcortex supera también a BTC."),
        "f3aria": ("Barras de equity final: a 21 días, baseline 50.5, subcortex 89.3 más menos 19.5 con cinco "
                   "trayectorias entre 68 y 121, regla diaria 78.6, BTC 95.3. Semanal: baseline 91.4, subcortex 101.6, BTC 95.3."),
        "f3t1": "Equity final desde 100 · 40 decisiones (21 d)", "f3t2": "Variante semanal · 120 decisiones (7 d)",
        "f3base21": "baseline (= regla c/21 d)", "f3sub": "subcortex · media de 5", "f3rule": "regla momentum diaria",
        "f3btc": "BTC buy & hold", "f3traj": "trayectoria",
        "f4cap": ("Figura 4. Ablaciones en marketworld (una trayectoria por variante, mismas 40 decisiones). "
                  "La señal conductual es la nítida: sin memoria, el comportamiento en régimen bajista vuelve "
                  "exactamente al del baseline (−3.50 % contra −3.51 %)."),
        "f4aria": ("Barras de ablación: equity con capa completa 89.3, sin gate 91.5, sin hábitos 74.9, sin "
                   "interocepción 69.7, sin memoria 69.2, baseline 50.5; y retorno por decisión en régimen "
                   "bajista de −1.24 por ciento con capa completa a −3.50 sin memoria, igual al baseline."),
        "f4t1": "Equity final", "f4t2": "Régimen bajista: retorno por decisión",
        "f4rows": ["capa completa (media de 5)", "sin gate", "sin hábitos", "sin interocepción", "sin memoria", "baseline"],
        "f4rows2": ["capa completa", "sin gate", "sin hábitos", "sin interocepción", "sin memoria", "baseline"],
        "f5cap": ("Figura 5. Costo operativo por episodio (llamadas al modelo, tokens y tiempo de pared) por "
                  "mundo. En opsworld la capa es más barata que el baseline; en marketworld paga tokens y "
                  "segundos extra por los precedentes en contexto y las corridas de evaluación."),
        "f5aria": ("Comparación por mundo de llamadas al modelo, tokens y segundos por episodio entre baseline "
                   "y subcortex; subcortex usa menos llamadas en opsworld y más tokens en marketworld."),
        "f5t": ["Llamadas al LLM / episodio", "Tokens / episodio (miles)", "Segundos / episodio"],
        "f6cap": ("Figura 6. Mismos A/B con cuatro motores (n=40 por brazo y modelo). En opsworld la capa "
                  "reduce llamadas y acciones dañinas con los cuatro; en marketworld subcortex queda por "
                  "encima del baseline con los cuatro (Gemini 3 Flash: media de cinco trayectorias)."),
        "f6aria": ("Comparación por motor, baseline a subcortex: llamadas por episodio en opsworld, Gemini 6.7 a "
                   "5.0, Sonnet 6.0 a 4.8, Opus 5.5 a 4.8, Fable 5.4 a 4.6; acciones dañinas, 4 a 2, 6 a 2, 5 a 2 "
                   "y 7 a 2; score de marketworld, −4.1 a 8.4, 1.7 a 9.6, 1.7 a 5.2 y −2.2 a 2.4."),
        "f6t": ["opsworld: llamadas / episodio", "opsworld: acciones dañinas", "marketworld: score medio"],
        "f6rows": ["Gemini 3 Flash", "Claude Sonnet 5", "Claude Opus 5", "Claude Fable 5.1"],
        "f5rows": ["opsworld", "market 21 d", "market 7 d"],
        "baseline": "baseline", "subcortex": "subcortex", "weekly_rule": None,
    },
    "en": {
        "f1cap": ("Figure 1. One turn of the loop: the five plugins hook into the tool-calling cycle. A veto "
                  "does not break the loop (it returns as a tool result and the model re-iterates); a compiled "
                  "habit executes the action without invoking the model."),
        "f1aria": ("Diagram of one turn: before_model with memory, habit and interoception; LLM; after_model "
                   "with winner-take-all; before_tool with validation and veto; tool; after_tool with learning."),
        "f1bm": "memory: precedents & rules · habit: bypass? · interoception: state",
        "f1llm": "proposes action + prediction", "f1am": "gate: a single action (winner-take-all)",
        "f1bt": "valid prediction · veto by default", "f1tool": "tool", "f1toolsub": "real world → observed_effect",
        "f1at": "prediction error → memory (surprise/success) → dopamine → habits → tone",
        "f1habit": "compiled habit", "f1nollm": "without calling the model", "f1auth": "authorized",
        "f1veto": "vetoed / rejected: the model re-iterates", "f1learn": "what was learned feeds the next decision",
        "f2cap": ("Figure 2. opsworld (40 incidents): mean score and model calls per third of the sequence. "
                  "The baseline does not change with experience; subcortex cuts its calls by a third once "
                  "habits are compiled."),
        "f2aria": ("Bars: mean score 46.3 baseline versus 56.3 subcortex; model calls per third: baseline "
                   "6.2, 6.7, 7.2; subcortex 5.9, 4.5, 4.6."),
        "f2t1": "Mean score", "f2t2": "LLM calls per episode, by third", "third": "T",
        "f3cap": ("Figure 3. marketworld: final equity per runner. Top: 40 decisions every 21 days (rings are "
                  "the five subcortex trajectories; the baseline is deterministic). Bottom: the weekly "
                  "variant, where subcortex also beats BTC."),
        "f3aria": ("Final-equity bars: at 21 days, baseline 50.5, subcortex 89.3 plus/minus 19.5 with five "
                   "trajectories between 68 and 121, daily rule 78.6, BTC 95.3. Weekly: baseline 91.4, subcortex 101.6, BTC 95.3."),
        "f3t1": "Final equity from 100 · 40 decisions (21 d)", "f3t2": "Weekly variant · 120 decisions (7 d)",
        "f3base21": "baseline (= rule @ 21 d)", "f3sub": "subcortex · mean of 5", "f3rule": "daily momentum rule",
        "f3btc": "BTC buy & hold", "f3traj": "trajectory",
        "f4cap": ("Figure 4. Ablations in marketworld (one trajectory per variant, same 40 decisions). The "
                  "behavioral signal is the sharp one: without memory, bear-regime behavior returns exactly "
                  "to the baseline's (−3.50 % vs −3.51 %)."),
        "f4aria": ("Ablation bars: equity with the full layer 89.3, no gate 91.5, no habits 74.9, no "
                   "interoception 69.7, no memory 69.2, baseline 50.5; and bear-regime return per decision "
                   "from −1.24 percent with the full layer to −3.50 without memory, equal to the baseline."),
        "f4t1": "Final equity", "f4t2": "Bear regime: return per decision",
        "f4rows": ["full layer (mean of 5)", "no gate", "no habits", "no interoception", "no memory", "baseline"],
        "f4rows2": ["full layer", "no gate", "no habits", "no interoception", "no memory", "baseline"],
        "f5cap": ("Figure 5. Operating cost per episode (model calls, tokens and wall-clock time) per world. "
                  "In opsworld the layer is cheaper than the baseline; in marketworld it pays extra tokens "
                  "and seconds for in-context precedents and evaluation runs."),
        "f5aria": ("Per-world comparison of model calls, tokens and seconds per episode between baseline and "
                   "subcortex; subcortex uses fewer calls in opsworld and more tokens in marketworld."),
        "f5t": ["LLM calls / episode", "Tokens / episode (thousands)", "Seconds / episode"],
        "f6cap": ("Figure 6. Same A/Bs on four engines (n=40 per arm and model). In opsworld the layer cuts "
                  "calls and harmful actions on all four; in marketworld subcortex ends above the baseline "
                  "on all four (Gemini 3 Flash: mean of five trajectories)."),
        "f6aria": ("Per-engine comparison, baseline to subcortex: opsworld calls per episode, Gemini 6.7 to 5.0, "
                   "Sonnet 6.0 to 4.8, Opus 5.5 to 4.8, Fable 5.4 to 4.6; harmful actions, 4 to 2, 6 to 2, 5 to 2 "
                   "and 7 to 2; marketworld score, −4.1 to 8.4, 1.7 to 9.6, 1.7 to 5.2 and −2.2 to 2.4."),
        "f6t": ["opsworld: calls / episode", "opsworld: harmful actions", "marketworld: mean score"],
        "f6rows": ["Gemini 3 Flash", "Claude Sonnet 5", "Claude Opus 5", "Claude Fable 5.1"],
        "f5rows": ["opsworld", "market 21 d", "market 7 d"],
        "baseline": "baseline", "subcortex": "subcortex", "weekly_rule": None,
    },
}


def fig(out: Path, name: str, caption: str, aria: str, w: int, h: int, body: str) -> None:
    svg = (f'<figure><svg viewBox="0 0 {w} {h}" role="img" aria-label="{aria}" '
           f'xmlns="http://www.w3.org/2000/svg" {FS} style="max-width:100%;height:auto;color:var(--ink)">'
           f"{body}</svg><figcaption>{caption}</figcaption></figure>")
    (out / f"{name}.svg").write_text(svg)
    print(out.name, name, "ok")


def hbar(x0, y, label, value_txt, color, maxw, vmax, value, h=18):
    bw = max(2, round(maxw * value / vmax))
    return (f'<text x="{x0 - 8}" y="{y + h - 5}" text-anchor="end" fill="{INK2}">{label}</text>'
            f'<rect x="{x0}" y="{y}" width="{bw}" height="{h}" rx="4" fill="{color}">'
            f'<title>{label}: {value_txt}</title></rect>'
            f'<text x="{x0 + bw + 6}" y="{y + h - 5}" fill="currentColor" font-weight="600">{value_txt}</text>')


def f1(out, t):
    def box(x, y, w, h, t1, t2=""):
        s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="none" stroke="currentColor" stroke-width="1.4"/>'
        s += f'<text x="{x + w / 2}" y="{y + (20 if t2 else h / 2 + 4)}" text-anchor="middle" fill="currentColor" font-weight="600">{t1}</text>'
        if t2:
            words, lines_, cur = t2.split(), [], ""
            for wd in words:
                if len(cur) + len(wd) > 34 and cur:
                    lines_.append(cur); cur = wd
                else:
                    cur = (cur + " " + wd).strip()
            lines_.append(cur)
            for k, ln in enumerate(lines_):
                s += f'<text x="{x + w / 2}" y="{y + 36 + 13 * k}" text-anchor="middle" fill="{INK2}" font-size="10.5">{ln}</text>'
        return s

    def arrow(x1, y1, x2, y2, label="", dy=-6, color="currentColor", dash=""):
        s = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.4" marker-end="url(#a)" {dash}/>'
        if label:
            s += f'<text x="{(x1 + x2) / 2}" y="{(y1 + y2) / 2 + dy}" text-anchor="middle" fill="{INK3}" font-size="11">{label}</text>'
        return s

    b = '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="currentColor"/></marker></defs>'
    b += box(20, 40, 200, 84, "before_model", t["f1bm"])
    b += box(290, 40, 150, 56, "LLM", t["f1llm"])
    b += box(510, 40, 170, 70, "after_model", t["f1am"])
    b += arrow(220, 79, 288, 70)
    b += arrow(440, 66, 508, 66)
    b += arrow(120, 124, 120, 158, t["f1habit"], 14, S1)
    b += f'<path d="M 120 158 C 120 195, 700 195, 750 132" fill="none" stroke="{S1}" stroke-width="1.6" marker-end="url(#a)"/>'
    b += f'<text x="410" y="210" text-anchor="middle" fill="{S1}" font-size="11">{t["f1nollm"]}</text>'
    b += box(510, 134, 170, 70, "before_tool", t["f1bt"])
    b += arrow(595, 110, 595, 132)
    b += box(750, 66, 150, 70, t["f1tool"], t["f1toolsub"])
    b += arrow(680, 158, 748, 118, t["f1auth"], -8)
    b += '<path d="M 510 176 C 390 176, 310 144, 306 98" fill="none" stroke="currentColor" stroke-width="1.2" stroke-dasharray="4 3" marker-end="url(#a)"/>'
    b += f'<text x="368" y="155" text-anchor="middle" fill="{INK3}" font-size="11">{t["f1veto"]}</text>'
    b += box(750, 210, 150, 96, "after_tool", t["f1at"])
    b += arrow(825, 136, 825, 208)
    b += '<path d="M 750 262 C 300 272, 125 248, 117 128" fill="none" stroke="currentColor" stroke-width="1.2" marker-end="url(#a)"/>'
    b += f'<text x="390" y="252" text-anchor="middle" fill="{INK3}" font-size="11">{t["f1learn"]}</text>'
    fig(out, "f1-arquitectura", t["f1cap"], t["f1aria"], 930, 320, b)


def f2(out, t):
    b = f'<text x="20" y="20" fill="currentColor" font-weight="600">{t["f2t1"]}</text>'
    b += hbar(120, 36, t["baseline"], "46.3", NEU, 220, 60, 46.3)
    b += hbar(120, 60, t["subcortex"], "56.3", S1, 220, 60, 56.3)
    b += f'<text x="480" y="20" fill="currentColor" font-weight="600">{t["f2t2"]}</text>'
    base = [6.23, 6.69, 7.21]
    sub = [5.92, 4.54, 4.64]
    x0, w, gap, vmax, hmax, ybase = 500, 44, 26, 8, 120, 170
    for i, (bl, sv) in enumerate(zip(base, sub)):
        gx = x0 + i * (2 * w + gap + 24)
        for j, (v, c) in enumerate(((bl, NEU), (sv, S1))):
            bh = round(hmax * v / vmax)
            name = t["baseline"] if j == 0 else t["subcortex"]
            b += (f'<rect x="{gx + j * (w + 4)}" y="{ybase - bh}" width="{w}" height="{bh}" rx="4" fill="{c}">'
                  f'<title>{name} · {t["third"]}{i + 1}: {v}</title></rect>'
                  f'<text x="{gx + j * (w + 4) + w / 2}" y="{ybase - bh - 6}" text-anchor="middle" fill="currentColor">{v}</text>')
        b += f'<text x="{gx + w + 2}" y="{ybase + 18}" text-anchor="middle" fill="{INK2}">{t["third"]}{i + 1}</text>'
    b += f'<line x1="490" y1="{ybase}" x2="900" y2="{ybase}" stroke="{LINE}"/>'
    b += (f'<rect x="120" y="100" width="12" height="12" rx="3" fill="{NEU}"/><text x="138" y="110" fill="{INK2}">{t["baseline"]}</text>'
          f'<rect x="120" y="120" width="12" height="12" rx="3" fill="{S1}"/><text x="138" y="130" fill="{INK2}">{t["subcortex"]}</text>')
    fig(out, "f2-opsworld", t["f2cap"], t["f2aria"], 920, 200, b)


def f3(out, t):
    rows = [(t["f3base21"], 50.5, NEU, ""), (t["f3sub"], 89.3, S1, "±19.5"),
            (t["f3rule"], 78.6, REF, ""), (t["f3btc"], 95.3, REF, "")]
    b = f'<text x="20" y="20" fill="currentColor" font-weight="600">{t["f3t1"]}</text>'
    y = 40
    for label, v, c, err in rows:
        b += hbar(230, y, label, f"{v}{' ' + err if err else ''}", c, 420, 130, v)
        y += 28
    for tv in (90.9, 121.0, 83.2, 68.3, 83.2):
        x = 230 + round(420 * tv / 130)
        b += f'<circle cx="{x}" cy="77" r="4" fill="none" stroke="{S1}" stroke-width="1.6"><title>{t["f3traj"]}: {tv}</title></circle>'
    b += f'<line x1="230" y1="152" x2="880" y2="152" stroke="{LINE}"/>'
    b += f'<text x="20" y="185" fill="currentColor" font-weight="600">{t["f3t2"]}</text>'
    y = 200
    for label, v, c in ((t["baseline"], 91.4, NEU), (t["subcortex"], 101.6, S1), (t["f3btc"], 95.3, REF)):
        b += hbar(230, y, label, str(v), c, 420, 130, v)
        y += 28
    fig(out, "f3-marketworld", t["f3cap"], t["f3aria"], 920, 300, b)


def f4(out, t):
    b = f'<text x="20" y="20" fill="currentColor" font-weight="600">{t["f4t1"]}</text>'
    vals = [89.3, 91.5, 74.9, 69.7, 69.2, 50.5]
    y = 36
    for label, v in zip(t["f4rows"], vals):
        b += hbar(230, y, label, str(v), NEU if label == "baseline" else S1, 250, 100, v)
        y += 26
    b += f'<text x="560" y="20" fill="currentColor" font-weight="600">{t["f4t2"]}</text>'
    vals2 = [-1.24, -1.28, -1.22, -2.27, -3.50, -3.51]
    y = 36
    for label, v in zip(t["f4rows2"], vals2):
        c = NEU if label == "baseline" else S1
        bw = round(200 * abs(v) / 3.6)
        b += (f'<text x="{700 - 8}" y="{y + 13}" text-anchor="end" fill="{INK2}">{label}</text>'
              f'<rect x="700" y="{y}" width="{bw}" height="18" rx="4" fill="{c}"><title>{label}: {v} %</title></rect>'
              f'<text x="{700 + bw + 6}" y="{y + 13}" fill="currentColor" font-weight="600">{v} %</text>')
        y += 26
    fig(out, "f4-ablaciones", t["f4cap"], t["f4aria"], 920, 210, b)


def f5(out, t):
    data = [[("opsworld", 6.7, 5.0), ("market 21 d", 4.1, 5.0), ("market 7 d", 3.9, 4.9)],
            [("opsworld", 11.6, 12.1), ("market 21 d", 8.0, 15.9), ("market 7 d", 7.3, 16.2)],
            [("opsworld", 12.2, 10.2), ("market 21 d", 9.3, 14.3), ("market 7 d", 14.4, 17.3)]]
    b = ""
    for gi, (title, rows) in enumerate(zip(t["f5t"], data)):
        gx = 20 + gi * 300
        b += f'<text x="{gx}" y="20" fill="currentColor" font-weight="600">{title}</text>'
        vmax = max(max(a, s) for _, a, s in rows) * 1.15
        y = 40
        for (label, a, s), rlab in zip(rows, t["f5rows"]):
            for v, c, name in ((a, NEU, t["baseline"]), (s, S1, t["subcortex"])):
                bw = round(150 * v / vmax)
                b += (f'<rect x="{gx + 90}" y="{y}" width="{bw}" height="12" rx="3" fill="{c}">'
                      f'<title>{rlab} · {name}: {v}</title></rect>'
                      f'<text x="{gx + 90 + bw + 5}" y="{y + 10}" fill="currentColor" font-size="11">{v}</text>')
                y += 15
            b += f'<text x="{gx + 82}" y="{y - 18}" text-anchor="end" fill="{INK2}" font-size="11">{rlab}</text>'
            y += 10
    b += (f'<rect x="20" y="180" width="12" height="12" rx="3" fill="{NEU}"/><text x="38" y="190" fill="{INK2}">{t["baseline"]}</text>'
          f'<rect x="110" y="180" width="12" height="12" rx="3" fill="{S1}"/><text x="128" y="190" fill="{INK2}">{t["subcortex"]}</text>')
    fig(out, "f5-eficiencia", t["f5cap"], t["f5aria"], 920, 205, b)


def f6(out, t):
    b = ""
    groups = [(t["f6t"][0], [(6.7, 5.0), (6.0, 4.8), (5.5, 4.8), (5.4, 4.6)], 8.0),
              (t["f6t"][1], [(4, 2), (6, 2), (5, 2), (7, 2)], 8.0)]
    for gi, (title, rows, vmax) in enumerate(groups):
        gx = 20 + gi * 300
        b += f'<text x="{gx}" y="20" fill="currentColor" font-weight="600">{title}</text>'
        y = 40
        for (a, s), rlab in zip(rows, t["f6rows"]):
            for v, c, name in ((a, NEU, t["baseline"]), (s, S1, t["subcortex"])):
                bw = round(140 * v / vmax)
                b += (f'<rect x="{gx + 108}" y="{y}" width="{bw}" height="12" rx="3" fill="{c}">'
                      f'<title>{rlab} · {name}: {v}</title></rect>'
                      f'<text x="{gx + 108 + bw + 5}" y="{y + 10}" fill="currentColor" font-size="11">{v}</text>')
                y += 15
            b += f'<text x="{gx + 102}" y="{y - 18}" text-anchor="end" fill="{INK2}" font-size="10">{rlab}</text>'
            y += 10
    gx, k = 620, 10
    zx = gx + 152
    b += f'<text x="{gx}" y="20" fill="currentColor" font-weight="600">{t["f6t"][2]}</text>'
    b += f'<line x1="{zx}" y1="32" x2="{zx}" y2="196" stroke="{LINE}"/>'
    y = 40
    for (a, s), rlab in zip([(-4.1, 8.4), (1.7, 9.6), (1.7, 5.2), (-2.2, 2.4)], t["f6rows"]):
        for v, c, name in ((a, NEU, t["baseline"]), (s, S1, t["subcortex"])):
            bw = round(abs(v) * k)
            x = zx - bw if v < 0 else zx
            b += (f'<rect x="{x}" y="{y}" width="{max(bw, 2)}" height="12" rx="3" fill="{c}">'
                  f'<title>{rlab} · {name}: {v}</title></rect>')
            tx, anchor = (zx - bw - 5, "end") if v < 0 else (zx + bw + 5, "start")
            b += f'<text x="{tx}" y="{y + 10}" text-anchor="{anchor}" fill="currentColor" font-size="11">{v}</text>'
            y += 15
        b += f'<text x="{gx + 102}" y="{y - 18}" text-anchor="end" fill="{INK2}" font-size="10">{rlab}</text>'
        y += 10
    b += (f'<rect x="20" y="208" width="12" height="12" rx="3" fill="{NEU}"/><text x="38" y="218" fill="{INK2}">{t["baseline"]}</text>'
          f'<rect x="110" y="208" width="12" height="12" rx="3" fill="{S1}"/><text x="128" y="218" fill="{INK2}">{t["subcortex"]}</text>')
    fig(out, "f6-motores", t["f6cap"], t["f6aria"], 920, 235, b)


for lang, t in T.items():
    out = BASE / lang
    out.mkdir(parents=True, exist_ok=True)
    f1(out, t); f2(out, t); f3(out, t); f4(out, t); f5(out, t); f6(out, t)
