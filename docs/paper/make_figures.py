"""Genera las figuras del paper como SVG embebible (una por archivo, en figures/).

Colores por rol vía CSS vars definidas en la hoja del paper (--viz-s1 subcortex,
--viz-neutral baseline, --viz-ref referencias, tinta en currentColor): las figuras
se adaptan al tema claro/oscuro sin duplicarse.
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

S1 = "var(--viz-s1)"        # subcortex
NEU = "var(--viz-neutral)"  # baseline
REF = "var(--viz-ref)"      # referencias sin LLM
INK2 = "var(--ink-2)"
INK3 = "var(--ink-3)"
LINE = "var(--line)"
FS = 'font-family="IBM Plex Sans, sans-serif" font-size="12"'


def fig(name: str, caption: str, aria: str, w: int, h: int, body: str) -> None:
    svg = (f'<figure><svg viewBox="0 0 {w} {h}" role="img" aria-label="{aria}" '
           f'xmlns="http://www.w3.org/2000/svg" {FS} style="max-width:100%;height:auto;color:var(--ink)">'
           f"{body}</svg><figcaption>{caption}</figcaption></figure>")
    (OUT / f"{name}.svg").write_text(svg)
    print(name, "ok")


def hbar(x0, y, w, label, value_txt, color, maxw, vmax, value, h=18):
    """Barra horizontal fina con extremo redondeado y etiqueta directa."""
    bw = max(2, round(maxw * value / vmax))
    return (f'<text x="{x0 - 8}" y="{y + h - 5}" text-anchor="end" fill="{INK2}">{label}</text>'
            f'<rect x="{x0}" y="{y}" width="{bw}" height="{h}" rx="4" fill="{color}">'
            f'<title>{label}: {value_txt}</title></rect>'
            f'<text x="{x0 + bw + 6}" y="{y + h - 5}" fill="currentColor" font-weight="600">{value_txt}</text>')


# ── F1: arquitectura de un turno ────────────────────────────────────────────────
def f1():
    def box(x, y, w, h, t1, t2="", stroke="currentColor", dash=""):
        s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="none" stroke="{stroke}" stroke-width="1.4" {dash}/>'
        s += f'<text x="{x + w / 2}" y="{y + (20 if t2 else h / 2 + 4)}" text-anchor="middle" fill="currentColor" font-weight="600">{t1}</text>'
        if t2:
            s += f'<text x="{x + w / 2}" y="{y + 36}" text-anchor="middle" fill="{INK2}" font-size="11">{t2}</text>'
        return s

    def arrow(x1, y1, x2, y2, label="", dy=-6, color="currentColor", dash=""):
        s = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.4" marker-end="url(#a)" {dash}/>'
        if label:
            s += f'<text x="{(x1 + x2) / 2}" y="{(y1 + y2) / 2 + dy}" text-anchor="middle" fill="{INK3}" font-size="11">{label}</text>'
        return s

    b = '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="currentColor"/></marker></defs>'
    # fila superior: before_model → LLM → after_model
    b += box(20, 40, 190, 78, "before_model", "memoria: precedentes y reglas · hábito: ¿bypass? · interocepción: estado")
    b += box(280, 40, 150, 52, "LLM", "propone acción + predicción")
    b += box(500, 40, 170, 66, "after_model", "gate: una sola acción (winner-take-all)")
    b += arrow(210, 79, 278, 70, "")
    b += arrow(430, 66, 498, 66, "")
    # hábito saltea el LLM
    b += arrow(115, 118, 115, 156, "hábito compilado", 14, S1)
    b += f'<path d="M 115 156 C 115 190, 690 190, 740 130" fill="none" stroke="{S1}" stroke-width="1.6" marker-end="url(#a)"/>'
    b += f'<text x="400" y="205" text-anchor="middle" fill="{S1}" font-size="11">sin llamar al modelo</text>'
    # before_tool → tool
    b += box(500, 130, 170, 66, "before_tool", "predicción válida · veto por defecto")
    b += arrow(585, 106, 585, 128, "")
    b += box(740, 80, 150, 66, "herramienta", "mundo real → observed_effect")
    b += arrow(670, 150, 738, 122, "autorizada", -8)
    # veto re-itera
    b += f'<path d="M 500 170 C 380 170, 300 140, 296 94" fill="none" stroke="currentColor" stroke-width="1.2" stroke-dasharray="4 3" marker-end="url(#a)"/>'
    b += f'<text x="360" y="150" text-anchor="middle" fill="{INK3}" font-size="11">vetada / rechazada: el modelo re-itera</text>'
    # after_tool learning
    b += box(740, 200, 150, 92, "after_tool", "error de predicción → memoria (sorpresa/éxito) → dopamina → hábitos → tono")
    b += arrow(815, 146, 815, 198, "")
    b += f'<path d="M 740 250 C 300 260, 120 240, 112 122" fill="none" stroke="currentColor" stroke-width="1.2" marker-end="url(#a)"/>'
    b += f'<text x="380" y="242" text-anchor="middle" fill="{INK3}" font-size="11">lo aprendido alimenta la próxima decisión</text>'
    fig("f1-arquitectura", "Figura 1. Un turno del bucle: los cinco plugins se cuelgan de los hooks "
        "del ciclo de tool-calls. Un veto no rompe el bucle (vuelve como resultado y el modelo re-itera); "
        "un hábito compilado ejecuta la acción sin invocar al modelo.",
        "Diagrama del flujo de un turno: before_model con memoria, hábito e interocepción; LLM; "
        "after_model con winner-take-all; before_tool con validación y veto; herramienta; after_tool con aprendizaje.",
        920, 310, b)


# ── F2: opsworld ────────────────────────────────────────────────────────────────
def f2():
    b = f'<text x="20" y="20" fill="currentColor" font-weight="600">Score medio</text>'
    b += hbar(120, 36, 0, "baseline", "46.3", NEU, 220, 60, 46.3)
    b += hbar(120, 60, 0, "subcortex", "56.3", S1, 220, 60, 56.3)
    b += f'<text x="480" y="20" fill="currentColor" font-weight="600">Llamadas al LLM por episodio, por tercio</text>'
    base = [6.23, 6.69, 7.21]
    sub = [5.92, 4.54, 4.64]
    x0, w, gap, vmax, hmax, ybase = 500, 44, 26, 8, 120, 170
    for i, (bl, sv) in enumerate(zip(base, sub)):
        gx = x0 + i * (2 * w + gap + 24)
        for j, (v, c) in enumerate(((bl, NEU), (sv, S1))):
            bh = round(hmax * v / vmax)
            b += (f'<rect x="{gx + j * (w + 4)}" y="{ybase - bh}" width="{w}" height="{bh}" rx="4" fill="{c}">'
                  f'<title>{"baseline" if j == 0 else "subcortex"} · tercio {i + 1}: {v}</title></rect>'
                  f'<text x="{gx + j * (w + 4) + w / 2}" y="{ybase - bh - 6}" text-anchor="middle" fill="currentColor">{v}</text>')
        b += f'<text x="{gx + w + 2}" y="{ybase + 18}" text-anchor="middle" fill="{INK2}">T{i + 1}</text>'
    b += f'<line x1="490" y1="{ybase}" x2="900" y2="{ybase}" stroke="{LINE}"/>'
    # leyenda
    b += (f'<rect x="120" y="100" width="12" height="12" rx="3" fill="{NEU}"/><text x="138" y="110" fill="{INK2}">baseline</text>'
          f'<rect x="120" y="120" width="12" height="12" rx="3" fill="{S1}"/><text x="138" y="130" fill="{INK2}">subcortex</text>')
    fig("f2-opsworld", "Figura 2. opsworld (40 incidentes): score medio y llamadas al modelo por tercio de la "
        "secuencia. El baseline no cambia con la experiencia; subcortex baja un tercio sus llamadas cuando los "
        "hábitos ya están compilados.",
        "Barras: score medio 46.3 baseline contra 56.3 subcortex; llamadas al modelo por tercio: baseline "
        "6.2, 6.7, 7.2; subcortex 5.9, 4.5, 4.6.", 920, 200, b)


# ── F3: marketworld equity ─────────────────────────────────────────────────────
def f3():
    rows = [("baseline (= regla c/21 d)", 50.5, NEU, ""), ("subcortex · media de 5", 89.3, S1, "±19.5"),
            ("regla momentum diaria", 78.6, REF, ""), ("BTC buy & hold", 95.3, REF, "")]
    b = f'<text x="20" y="20" fill="currentColor" font-weight="600">Equity final desde 100 · 40 decisiones (21 d)</text>'
    y = 40
    for label, v, c, err in rows:
        b += hbar(230, y, 0, label, f"{v}{' ' + err if err else ''}", c, 420, 130, v)
        y += 28
    # trayectorias individuales como puntos
    for tv in (90.9, 121.0, 83.2, 68.3, 83.2):
        x = 230 + round(420 * tv / 130)
        b += f'<circle cx="{x}" cy="77" r="4" fill="none" stroke="{S1}" stroke-width="1.6"><title>trayectoria: {tv}</title></circle>'
    b += f'<line x1="230" y1="152" x2="880" y2="152" stroke="{LINE}"/>'
    b += f'<text x="20" y="185" fill="currentColor" font-weight="600">Variante semanal · 120 decisiones (7 d)</text>'
    y = 200
    for label, v, c in (("baseline", 91.4, NEU), ("subcortex", 101.6, S1), ("BTC buy & hold", 95.3, REF)):
        b += hbar(230, y, 0, label, str(v), c, 420, 130, v)
        y += 28
    fig("f3-marketworld", "Figura 3. marketworld: equity final por corredor. Arriba, 40 decisiones cada 21 días "
        "(los aros son las cinco trayectorias de subcortex; el baseline es determinista). Abajo, la variante "
        "semanal, donde subcortex supera también a BTC.",
        "Barras de equity final: a 21 días, baseline 50.5, subcortex 89.3 más menos 19.5 con cinco "
        "trayectorias entre 68 y 121, regla diaria 78.6, BTC 95.3. Semanal: baseline 91.4, subcortex 101.6, BTC 95.3.",
        920, 300, b)


# ── F4: ablaciones ─────────────────────────────────────────────────────────────
def f4():
    b = f'<text x="20" y="20" fill="currentColor" font-weight="600">Equity final</text>'
    y = 36
    for label, v, c in (("capa completa (media de 5)", 89.3, S1), ("sin gate", 91.5, S1), ("sin hábitos", 74.9, S1),
                        ("sin interocepción", 69.7, S1), ("sin memoria", 69.2, S1), ("baseline", 50.5, NEU)):
        b += hbar(230, y, 0, label, str(v), c, 250, 100, v)
        y += 26
    b += f'<text x="560" y="20" fill="currentColor" font-weight="600">Régimen bajista: retorno por decisión</text>'
    y = 36
    for label, v, c in (("capa completa", -1.24, S1), ("sin gate", -1.28, S1), ("sin hábitos", -1.22, S1),
                        ("sin interocepción", -2.27, S1), ("sin memoria", -3.50, S1), ("baseline", -3.51, NEU)):
        bw = round(200 * abs(v) / 3.6)
        b += (f'<text x="{700 - 8}" y="{y + 13}" text-anchor="end" fill="{INK2}">{label}</text>'
              f'<rect x="700" y="{y}" width="{bw}" height="18" rx="4" fill="{c}"><title>{label}: {v} %</title></rect>'
              f'<text x="{700 + bw + 6}" y="{y + 13}" fill="currentColor" font-weight="600">{v} %</text>')
        y += 26
    fig("f4-ablaciones", "Figura 4. Ablaciones en marketworld (una trayectoria por variante, mismas 40 "
        "decisiones). La señal conductual es la nítida: sin memoria, el comportamiento en régimen bajista "
        "vuelve exactamente al del baseline (−3.50 % contra −3.51 %).",
        "Barras de ablación: equity con capa completa 89.3, sin gate 91.5, sin hábitos 74.9, sin "
        "interocepción 69.7, sin memoria 69.2, baseline 50.5; y retorno por decisión en régimen bajista "
        "de −1.24 por ciento con capa completa a −3.50 sin memoria, igual al baseline.", 920, 210, b)


# ── F5: eficiencia ─────────────────────────────────────────────────────────────
def f5():
    grupos = [("Llamadas al LLM / episodio", 1, [("opsworld", 6.7, 5.0), ("market 21 d", 4.1, 5.0), ("market 7 d", 3.9, 4.9)]),
              ("Tokens / episodio (miles)", 1, [("opsworld", 11.6, 12.1), ("market 21 d", 8.0, 15.9), ("market 7 d", 7.3, 16.2)]),
              ("Segundos / episodio", 1, [("opsworld", 12.2, 10.2), ("market 21 d", 9.3, 14.3), ("market 7 d", 14.4, 17.3)])]
    b = ""
    x0 = 20
    for gi, (title, _, rows) in enumerate(grupos):
        gx = x0 + gi * 300
        b += f'<text x="{gx}" y="20" fill="currentColor" font-weight="600">{title}</text>'
        vmax = max(max(a, s) for _, a, s in rows) * 1.15
        y = 40
        for label, a, s in rows:
            for v, c, name in ((a, NEU, "baseline"), (s, S1, "subcortex")):
                bw = round(150 * v / vmax)
                b += (f'<rect x="{gx + 90}" y="{y}" width="{bw}" height="12" rx="3" fill="{c}">'
                      f'<title>{label} · {name}: {v}</title></rect>'
                      f'<text x="{gx + 90 + bw + 5}" y="{y + 10}" fill="currentColor" font-size="11">{v}</text>')
                y += 15
            b += f'<text x="{gx + 82}" y="{y - 18}" text-anchor="end" fill="{INK2}" font-size="11">{label}</text>'
            y += 10
    b += (f'<rect x="20" y="180" width="12" height="12" rx="3" fill="{NEU}"/><text x="38" y="190" fill="{INK2}">baseline</text>'
          f'<rect x="110" y="180" width="12" height="12" rx="3" fill="{S1}"/><text x="128" y="190" fill="{INK2}">subcortex</text>')
    fig("f5-eficiencia", "Figura 5. Costo operativo por episodio (llamadas al modelo, tokens y tiempo de pared) "
        "por mundo. En opsworld la capa es más barata que el baseline; en marketworld paga tokens y segundos "
        "extra por los precedentes en contexto y las corridas de evaluación.",
        "Comparación por mundo de llamadas al modelo, tokens y segundos por episodio entre baseline y "
        "subcortex; subcortex usa menos llamadas en opsworld y más tokens en marketworld.", 920, 205, b)


f1(); f2(); f3(); f4(); f5()
