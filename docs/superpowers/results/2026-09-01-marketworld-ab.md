# Proyecto Momentum en replay — marketworld (2026-09-01)

**Setup:** velas diarias reales de Binance para los 10 majors del paper trader, 40 decisiones cada
21 días (2024-03-24 → 2026-06-21: fase alcista 2024-25 y bajista 2025-26), consecuencia medida
a 21 días (hasta la decisión siguiente), cartera persistente (replay secuencial), costos 0.15 % por
lado. El agente no ve fechas. Mismo `LlmAgent` (gemini-3-flash-preview) con y sin `attach()`.
Datos: `results-market.json` (+ `-refs.json`). La corrida 1 (horizonte 7 días, inválida por
horizonte < espaciado) quedó en `results-market-h7.json`.

## Referencias sin LLM (misma ventana)

| corredor | equity final (desde 100) |
|---|---|
| regla momentum **diaria** (la del paper trader real) | 78.6 |
| regla momentum al ritmo del agente (cada 21 días) | 50.5 |
| BTC buy & hold | 95.3 |

Ventana dura para momentum: la regla pierde. Lo que se mide no es "ganar", es cómo decide cada
variante frente a la misma secuencia.

## Resultado

| métrica | baseline | subcortex |
|---|---|---|
| **equity final** | **50.5** | **90.9** |
| score medio (retorno 21d × 10) | −4.1 | 6.6 |
| suma de retornos a 21 días | −16.2 % | +26.3 % |
| decisiones con retorno > 0 | 10/40 | 9/40 |
| `worsens` (< −6 %) | 13 | 9 |
| llamadas al LLM / decisión | 4.1 | 5.0 |
| acciones | follow 26 · hold 9 · cash 5 | follow 13 · **hold 22** · cash 4 · rotate 1 |
| episodios escritos / reglas / hábitos | — | 29 / 4 / 0 |
| score por tercios | 58 → −11 → −55 | 27 → 20 → −24 |

El baseline terminó **exactamente en 50.5, igual que la regla al ritmo del agente**: siguió la regla
26 veces y sus `hold`/`go_cash` no cambiaron el resultado. subcortex terminó en 90.9: por encima
de la regla diaria (78.6) y por debajo de BTC (95.3).

## Por régimen

| régimen | n | baseline → retorno 21d | subcortex → retorno 21d | regla |
|---|---|---|---|---|
| down_low_high | 11 | follow 3 · cash 3 · hold 5 → **−2.4 %** | follow 1 · **hold 8** · cash 2 → **−0.3 %** | −2.4 % |
| down_low_low | 7 | follow 2 · cash 2 · hold 3 → −5.8 % | follow 2 · cash 1 · hold 4 → −2.8 % | −5.8 % |
| up_high_low | 10 | follow 10 → +8.6 % | follow 2 · hold 7 · cash 1 → +8.9 % | +8.6 % |
| up_mid_low | 3 | follow 3 → +6.7 % | hold 2 · follow 1 → +7.2 % | +6.7 % |
| up_high_high | 2 | follow 2 → −13.4 % | follow 1 · hold 1 → −19.5 % | −13.4 % |
| up_low_low | 2 | follow 2 → −15.5 % | follow 1 · rotate 1 → −11.9 % | −15.5 % |
| otros (3 regímenes) | 5 | — | — | — |

La diferencia está en los **regímenes bajistas**: subcortex mantuvo la cartera (que ya estaba
mayormente en cash o en pocos activos) en vez de seguir rotando por la regla, y ahí la regla pierde
2–6 % por decisión. En los alcistas ambos rinden igual (mantener las posiciones momentum ya
tomadas equivale a seguir la regla).

## Qué aprendió la capa

- **29 episodios** escritos (sorpresas y éxitos) y **4 reglas destiladas** por `consolidate()`. Dos de
  ellas son exactamente la estructura del problema, descubiertas sin que nadie se la dijera:
  - *"Con dispersion=wide, follow_momentum tiende a worsens (n=4)."*
  - *"Con btc_trend=up, dispersion=narrow, follow_momentum tiende a resolves (n=3)."*
- **0 hábitos y 0 vetos.** La dopamina de `hold` en el régimen bajista dominante quedó en 2 éxitos / 6
  fallos: mantener en un mercado que cae más de 6 % en 21 días sigue siendo `worsens`. El gate
  nunca intervino. La ganancia vino de la **memoria** (precedentes y reglas en contexto) y la
  **interocepción**, no del veto ni del hábito: la corteza leyendo su propia experiencia.

## Cambios de diseño que salieron de la corrida 1

1. **Horizonte = espaciado.** Medir a 7 días con decisiones cada 21 dejaba 14 días de deriva
   invisible; el score y el equity se contradecían (baseline mejor a 7 días, peor en equity).
2. **Cumplir la expectativa de `no_change` es éxito** (`is_success`). Antes `hold` en régimen bajista
   contaba 0 éxitos / 8 fallos: mantener en 0 % cuando todo cae es acertar. Es el error de
   predicción del paso 15 del ensayo, no la recompensa absoluta.

## Advertencias

- **n = 40, una trayectoria, un modelo no determinista.** Un solo `follow_momentum` en la semana
  equivocada mueve el equity final 15 puntos. La diferencia de 40 puntos es grande pero no es
  una estadística. Replicación en curso (`results-market-run2.json`); ver sección siguiente.
- El baseline reproduce la regla a 21 días, que en esta ventana es la peor referencia. Contra la
  regla diaria (78.6) la ventaja es de 12 puntos; contra BTC (95.3), subcortex pierde.
- Retorno a 21 días con umbrales ±6 %: 24 de 40 decisiones caen en `no_change`, lo que limita
  la señal para dopamina y hábitos.
