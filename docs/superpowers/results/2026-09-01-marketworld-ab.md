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

## Replicación (misma noche, mismas 40 decisiones, ambas variantes de nuevo)

| | baseline run 1 | baseline run 2 | subcortex run 1 | subcortex run 2 |
|---|---|---|---|---|
| equity final | 50.5 | 50.5 | 90.9 | **121.0** |
| score medio | −4.1 | −4.1 | 6.6 | 16.2 |
| suma de retornos 21d | −16.2 % | −16.2 % | +26.3 % | +65.0 % |
| `worsens` | 13 | 13 | 9 | 7 |
| decisiones iguales entre corridas | 40/40 | | 24/40 | |
| régimen bajista (19 decisiones): media | −3.5 % | −3.5 % | −1.2 % | −1.3 % |
| régimen bajista: `hold` / `follow` | 9 / 5 | 9 / 5 | 12 / 4 | 13 / 2 |
| episodios / reglas / hábitos | — | — | 29 / 4 / 0 | 24 / 6 / **2** |

Datos: `results-market-run1.json`, `results-market-run2.json` (+ `-refs.json`).

Tres cosas quedan claras con dos trayectorias:

1. **El baseline es determinista**: las 40 decisiones fueron idénticas en las dos corridas
   (temperatura por defecto, prompt simple). Es exactamente la regla a 21 días, 50.5.
2. **La dirección de subcortex se sostiene; la magnitud no**: 90.9 y 121.0, con solo 24 de 40
   decisiones iguales entre corridas. Lo que se repite es la *conducta en régimen bajista* —
   mantener en vez de rotar, −1.2 % y −1.3 % por decisión contra −3.5 % del baseline— y ahí se
   gana toda la diferencia. La varianza entre corridas de subcortex es del tamaño de la mitad de
   su ventaja: la ventaja es real, el número exacto no.
3. **En la corrida 2 aparecieron hábitos y reglas más nítidas.** Hábito `hold` en la clase de
   escena bajista dominante (fuerza 0.85, 4/0) y `follow_momentum` en una alcista (3/1, debilitado
   a 0.4 tras un fallo: la des-habituación funcionó). Reglas destiladas: *"con btc_trend=up,
   follow_momentum tiende a resolves (n=5)"*, *"con btc_trend=down, go_cash tiende a no_change
   (n=4)"*, *"con dispersion=wide, follow_momentum tiende a worsens (n=5)"*. Es la estructura del
   problema, aprendida de 40 muestras sin etiquetas.

Por qué la varianza: la memoria es dependiente del camino. Una decisión distinta en la semana 3
cambia qué episodios existen en la semana 10 y, por lo tanto, qué precedentes ve el modelo.
El baseline no tiene ese canal, así que no varía. Con 3–5 trayectorias por variante se podría
reportar media ± desvío; con dos, lo honesto es: subcortex terminó entre 91 y 121 contra 50.5,
78.6 (regla diaria) y 95.3 (BTC), y en ambas corridas la ganancia vino del mismo lugar.

## Cinco trayectorias de subcortex (loop, 2026-09-01)

El baseline es determinista (50.5); subcortex se corrió cinco veces sobre las mismas 40 decisiones.

| trayectoria | equity final | régimen bajista: media por decisión | hold / follow en bajista | episodios | hábitos disparados |
|---|---|---|---|---|---|
| run 1 | 90.9 | −1.22 % | 12 / 4 | 29 | 0 |
| run 2 | 121.0 | −1.29 % | 13 / 2 | 24 | 2 |
| run 3 | 83.2 | −1.21 % | 13 / 3 | 27 | 1 |
| run 4 | 68.3 | −1.29 % | 12 / 4 | 26 | 2 |
| run 5 | 83.2 | −1.21 % | 13 / 3 | 30 | 1 |
| **media ± desvío** | **89.3 ± 19.5** | **−1.24 %** | | | |
| baseline | 50.5 | −3.51 % | 9 / 5 | — | — |

- **5 de 5 por encima del baseline** (mínimo 68.3 vs 50.5); 4 de 5 por encima de la regla diaria (78.6);
  2 de 5 por encima de BTC (95.3).
- **La conducta en régimen bajista es casi idéntica en las cinco** (−1.21 a −1.29 % por decisión, 12–13
  `hold` de 19): ese es el mecanismo, y es estable. La dispersión del equity (68–121) viene de las
  decisiones alcistas, donde subcortex a veces sigue la regla y a veces no; ahí la varianza del
  modelo domina.
- Datos: `results-market-run{1..5}.json`.

## Variante semanal: 120 decisiones cada 7 días, horizonte 7 (loop, 2026-09-01)

Misma ventana (mar-2024 → jun-2026), tres veces más decisiones y consecuencias a una semana.
Datos: `results-market-weekly.json`.

| corredor | equity final | acciones | régimen bajista (58 decisiones): media | episodios / hábitos disparados / vetos |
|---|---|---|---|---|
| baseline | 91.4 | follow 48 · hold 65 · cash 5 | −0.62 % | — |
| **subcortex** | **101.6** | follow 41 · hold 73 · cash 6 | **+0.16 %** | 77 / 11 / 5 |
| regla al ritmo del agente (semanal) | 90.9 | | | |
| regla diaria | 78.6 | | | |
| BTC | 95.3 | | | |

- Con cadencia semanal el baseline ya no es la regla pura (91.4 vs 90.9, casi): el target semanal
  cambia poco y el modelo mantiene 65 veces. subcortex igual lo supera (+10) y **queda por encima de
  BTC**, que a 21 días no había logrado en 3 de 5 trayectorias.
- **Los hábitos se dispararon 11 veces** (2 compilados, 6 reglas) — con 120 muestras la repetición
  por clase de escena alcanza para que el putamen trabaje; a 40 decisiones apenas 0–2.
- La ventaja sigue viniendo del régimen bajista (+0.16 % vs −0.62 % por decisión).
- Una sola trayectoria: misma advertencia de varianza que a 21 días.

## Ablaciones y variantes (loop, 2026-09-01; una trayectoria cada una, mismas 40 decisiones)

| variante | equity final | régimen bajista: media por decisión | hábitos disparados |
|---|---|---|---|
| capa completa (5 trayectorias) | 89.3 ± 19.5 | −1.24 % | 0–2 |
| **sin memoria** | **69.2** | **−3.50 %** | 11 |
| sin interocepción | 69.7 | −2.27 % | 2 |
| sin hábitos | 74.9 | −1.22 % | 0 |
| sin gate | 91.5 | −1.28 % | 1 |
| +hist (confianza por historial) | 92.3 | −1.22 % | 1 |
| +recon (cingulado) | 69.8 | −1.22 % | 1 |
| +llmrules | *incompleta (14/40: créditos agotados)* | — | — |
| baseline | 50.5 | −3.51 % | — |

Lectura, con la cautela de que cada ablación es una sola trayectoria contra una media con σ ≈ 20:

- **La memoria es el mecanismo.** Sin ella, la conducta bajista vuelve exactamente a la del
  baseline (−3.50 % vs −3.51 %): sin precedentes en contexto, el modelo rota como la regla.
  Es la señal más limpia porque es conductual, no de equity. Curioso: sin memoria los hábitos se
  dispararon 11 veces (la dopamina sigue funcionando), y no alcanzó.
- **La interocepción aporta** (69.7, bajista −2.27 %): sin el estado interno el modelo pierde
  parte de la prudencia, aunque conserva los precedentes.
- **Los hábitos aportan algo** (74.9) y **el gate nada en este dominio** (91.5, dentro del rango
  de la capa completa) — consistente con los 0 vetos de todas las corridas: acá no hay acciones
  irreversibles que frenar.
- **+hist y +recon quedan dentro del rango** de la capa completa: con una trayectoria no se puede
  afirmar efecto; quedan implementados, testeados y medibles cuando haya presupuesto.
- El seed extra de opsworld y el segundo modelo no corrieron (créditos agotados); quedan como
  pendientes documentados.

## Advertencias

- **n = 40 decisiones, cinco trayectorias de subcortex, una de baseline (determinista).** La dirección es
  robusta (5/5); la magnitud tiene desvío ~20 puntos. Citar **89 ± 20 contra 50.5**, no un valor.
- El baseline reproduce la regla a 21 días, que en esta ventana es la peor referencia. Contra la
  regla diaria (78.6) la ventaja es de 12 puntos; contra BTC (95.3), subcortex pierde.
- Retorno a 21 días con umbrales ±6 %: 24 de 40 decisiones caen en `no_change`, lo que limita
  la señal para dopamina y hábitos.
