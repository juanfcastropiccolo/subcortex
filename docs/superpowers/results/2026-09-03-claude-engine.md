# Claude como motor: mismas pruebas, otro cerebro (2026-09-03)

**Setup:** adaptador `adapters/claude_code_llm.py` (`ClaudeCodeLlm`): Claude Code headless
(`claude -p --output-format json --json-schema … --tools "" --max-turns 1`) como `BaseLlm` de ADK,
corriendo dentro del plan del usuario (sin API key ni crédito). Modelo `claude-code:sonnet`
(Sonnet 5). Mismos experimentos que con Gemini: opsworld A/B (`--n 40 --seed 42`) y
marketworld A/B (`--n 40`, horizonte 21d), `consolidate()` cada 10 episodios en el brazo
subcortex. Datos crudos en `results-ops-cc.json` y `results-market-cc.json`.

## El bug del primer intento: episodios vacíos

La primera corrida se cortó a los 12 episodios: 3 de 12 (25 %) terminaban con `steps=0 llm=1` —
Claude respondía `kind=text` en el primer turno, sin ejecutar ninguna herramienta. Gemini nunca
hizo esto: el canal nativo de function calling empuja al modelo a actuar; con un contrato JSON
por texto, "opinar" es una salida válida más. Fix: el contrato ahora prohíbe `kind=text` sin
haber ejecutado al menos una herramienta (commit `c70bccb`). Resultado: **0 episodios vacíos en
160 episodios** de la corrida completa. Lección para el adaptador: con tool-calling por contrato,
la obligación de actuar hay que escribirla; no viene gratis del canal.

## opsworld (incidentes): baseline más fuerte, la ganancia se muda a la eficiencia

| Métrica (n=40 por brazo) | Claude baseline | Claude subcortex | Δ | (Gemini: base → sub) |
|---|---|---|---|---|
| score medio | 42.0 | 41.0 | ≈ empate | 46.3 → 56.3 (+10) |
| resolución | 82 % | 72 % | −10 pts | 93 % → 100 % |
| llm_calls / episodio | 6.03 | 4.75 | **−21 %** | 6.72 → 5.03 (−25 %) |
| empeoramientos | 6 | 2 | **−67 %** | 4 → 2 |
| hábitos disparados | 0 | 6 | — | 0 → 9 |
| tercio final: calls / worsens | 6.14 / 3 | 4.36 / 0 | tendencia clara | — |

Dos observaciones honestas:

1. **El efecto en score desaparece, el de eficiencia y seguridad no.** El baseline de Claude ya
   resuelve bien las causas que a Gemini le costaban (p. ej. escala bien `dependency_down` en vez
   de hacer rollbacks dobles), así que el margen de score que subcortex explotaba con Gemini acá
   no existe. Lo que queda es lo estructural: −21 % de llamadas, un tercio de los
   empeoramientos, hábitos que saltean al LLM en el tercio final (5 de los 6 disparos), y el
   tercio final con 0 empeoramientos.
2. **La resolución bajó 10 puntos con subcortex.** Mirando por episodio: los no-resueltos extra
   se concentran en episodios donde el recall episódico ancló al agente en un precedente parecido
   pero no idéntico y cerró antes de tiempo. Con un modelo fuerte, el prior de la memoria compite
   con un juicio en frío que ya era bueno. Es el mismo fenómeno que el paper anticipa al revés:
   cuanto peor el modelo base, más aporta la memoria; cuanto mejor, más fino tiene que ser el
   umbral de recall (`recall_min_overlap`) para no molestar.

## marketworld (momentum): la misma historia que con Gemini, más marcada

| Métrica (n=40 por brazo) | Claude baseline | Claude subcortex | (Gemini run1: base → sub) |
|---|---|---|---|
| score medio | 1.7 | **9.6** | −4.05 → 6.58 (runs 2–5: 3.4–16.2) |
| empeoramientos | 11 | 6 | 13 → 9 |
| llm_calls / episodio | 3.08 | 2.77 | 4.12 → 5.0 |
| hábitos disparados | 0 | **11** (1 des-habituación) | 0 |
| memoria escrita | 0 | 16 episodios | — |

La dirección replica exactamente lo visto con Gemini (subcortex > baseline, menos
empeoramientos), con dos diferencias: el score de subcortex-Claude (9.6) queda dentro del rango
de las 5 réplicas con Gemini (3.4–16.2), y **acá los hábitos sí se dispararon** (11 veces, con
una des-habituación correcta cuando el régimen cambió) — Claude declara confianzas más altas y
sus éxitos repetidos compilan hábito antes. Es la primera corrida de marketworld donde el
putamen trabajó.

## Costos operativos del motor Claude (medidos)

- **Tiempo:** opsworld 109 min y marketworld 66 min (160 episodios ≈ 1.1 min/episodio).
  Con Gemini, un brazo de marketworld salía en ~17 min contra ~33 min acá: **~2× más lento**,
  por el spawn del proceso `claude -p` en cada llamada (~15–25 s de overhead).
- **Tokens:** los `tokens_mean` de CC (179–216k/episodio) **no son comparables** con los de
  Gemini (8–16k): el envelope de Claude Code cuenta el system prompt del CLI y las lecturas de
  caché en cada invocación. El costo monetario marginal fue $0: todo dentro del plan.
- **Fiabilidad:** 0 errores de proveedor, 0 timeouts, 0 episodios vacíos post-fix en 160
  episodios.

## Conclusiones

1. **La librería es agnóstica del motor de verdad**: mismos plugins, cero cambios, dos
   proveedores. El único código específico fue el contrato de salida del adaptador.
2. **El mecanismo replica entre modelos**: en el dominio con margen (market), subcortex gana con
   ambos motores; en el dominio donde el modelo fuerte ya satura (ops), la ganancia se muda de
   score a eficiencia/seguridad — que es la predicción del modelo cerebral: los ganglios basales
   no te hacen más inteligente, te hacen más barato y menos peligroso.
3. **Nueva variable a estudiar**: la interacción memoria × capacidad del modelo (la resolución
   −10 pts). Sugiere hacer `recall_min_overlap` adaptativo a la tasa de acierto del propio modelo.

**Próximo paso propuesto** (de `docs/superpowers/research/2026-09-02-nuevos-dominios.md`):
tau2-bench primero (vetos con semántica de policy real), AIOpsLab segundo, AppWorld tercero.
Con el motor Claude dentro del plan, ninguno depende de recargar crédito de Gemini.
