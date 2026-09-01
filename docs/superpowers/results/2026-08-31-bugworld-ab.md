# Corrida A/B fuera del simulador — bugworld (2026-08-31)

**Setup:** 40 bugs reales inyectados por mutación en `toolz` 1.1.0 (6 clases de código + `broken_test`),
tools reales (`run_tests`, `read_file`, `search`, `edit_file`, `rewrite_file`, `revert_file`, `finish`,
`escalate_to_human`), `gemini-3-flash-preview`, presupuesto de 10 tool calls, mismo agente e
instrucción en ambas variantes. Cada `edit_file` reejecuta la suite completa (180 tests).
La corrida se cortó dos veces (cap mensual de gasto de AI Studio; bucle de hábito inválido,
abajo) y se retomó con `--resume`; 34 filas de baseline se reconstruyeron del log y **no tienen
tokens ni `worsens`**, así que esas dos columnas no son comparables. Datos en `results-bugs.json`.

## Resultado general

| métrica | baseline | subcortex |
|---|---|---|
| score medio | 22.5 | 25.0 |
| tasa de resolución | 0.70 | 0.72 |
| llamadas al LLM / bug | 12.6 | 11.8 |
| pasos / bug | 8.1 | 8.05 |
| vetos | — | 3 |
| episodios escritos en memoria | — | **0** |
| hábitos compilados / disparos | — | 0 / 0 |
| reglas destiladas | — | 0 |
| score por tercios | 21 → 26 → 20 | 18 → 34 → 22 |

**Veredicto: empate, y un empate informativo.** La diferencia (+2.5 de score, +1 bug resuelto,
−6 % de llamadas) está dentro del ruido del modelo. Pero el dato que importa es la fila de
episodios escritos: **cero**. La capa subcortical no aprendió nada en 40 bugs porque nunca
tuvo una sorpresa que registrar. En este dominio el framework, tal como está diseñado, es inerte.

## Por clase

| clase | score base | score sub | resueltos base | resueltos sub |
|---|---|---|---|---|
| arith_swap | 43.3 | 54.2 | 5/6 | 6/6 |
| compare_swap | 16.7 | 2.5 | 4/6 | 3/6 |
| bool_swap | 21.7 | 20.0 | 4/6 | 4/6 |
| negate_condition | 50.8 | 54.2 | 6/6 | 6/6 |
| constant_shift | 35.8 | 52.5 | 5/6 | 6/6 |
| return_none | 28.0 | 30.0 | 4/5 | 4/5 |
| broken_test | −50.0 | −50.0 | 0/5 | 0/5 |

## Por qué la memoria no escribió nada

El único evento evaluable por episodio fue **la única edición válida**, y cuando una edición se
aplica, casi siempre es la correcta: `expected=resolves`, `observed=resolves`, error 0, sin sorpresa.
Todo lo demás que hizo el agente —`old` mal copiado, editar un test, seguir tras el cierre— fueron
llamadas `invalid`, que por contrato (y con razón) no son resultados sobre el mundo. Los bugs no
resueltos terminaron sin ninguna acción evaluada. Resultado: 0 episodios, 0 dopamina útil, 0 reglas, 0 hábitos.

La regla "escribir solo ante sorpresa" viene del paso 15 del ensayo (error de predicción), y en
opsworld funcionó porque las acciones tenían efectos variados y repetibles. En una tarea de un
solo golpe como arreglar un bug, **el evento informativo es el éxito mismo** —qué función, qué
patrón de fix, para qué `finding`— y hoy ese evento no se guarda. El hipocampo también codifica
episodios nuevos que salieron bien; el diseño lo omitió.

## Qué sí pasó

- **Escalación: 0/5 en ambas.** Ni el baseline ni subcortex escalaron nunca un `broken_test`;
  los dos agotaron los 10 pasos intentando editar código. En opsworld el gate arregló el caso
  análogo (`dependency_down`) porque la acción errónea era una *acción* que se podía vetar. Acá la
  conducta errónea es *no actuar* (leer y probar ediciones inválidas), y no hay mecanismo que la
  detecte: la interocepción cuenta pasos pero no empuja a escalar cuando el presupuesto se va sin progreso.
- **3 vetos**, todos de valor bajo (confianza < 0.34 en `edit_file`): irrelevantes.
- **`compare_swap` peor con subcortex (2.5 vs 16.7).** Un bug de diferencia (#36, sin edición
  válida); con n = 6 por clase es ruido.
- **Bug de framework encontrado:** tres `edit_file` exitosos en la misma clase de escena compilaban
  un "hábito" con el `old`/`new` de otro bug; al dispararse daba `invalid`, no se des-habituaba y
  no marcaba el episodio como actuado → bucle infinito sin llamadas al LLM (colgó la corrida 54
  minutos). Corregido: un hábito exige la **misma acción con los mismos args** tres veces, se intenta
  **una vez por episodio** y se **debilita** si produce una llamada inválida o vetada. Con tests.
- **Estado `invalid`** agregado al contrato (prueba de humo): sin él, tres `old` mal copiados hundían
  el `tone` a 0.05 y la dopamina de `edit_file` a 0.2.

## Lo que esto enseña sobre el framework

1. **El disparador de escritura debe incluir el éxito.** Escribir episodios cuando la tarea se
   resuelve (escena → acción que resolvió, con su `finding`), no solo ante sorpresa. Y recuperarlos
   como "precedentes de éxito": "en `numeric_mismatch` sobre `test_itertoolz`, el fix fue un límite en `sliding_window`".
2. **La identidad de la acción es del dominio.** En opsworld la acción es la tool; en código es
   el contenido del edit. Los hábitos necesitan una noción de "misma acción" configurable; hoy la
   plantilla `$feature` es un parche.
3. **Falta el mecanismo de "no progreso".** Interocepción debería producir una señal explícita
   cuando el presupuesto se consume sin ningún resultado evaluado, y el gate debería usarla para
   favorecer `escalate_to_human` (el análogo de "tirar la toalla" que el ensayo no modela).
4. **`invalid` es parte del contrato**, no un detalle: los mundos reales rechazan muchas más
   llamadas de las que ejecutan.

## Iteración 2 (misma noche): escritura por éxito + señal de no-progreso

Se implementaron los puntos 1 y 3 (`write_on_success`, `is_stalled` + aviso "SIN PROGRESO… escalá")
y se corrió de nuevo solo subcortex contra el mismo baseline. Datos en `results-bugs-iter2.json`.

| métrica | baseline | subcortex v1 | subcortex v2 |
|---|---|---|---|
| score medio | 22.5 | 25.0 | 19.0 |
| resueltos | 28/40 | 29/40 | 26/40 |
| llamadas al LLM / bug | 12.6 | 11.8 | 11.8 |
| episodios escritos | — | 0 | **26** (20 escenas distintas) |
| estancados (10 pasos sin acción) | 12 | 11 | 14 |
| escalaciones | 0 | 0 | **0** |
| score por tercios | 21 → 26 → 20 | 18 → 34 → 22 | 6 → 34 → 17 |

| clase | base | v1 | v2 |
|---|---|---|---|
| arith_swap | 43.3 | 54.2 | 55.8 |
| compare_swap | 16.7 | 2.5 | 20.0 |
| bool_swap | 21.7 | 20.0 | 5.0 |
| negate_condition | 50.8 | 54.2 | 40.8 |
| constant_shift | 35.8 | 52.5 | 36.7 |
| return_none | 28.0 | 30.0 | 12.0 |
| broken_test | −50.0 | −50.0 | −50.0 |

**Veredicto: tres corridas, tres empates dentro del ruido (±5).** Los dos mecanismos nuevos
hicieron exactamente lo que se diseñó y ninguno movió el resultado:

- **La memoria escribió 26 episodios de éxito** (`edit_file` con `old`/`new` reales, p. ej.
  `mid = len(seqs) * 2 → // 2` para `RecursionError/exception` en `test_itertoolz`), pero fueron
  **20 escenas distintas en 26 episodios**: casi nada se repite. Un precedente de éxito de otro
  bug, aunque comparta `error_type` y `finding`, describe otra función y otro fix: no ayuda y ocupa
  contexto (el primer tercio de v2 fue el peor de las tres corridas, 6.2, aunque puede ser ruido).
- **La señal de no-progreso no produjo ninguna escalación** en 14 episodios estancados. El aviso
  entra en el estado interno del sistema, pero el modelo, con el problema delante, prefiere seguir
  leyendo y probando a rendirse; una línea en el system prompt no compite con la tarea.
  `broken_test` sigue 0/5. Para mover esto haría falta que el gate lo *imponga* (tras el umbral,
  solo `escalate_to_human`/`finish` autorizados), y eso cambia el score por diseño, no por aprendizaje.

## Conclusión de bugworld

En arreglar bugs de un solo golpe, con este modelo y este presupuesto, la capa subcortical no
tiene palanca: las situaciones no se repiten a la granularidad en la que se decide, la acción no es
reutilizable, y el veto no aplica porque la conducta errónea es no actuar. Lo que la capa aporta acá
es diagnóstico (el `veto_log`, los `invalid`, la métrica de estancamiento) y dos bugs de framework
encontrados, no mejor rendimiento. La hipótesis del ensayo se sostiene donde hay repetición y
acciones con efectos —opsworld— y no se sostiene donde cada episodio es único. Ese límite es el
resultado.

## Siguientes pasos

1. No insistir con bugworld a este nivel de escena. Si se retoma, la escena tendría que ser la
   *función bajo test* + `finding`, y la memoria guardar el *patrón* del fix (operador cambiado),
   no el edit literal: es un experimento distinto.
2. Probar un tercer dominio con repetición real y acciones reutilizables (Proyecto Momentum en
   replay: régimen de mercado como escena, indicadores como `finding`, rotar/mantener como acción).
3. 3 seeds en opsworld para consolidar el número que sí se sostiene.
