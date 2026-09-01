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

## Siguientes pasos

1. Implementar 1 y 3 (escritura por éxito + señal de no-progreso) y repetir bugworld: son los
   dos mecanismos que este dominio necesita y opsworld no exigía.
2. Tokens: medir con baseline real (no reconstruido). Los 71 k/episodio de subcortex incluyen
   lecturas de archivo de 120 líneas en cada llamada; comparar contra baseline real antes de concluir.
3. Recién después, 3 seeds en ambos mundos.
