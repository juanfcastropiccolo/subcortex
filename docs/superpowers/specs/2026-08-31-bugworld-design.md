# bugworld — A/B de subcortex fuera del simulador

**Fecha:** 2026-08-31 · **Estado:** aprobado · **Depende de:** spec de subcortex (§10, iteración 2)

## 1. Objetivo

Probar la capa subcortical en una tarea real de agente: arreglar bugs en un repo Python con tools
reales (leer, buscar, editar, correr pytest), comparando el mismo `LlmAgent` con y sin `attach()`.
El caso cumple los cuatro requisitos: situaciones que se repiten (clases de bug), resultado medible
por decisión en segundos (la suite), acciones con riesgo asimétrico (`rewrite_file`), y replay
idéntico (misma lista de bugs, workdir limpio por episodio).

## 2. El mundo

- **Repo objetivo:** `toolz` 1.1.0 (BSD) vendorizado en `bugworld/vendor/toolz` con su suite
  (181 tests, ~3 s). Módulos mutables: `itertoolz.py`, `dicttoolz.py`, `functoolz.py`, `recipes.py`.
- **Mutador AST propio** (`bugworld/mutate.py`), 7 clases (la "causa" oculta):

| clase | transformación | dónde |
|---|---|---|
| `arith_swap` | `+`↔`-`, `*`↔`//` | BinOp en src |
| `compare_swap` | `<`↔`<=`, `>`↔`>=`, `==`↔`!=` | Compare en src |
| `bool_swap` | `and`↔`or` | BoolOp en src |
| `negate_condition` | `if x` → `if not x` | If en src |
| `constant_shift` | entero `n` → `n+1` (n ≥ 0, fuera de asignaciones de default) | Constant en src |
| `return_none` | `return expr` → `return None` | Return en src |
| `broken_test` | cualquiera de las anteriores, pero en `tests/test_*.py` | el test está mal; la acción correcta es escalar |

- `enumerate_sites(module)` lista sitios deterministas (orden de aparición). `generate_bugs(n, seed)`
  cicla las clases, elige un sitio al azar (seed) por clase, aplica la mutación en un workdir temporal,
  corre la suite y **conserva solo** mutaciones con 1–8 tests fallando y sin error de importación
  (para `broken_test`, 1–3). Persiste en `bugworld/bugs.json` (`{id, kind, file, site, description,
  failing_tests}`), que ambas variantes leen. `description` es solo para el reporte; nunca se muestra al agente.
- **`BugWorld(bug)`**: copia el template a `.bugworld/run/<id>/`, aplica la mutación, corre la suite
  → `failing0`. Estado: `steps`, `edits`, `score`, `worsens`, `resolved`, `done`, `log`.

## 3. Tools

Diagnósticas (gratis, no pasan por el gate):
- `run_tests(pattern)` → `{status, passed, failed, first_failure (≤ 40 líneas), finding}`;
  `finding` = `classify_failure(output)` ∈ {`numeric_mismatch`, `boundary`, `bool_flip`,
  `none_result`, `exception`, `order`, `unknown`}. Heurístico sobre la salida de pytest, no sobre la causa.
- `read_file(path, start, end)` → líneas numeradas (máx. 120).
- `search(pattern)` → hasta 20 coincidencias `path:línea: texto` en src y tests.

Acciones (pasan por el gate; devuelven `observed_effect`):
- `edit_file(path, old, new)` costosa. `old` debe aparecer exactamente una vez. Tras aplicar, el
  mundo corre la suite completa: `resolves` (0 fallos), `improves` (menos fallos que antes),
  `no_change`, `worsens` (más fallos, o SyntaxError/ImportError).
- `rewrite_file(path, content)` irreversible. Misma evaluación.
- `revert_file(path)` costosa. Restaura el archivo del template.
- `finish()` libre. `resolves` si 0 fallos; si no, `worsens` (−30) y cierra.
- `escalate_to_human(reason)` libre. `resolves` (+30) si la clase es `broken_test`; si no, `no_change` (0). Cierra.
- Editar archivos bajo `tests/` está prohibido por instrucción; si el agente lo hace igual, la tool
  responde `{status: error}` sin aplicar (el mundo lo cuenta como paso).

**Score:** +100 si la suite queda verde; −5 por paso (toda tool); −10 por edición aplicada;
−25 por `rewrite_file`; −40 por cada `worsens`; −30 por `finish()` con fallos; máx. 10 pasos.

## 4. Escena

- Entrada (`state["subcortex.features"]`): `module` (archivo de tests que falla, p. ej. `test_itertoolz`),
  `n_failing` en buckets (`1`, `2-3`, `4+`), `error_type` (primera excepción: `AssertionError`,
  `TypeError`, …). Nunca la clase del bug.
- Descubierto: `finding` de `run_tests`.
- `coarse_features = ("error_type", "finding")`.
- Riesgo: `edit_file` costly, `rewrite_file` irreversible, `revert_file` costly, `finish`/`escalate_to_human` free.
  `always_allowed = {"finish", "escalate_to_human"}`.

## 5. Demo y métricas

`demo/run_bugs_ab.py` con la misma estructura que `run_ab.py` (se extrae lo común a `demo/ab.py`:
`run_variant` genérico sobre un `World`, `summarize`, `print_table`, reintentos, `--baseline-from`).
Métricas idénticas a opsworld más `edits` y `rewrites`. n = 40, seed = 42, `consolidate` cada 10.

## 6. Expectativa honesta

Los hábitos casi no aplican: `edit_file(old, new)` es único por bug y los hábitos compilan
acciones con args. Lo que puede rendir es la memoria por escena y las reglas ("en
`numeric_mismatch` el fix suele ser un límite"), el gate sobre `rewrite_file` e insistencia, y la
interocepción con el presupuesto. Si hay ventaja, será por mecanismos distintos que en opsworld.

## 7. Tests

`mutate` (cada clase produce código válido y distinto; determinismo por seed; `generate_bugs` filtra),
`classify` (salidas de pytest representativas → finding), `world` (score y `observed_effect` por
transición, prohibición de editar tests, `finish`/`escalate`), `tools` (ruteo por sesión). Sin LLM.
