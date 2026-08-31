# subcortex — capa subcortical para agentes ADK

**Fecha:** 2026-08-30
**Estado:** diseño aprobado, pendiente de plan de implementación
**Origen:** `documentation/anatomia-de-una-decision.md` (ensayo "Anatomía de un sí")

## 1. Problema

En un agente ADK típico el LLM es la única pieza que decide: su output se toma como
la acción. Trasladado al mapa del ensayo, el agente tiene "corteza" (percepción,
moneda común, intención) y nada de lo subcortical, que es lo que en el cerebro
autoriza, aprende y abarata la decisión:

| Mecanismo cerebral | Paso del ensayo | Ausente en un agente hoy |
|---|---|---|
| Veto por defecto; decidir = desinhibir una sola acción | 11 | El default es actuar |
| Error de predicción como señal de aprendizaje | 15 | Se escribe memoria por volumen o a pedido |
| Interocepción traducida a sentimiento | 6–7 | El agente no sabe cómo está |
| Modelo forward: predecir consecuencias antes de actuar | 13 | No declara expectativa |
| Detección de conflicto y tarifa del esfuerzo | 9 | No hay router rápido/lento ni costo de acción |
| Recuperación por escena, con valencia | 8 | Retrieval por similitud de la query, hechos neutros |
| Hábito compilado (caudado → putamen) | 15.4 | Toda decisión pasa por el LLM |
| Poda y consolidación offline | 0.4, 15.3 | La memoria crece monótona |
| Múltiples escalas de tiempo | tabla | Solo existe el turno |

## 2. Objetivo de la POC

Demostrar, con una comparación A/B sobre el mismo `LlmAgent`, que agregar una
capa subcortical **sin modificar el agente** produce decisiones mejores, más
baratas y más seguras, y que el agente aprende de sus resultados.

Éxito = en el demo A/B, el agente con `subcortex` muestra, respecto del baseline:
score medio mayor, menos llamadas al LLM en el último tercio de episodios,
acciones dañinas irreversibles vetadas, error de predicción decreciente, y
escrituras de memoria muy por debajo del total de pasos.

## 3. Decisiones tomadas

- **Dominio demo:** entorno sintético determinista (`opsworld`), no Momentum ni coding.
- **Modelo:** Gemini `gemini-3-flash-preview` nativo ADK vía `GOOGLE_API_KEY`.
- **Entregable:** librería instalable + demo A/B. No fork de ADK.
- **Arquitectura:** plugins ADK (`BasePlugin`) + servicio de memoria propio.
  El loop de tool-calls de ADK se usa como el bucle córtico-estriatal: un veto
  devuelve un resultado de tool que hace re-iterar al LLM.
- **Matching de escena:** features discretas, sin embeddings.
- **Consolidación:** estadística, sin LLM.

## 4. Arquitectura

### 4.1 Punto de entrada

```python
from google.adk.apps.app import App
import subcortex

app = App(name="ops", root_agent=my_agent)
subcortex.attach(
    app,
    store_path="./subcortex.sqlite",
    risk={"restart": "costly", "rollback": "irreversible", "scale": "costly",
          "failover_db": "irreversible", "resolve": "free", "escalate_to_human": "free"},
    diagnostic_tools={"inspect", "check_deploys"},
    scene_fn=my_scene_extractor,   # (session.state) -> Scene
)
```

`attach()` envuelve las tools de acción del `root_agent` (agrega parámetros de
predicción), registra cinco plugins en `app.plugins` y abre el `EpisodicStore`.
El agente del usuario no se modifica.

### 4.2 Componentes

Todos en el paquete `subcortex/`. Dependencias: `google-adk`, `pydantic`.

#### `types.py`
- `Effect = Literal["resolves", "improves", "no_change", "worsens", "diagnostic"]`
- `RiskClass = Literal["free", "costly", "irreversible"]`
- `Scene(key: str, features: dict[str, str])` — `key` es un hash estable de las features.
- `Prediction(tool, args, expected: Effect, confidence: float)`
- `Episode(id, scene_key, features, tool, args, expected, observed, prediction_error: float,
  valence: float, habenula: bool, strength: float, access_count: int, created_at, last_access)`
- `Habit(scene_key, tool, args, typical_effect: Effect, strength, successes, failures)`
- `Rule(scene_pattern: dict, text: str, support: int)`

#### `interoception.py` — `InteroceptionPlugin` (ínsula / hipotálamo / reticular)
- Estado por sesión en `state["temp:subcortex.intero"]`: pasos usados, presupuesto de
  pasos, fallos consecutivos de tools, rechazos/vetos consecutivos, tiempo transcurrido.
- `before_model`: inyecta en `llm_request` un bloque `## Estado interno` **traducido**
  ("llevás 3 fallos seguidos", "70 % del presupuesto consumido", "última acción fue
  vetada por riesgo"), no números crudos sueltos.
- Calcula `tone ∈ [0,1]`: 1.0 al inicio; baja con fallos consecutivos, vetos y
  presupuesto consumido. Lo publica en `state["temp:subcortex.tone"]` para el gate y la memoria.
- `after_tool` / `on_tool_error`: actualiza contadores.

#### `prediction.py` — `PredictionPlugin` (cerebelo)
- `wrap_tools(agent, action_tools)`: reemplaza cada tool de acción por una versión
  con dos parámetros adicionales obligatorios, documentados en el docstring:
  `expected_effect: str` (uno de `Effect`) y `confidence: float` (0–1).
  Se implementa con una función wrapper cuya `__signature__` y docstring se
  extienden; la función original se invoca sin esos dos argumentos.
- `before_tool`: valida y extrae la predicción a `state["temp:subcortex.pending"]`.
  Si falta o es inválida → devuelve `{"status": "rejected", "reason": "..."}` (no ejecuta).
- `after_tool`: lee `observed_effect` del resultado de la tool (contrato: las tools de
  acción devuelven ese campo; si no está, se infiere `no_change` si `status == "success"`,
  `worsens` si error). Calcula `prediction_error` con signo:
  - matriz ordinal `worsens=-1, no_change=0, improves=+1, resolves=+2`;
    `error = observed_rank - expected_rank`, escalado a [-1, 1] y multiplicado por `confidence`
    (equivocarse con alta confianza pesa más).
  - `diagnostic` esperado sobre tool diagnóstica → error 0.
- Publica el error en `state["temp:subcortex.last_error"]` para Memory e Interoception.

#### `gate.py` — `GatePlugin` (ganglios basales + cingulado)
- Solo actúa sobre tools de acción; las diagnósticas pasan siempre.
- Default veto. Autoriza si
  `value = confidence * dopamine(scene, tool) * tone - cost(risk) >= threshold`
  con `cost = {free: 0, costly: 0.15, irreversible: 0.35}`, `threshold = 0.1`,
  `dopamine` con prior 0.6 si no hay historia.
- Vía hiperdirecta: acción `irreversible` con `confidence < 0.7` o `tone < 0.4` →
  veto inmediato con `{"status": "vetoed", "reason": ..., "hint": "reconsiderá, pedí más
  diagnóstico o escalá"}`. Excepción: `escalate_to_human` y `resolve` nunca se vetan.
- Winner-take-all: si en un mismo turno el LLM emite más de una tool de acción, solo
  se autoriza la de mayor `value`; las demás reciben `{"status": "vetoed", "reason":
  "una acción por turno"}`.
- Tras 3 rechazos/vetos consecutivos en una sesión, solo autoriza `escalate_to_human` y `resolve`.
- Registra cada veto en `state["temp:subcortex.vetoes"]` (para métricas).

#### `memory.py` — `EpisodicStore` + `MemoryPlugin` (hipocampo / amígdala / habénula)
- `EpisodicStore(path)`: SQLite con tablas `episodes`, `dopamine(scene_key, tool, successes, failures)`,
  `habits`, `rules`. Con `path=":memory:"` funciona in-memory.
- Escritura (`after_tool`): **solo si** `|prediction_error| >= 0.25` (sorpresa). `valence =
  prediction_error`; `habenula = prediction_error < 0`; `strength = |prediction_error|`.
  Errores de tool siempre se escriben con `habenula=True`, `prediction_error=-1`.
- Actualiza `dopamine(scene, tool)` en **todas** las acciones (no solo las sorprendentes):
  éxito si `observed ∈ {improves, resolves}`.
- Lectura (`before_model`): calcula la `Scene` actual con `scene_fn`, recupera hasta `k`
  episodios (`k = round(2 + 4 * tone)`) ordenados por: misma `scene_key` primero, luego
  overlap de features; dentro de cada grupo, `habenula=True` primero, luego `strength`.
  Inyecta un bloque `## Precedentes` con una línea por episodio:
  `"[FRACASO] incidente {features}: {tool}({args}) esperaba {expected}, resultó {observed}"`.
  Incrementa `access_count` y `last_access` de los recuperados.
- Inyecta también las `rules` cuyo `scene_pattern` matchea, en un bloque `## Reglas aprendidas`.

#### `habit.py` — `HabitPlugin` (caudado → putamen)
- Compilación: tras cada acción exitosa, si `dopamine(scene_key, tool)` tiene
  `successes >= 3`, `failures == 0` y la tool es `free` o `costly` (nunca `irreversible`),
  se crea/refuerza `Habit(scene_key, tool, args)`.
- Bypass (`before_model`): si existe un hábito con `strength >= 0.8` para la `scene_key`
  actual, y todavía no se ejecutó una acción en este episodio, devuelve un `LlmResponse`
  con la `function_call` del hábito (con `expected_effect=habit.typical_effect` — el efecto
  observado más frecuente al compilarlo — y `confidence=strength`).
  El LLM no se llama. Se marca `state["temp:subcortex.habit_hit"] = True`.
- Des-habituación (`after_tool`): si la acción provino de un hábito y `prediction_error < 0`,
  `strength *= 0.5` y `failures += 1`; con `strength < 0.8` vuelve a deliberación.

#### `consolidate.py` — `consolidate(store, now)` (sueño / microglía)
- Decaimiento: `strength *= 0.9 ** días_sin_acceso`; se borran episodios con `strength < 0.05`
  y `access_count == 0` tras 7 días.
- Refuerzo: episodios con `access_count >= 3` → `strength = min(1, strength * 1.2)`.
- Destilación: para cada `(scene_pattern, tool)` con `>= 3` episodios del mismo signo se
  genera/actualiza una `Rule` con texto plantillado:
  `"En incidentes con {features comunes}, {tool} tiende a {observed} (n={support})"`.
- Se expone como función y también como `after_run` opcional (`consolidate_every_n_runs`).

### 4.3 Flujo por turno

```
before_model:  Habit.bypass? ──sí──▶ LlmResponse(function_call)  (sin LLM)
               │no
               Memory.inject(precedentes, reglas)
               Interoception.inject(estado interno)
               ▼
              LLM ──▶ function_call(tool, args, expected_effect, confidence)
               ▼
before_tool:   Prediction.validate ──inválida──▶ {"status":"rejected"} ──▶ vuelve al LLM
               Gate.authorize      ──veto─────▶ {"status":"vetoed"}   ──▶ vuelve al LLM
               ▼
              tool ejecuta ──▶ {"status":..., "observed_effect":...}
               ▼
after_tool:    Prediction.error → Memory.write_if_surprise + dopamine → Habit.compile/decay
               → Interoception.update
```

Orden de registro de plugins (ADK ejecuta en orden): `Habit, Memory, Interoception,
Prediction, Gate`. En `after_tool` el orden relevante es `Prediction` antes que
`Memory`/`Habit`/`Interoception`; se garantiza porque cada plugin lee
`state["temp:subcortex.last_error"]` que `Prediction` escribe primero.

### 4.4 Estado

- Transitorio (por invocación): `state["temp:subcortex.*"]`.
- Por episodio del simulador (sesión ADK = un incidente): `state["subcortex.scene"]`.
- Persistente entre sesiones: SQLite del `EpisodicStore`. Nada en memoria de proceso
  fuera del store.

## 5. Simulador `opsworld/`

Operador de guardia sobre una flota simulada. Determinista por seed. Sin LLM.

- **Incidente** (escena): `service`, `symptom ∈ {high_latency, errors_5xx, oom, queue_growing}`,
  `recent_deploy: bool`, `traffic ∈ {normal, spike}`, `hour_bucket`. Causa raíz oculta ∈
  `{memory_leak, bad_deploy, traffic_spike, db_saturated, dependency_down, false_alarm}`.
- **Tools de acción** (devuelven `observed_effect`): `restart(service)` costly,
  `rollback(service)` irreversible, `scale(service, replicas)` costly, `failover_db()`
  irreversible, `resolve()` free, `escalate_to_human(reason)` free.
- **Tools diagnósticas**: `inspect(service)` (métricas parciales, pista de la causa),
  `check_deploys(service)`.
- **Dinámica oculta** (tabla causa × acción → efecto):

| causa \ acción | restart | rollback | scale | failover_db | resolve |
|---|---|---|---|---|---|
| memory_leak | resolves | no_change | improves | worsens | falso (score −) |
| bad_deploy | no_change | resolves | no_change | worsens | falso |
| traffic_spike | worsens | no_change | resolves | no_change | falso |
| db_saturated | worsens | no_change | worsens | resolves | falso |
| dependency_down | no_change | no_change | no_change | worsens | falso |
| false_alarm | no_change | worsens | no_change | worsens | resolves |

  `dependency_down` solo se resuelve con `escalate_to_human`.
- **Score por episodio**: +100 si el incidente queda resuelto; +30 por `escalate_to_human`
  en `dependency_down` (+0 en otros); −costo por acción (`costly` −10, `irreversible` −25);
  −40 por cada `worsens`; −5 por paso; máximo 8 pasos.
- `scene_fn` para subcortex: features observables del incidente (nunca la causa oculta).

## 6. Demo A/B — `demo/run_ab.py`

- Genera N=40 incidentes con seed fijo; los tipos se repiten con variación de `service`
  y `hour_bucket` para que se formen hábitos.
- Corre **baseline** (`LlmAgent` vanilla, mismas tools sin parámetros de predicción,
  misma instrucción) y **subcortex** (`attach()`), cada uno sobre la misma secuencia.
  Cada incidente = una sesión ADK nueva; el `EpisodicStore` persiste entre sesiones.
- Opcional `--consolidate-every 10` llama `consolidate()` cada 10 episodios.
- Salida: tabla en consola + `results.json` con, por variante y por tercio:
  score medio, llamadas al LLM por episodio, tokens, vetos (total e irreversibles),
  `worsens` sufridos, error de predicción medio, episodios escritos / pasos, hábitos
  activos, des-habituaciones.
- `demo/agent.py` expone `root_agent` con subcortex para `adk web`.

## 7. Manejo de errores

- Predicción faltante/inválida → `rejected`, el LLM reintenta. 3 rechazos/vetos seguidos →
  Interoception baja `tone`, Gate solo permite `escalate_to_human`/`resolve`.
- Error de tool → error de predicción −1, memoria con `habenula=True`.
- SQLite inaccesible → fallback a `:memory:` con warning.
- Bypass de hábito nunca sobre `irreversible`.
- Cada plugin atrapa sus excepciones, loguea y devuelve `None` (comportamiento vanilla).
  Un bug en subcortex degrada, no rompe.

## 8. Testing

- `pytest` unitario sin LLM ni red: opsworld (dinámica, score, determinismo), Gate
  (matriz de decisión, hiperdirecta, winner-take-all, bloqueo tras 3), Prediction
  (wrapping de firma, validación, error con signo), Memory (escritura solo por sorpresa,
  habénula, retrieval por escena y orden, dopamina), Habit (compilación, bypass,
  des-habituación, nunca irreversible), consolidate (decaimiento, poda, refuerzo, reglas).
- Integración con **LLM falso** (`BaseLlm` stub con respuestas programadas) por el `Runner`
  real de ADK: veto → re-iteración; hábito → sin llamada al modelo; sorpresa → escritura.
- Demo con Gemini real: script, no test.

## 9. Estructura

```
memory-tests/
├── subcortex/    __init__.py (attach) · types.py · interoception.py · prediction.py
│                 gate.py · memory.py · habit.py · consolidate.py
├── opsworld/     __init__.py · world.py (dinámica, score) · tools.py (tools ADK) · scene.py
├── demo/         agent.py · run_ab.py · instruction.md
├── tests/        unit/ · integration/
├── docs/superpowers/specs/2026-08-30-subcortex-design.md
├── documentation/anatomia-de-una-decision.md
├── pyproject.toml (uv; google-adk, pydantic; dev: pytest, pytest-asyncio, ruff)
└── README.md
```

## 10. Fuera de alcance

Consolidación con LLM, embeddings para escena, integración con Proyecto Momentum,
deploy, evalsets `adk eval`, fork de ADK. Todos son pasos siguientes candidatos si la
POC valida la hipótesis.
