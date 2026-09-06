# Preregistro del experimento confirmatorio de subcortex

**Fecha:** 2026-09-06 · **Config congelada:** `CONFIG_SHA = 6db371dcff30e02f` ·
**Motivo:** auditoría externa (PhD) del 2026-09-05.

Este documento se escribe y se commitea **antes de correr un solo episodio de evaluación**. Su
función es hacer imposible el ajuste post-hoc: cualquier cambio de arquitectura, de endpoint o de
criterio de éxito posterior a este commit cambia `CONFIG_SHA` y obliga a declarar una corrida nueva.

---

## 1. Por qué existe este documento

La auditoría identificó un problema que invalida las conclusiones fuertes de las corridas previas:
**reuso adaptativo del banco de evaluación**. La primera corrida de `opsworld` favoreció al
baseline (37.1 vs 42.1); el análisis de esas fallas motivó tres cambios de arquitectura; la
corrida siguiente, **sobre los mismos 40 incidentes con la misma semilla**, dio el titular
46.3 → 56.3. Eso es ingeniería iterativa legítima, pero convierte a `opsworld` en un conjunto de
desarrollo. El mismo patrón afecta a `bugworld` (los fallos motivaron `write_on_success` y
detección de estancamiento antes de reevaluar las mismas 40 mutaciones) y a `marketworld` (el
horizonte pasó de 7 a 21 días tras observar que score y equity se contradecían, sobre la misma
serie histórica).

**Consecuencia asumida:** todos los resultados anteriores a este documento se reclasifican como
**desarrollo / prueba de concepto**. No son validación confirmatoria y el paper debe decirlo.

Un segundo problema, igual de serio: la comparación "mismo agente con y sin la capa" **no era
limpia**. Al adjuntar subcortex, el agente pasa a declarar `expected_effect` y `confidence` y
recibe instrucciones de protocolo. Cualquier mejora podía venir de eso —de obligarlo a pensar la
consecuencia antes de actuar— y no de la memoria, el gate o los hábitos. Este experimento aísla eso.

---

## 2. Qué queda congelado

Todos los umbrales de `demo/arms.py::FROZEN` (idénticos a la última iteración de desarrollo):
`gate_threshold=0.10`, `dopamine_prior=0.6`, `hyperdirect_confidence=0.6`, `hyperdirect_tone=0.4`,
`habit_min_successes=3`, `habit_min_strength=0.8`, `surprise_threshold=0.25`,
`recall_min_overlap=2`, `recall_max=4`, `step_budget=8`, costos `{free 0, costly 0.10,
irreversible 0.20}`, `write_on_success=True`, `confidence_from_history=False`,
`cingulate_reconsider=False`.

También quedan congelados: el conjunto de brazos, el endpoint primario, λ, las semillas de
evaluación y la regla estadística. El hash cubre todo eso.

**Prohibido durante la corrida:** tocar umbrales, cambiar `coarse_features`, cambiar el prompt del
agente, cambiar el mundo, agregar semillas después de mirar resultados, o elegir el endpoint
después de ver los datos.

---

## 3. Brazos (el control que faltaba)

| Brazo | Predicción | Memoria | Gate | Hábitos | Estado interno | Qué mide |
|---|---|---|---|---|---|---|
| `vanilla` | — | — | — | — | — | Baseline existente |
| `protocol` | ✔ | — | — | — | — | **Control crítico**: efecto de solo obligar a predecir |
| `retrieval` | ✔ | ✔ | — | — | — | La alternativa simple: recuperación de experiencia |
| `cache` | ✔ | — | — | ✔ | — | Memoización procedural pura |
| `gate` | ✔ | — | ✔ | — | — | Arbitraje/seguridad aislado |
| `telemetry` | ✔ | — | — | — | neutro | Si la "interocepción" era solo información |
| `full` | ✔ | ✔ | ✔ | ✔ | interoceptivo | Arquitectura propuesta |

El brazo `telemetry` recibe **los mismos números** que la interocepción (presupuesto consumido,
errores, bloqueos) en lenguaje de telemetría plana, sin encuadre corporal ni recomendación.

**Etapas** (para que un corte de cuota no invalide nada): **A** = `vanilla`, `protocol`, `full` ·
**B** = `retrieval`, `cache` · **C** = `gate`, `telemetry`. La etapa A por sí sola responde la
crítica central.

---

## 4. Diseño

- **Mundo:** `opsworld`. Es sintético y lo diseñó el mismo autor de la capa: **no alcanza para una
  afirmación de generalización** y así se reportará. Su función acá es responder la crítica del
  confound de protocolo con datos limpios. La validación externa (τ-bench / AppWorld) queda
  explícitamente pendiente y se declara como tal.
- **Semillas de evaluación (held-out):** `101, 202, 303, 404, 505`. Nunca se usaron en desarrollo.
  Las semillas 42 y 7 quedan quemadas y no se reportan como confirmatorias.
- **Episodios por trayectoria:** 30. Elegido para que los hábitos tengan oportunidad real de
  compilar (6 causas × ~5 apariciones; el umbral es 3 éxitos de la misma acción).
- **Réplica:** la **trayectoria** (una semilla completa con su memoria), no el episodio. Dentro de
  una trayectoria los episodios están acoplados en serie por la memoria; tratarlos como
  independientes era el error estadístico principal de las corridas previas.
- **Pareo:** todos los brazos corren la **misma secuencia de incidentes** por semilla. Las
  comparaciones son pareadas por semilla.
- **Orden:** el orden de brazos dentro de cada semilla se baraja con `ORDER_SEED = 20260906`, para
  que ningún brazo cargue sistemáticamente con la degradación del proveedor.
- **Modelo:** `claude-code:sonnet` (Claude Sonnet 5) para todos los brazos. Un único modelo:
  la generalización entre modelos no se afirma en este experimento.

---

## 5. Endpoint primario y análisis

**Primario:** `utility_mean` por trayectoria, con
`U = score − λ_c · llm_calls`, **λ_c = 2.0** (declarado acá, antes de correr).
El score del mundo ya cobra el daño (−40 por empeoramiento) y el paso (−5); λ_c cobra el cómputo
que el score no ve. **No se resta el daño otra vez**: sería doble conteo.

**Análisis primario:** diferencia pareada por semilla contra `vanilla`, con bootstrap percentil
(20 000 remuestreos de semillas) e informe de la consistencia de signo. Con 5 semillas el intervalo
va a ser ancho: eso es la incertidumbre real, no un defecto del reporte.

**Secundarios preespecificados** (se reportan todos, no se elige el que convenga):
tasa de resolución · acciones dañinas (`worsens`) · llamadas al modelo · tokens · segundos ·
vetos desglosados por causa (valor / hiperdirecto / racha) · descartes por arbitraje (contados
aparte de los vetos) · disparos de hábito · des-habituaciones · calibración (Brier, log-loss, ECE,
exactitud) sobre las predicciones declaradas.

**Sensibilidad:** U con λ_c ∈ {0, 2, 5, 10}. Si la conclusión cambia de signo dentro de ese rango,
se reporta como no concluyente.

**Frontera seguridad-utilidad:** se reporta el frente de Pareto (utilidad ↑, daño ↓) en vez de citar
la reducción de daño sola. Un brazo que reduce daño bloqueando acciones necesarias no es un
resultado de seguridad.

**Sin corrección de multiplicidad formal**, pero con endpoint primario único declarado: los
secundarios son descriptivos y así se etiquetan.

---

## 6. Criterios de falsación (declarados antes de ver datos)

**H-A (la arquitectura aporta por encima del protocolo).**
Se sostiene si `full − protocol` en U tiene media positiva y ≥4 de 5 semillas a favor.
**Se refuta si** la diferencia es ≤0 o el intervalo bootstrap incluye ampliamente el cero con signo
inconsistente (≤3 de 5). *Si se refuta, la conclusión honesta es: el efecto observado en desarrollo
venía de obligar al modelo a declarar consecuencia y confianza, no de la memoria/gate/hábitos.*

**H-B (la capa completa supera a las alternativas simples).** *(etapa B)*
Se sostiene si `full` supera a `retrieval` y a `cache` en U con ≥4 de 5 semillas a favor.
**Se refuta si** `retrieval` solo iguala o supera a `full`. *Eso confirmaría la explicación
alternativa del auditor: subcortex es recuperación con pasos extra.*

**H-C (la interocepción es más que telemetría).** *(etapa C)*
**Se refuta si** `telemetry` iguala a la interocepción completa. En ese caso el aporte era la
información de presupuesto, no el encuadre, y hay que decirlo.

**H-D (el gate mejora la frontera seguridad-utilidad).**
**Se refuta si** la reducción de daño del brazo `gate` viene acompañada de pérdida de utilidad tal
que queda **dominado en el frente de Pareto** por `protocol`.

**Regla de parada:** las 5 semillas de cada etapa se corren completas. No se detiene la corrida por
resultados intermedios favorables ni desfavorables. Si la cuota corta, se retoma exactamente donde
quedó; los pares (brazo, semilla) incompletos no entran al análisis.

**Compromiso de publicación:** los resultados se reportan completos, con este preregistro
enlazado, **cualquiera sea el signo**. Un resultado que refute H-A es un hallazgo publicable y
degrada la afirmación central del paper, no se descarta.

---

## 7. Lo que este experimento NO prueba

- **No prueba generalización**: un solo mundo sintético, diseñado por el autor. La validación
  externa (τ-bench, AppWorld, AIOpsLab) sigue pendiente y es la próxima prioridad.
- **No prueba generalidad entre modelos**: un solo modelo (Sonnet 5).
- **No prueba nada biológico**: las analogías neuroanatómicas son inspiración funcional; los
  experimentos que las pondrían a prueba (devaluación de resultado, aprendizaje reverso, replay
  priorizado) están documentados en `docs/superpowers/research/2026-09-06-auditoria-neurociencia.md`
  y no se corren acá.
- **No mide costo en dinero**: la latencia y los tokens del canal CLI no son comparables con una API.
