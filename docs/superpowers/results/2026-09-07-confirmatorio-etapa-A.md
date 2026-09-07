# Experimento confirmatorio, etapa A: la arquitectura no sobrevive a datos nuevos

**Fecha:** 2026-09-07 · **Config congelada:** `CONFIG_SHA = 6db371dcff30e02f` ·
**Preregistro:** `docs/superpowers/specs/2026-09-06-preregistro-confirmatorio.md` (commiteado antes
de correr el primer episodio) · **Motor:** Claude Sonnet 5 (`claude-code:sonnet`) ·
**Datos crudos:** `results-confirm-A.json`

**Resultado en una línea: la hipótesis central queda refutada.** En cinco semillas que la
arquitectura nunca vio, la capa completa rinde **peor que el agente desnudo** en el endpoint
primario, con 0 de 5 semillas a favor. El efecto positivo reportado en las secciones 6.1–6.5 del
paper era, en buena medida, sobreajuste al banco de desarrollo.

---

## 1. Diseño

Tres brazos sobre la misma secuencia de incidentes, pareados por semilla:

| brazo | qué tiene | para qué |
|---|---|---|
| `vanilla` | nada | baseline |
| `protocol` | solo el contrato de predicción obligatoria | separar "la arquitectura ayuda" de "obligar a anticipar la consecuencia ayuda" |
| `full` | la arquitectura propuesta completa | la hipótesis |

5 semillas held-out (101, 202, 303, 404, 505) × 30 episodios × 3 brazos = **450 episodios**.
Réplica = trayectoria. Orden de brazos barajado por semilla. Endpoint primario declarado de
antemano: `U = score − 2.0 · llm_calls`, comparado pareado con bootstrap de 20 000 remuestreos.

---

## 2. Resultado primario

| brazo | U | score | resolución | daño | llamadas/ep |
|---|---|---|---|---|---|
| `vanilla` | 29.53 | 41.37 | 0.81 | 4.0 | 5.92 |
| `protocol` | **31.79** | 43.20 | 0.81 | 3.6 | 5.71 |
| `full` | **15.61** | 25.97 | **0.55** | **0.6** | **5.18** |

**Comparación pareada (ΔU contra `vanilla`):**

| | ΔU | IC 95 % | semillas a favor | por semilla |
|---|---|---|---|---|
| `full` | **−13.92** | [−20.81, −6.52] | **0/5** | −25.1, −15.7, −14.6, −13.0, −1.1 |
| `protocol` | +2.26 | [−1.57, +4.79] | 4/5 | +2.6, +4.4, −5.3, +4.0, +5.6 |

**H-A (¿la arquitectura aporta sobre el contrato?) — `full` vs `protocol`:**

| métrica | Δ | IC 95 % | a favor |
|---|---|---|---|
| utilidad | **−16.18** | [−22.52, −9.84] | **0/5** |
| score | −17.23 | [−23.60, −10.87] | 0/5 |
| resolución | −0.27 | [−0.33, −0.20] | 0/5 |
| daño | +3.00 (menos daño) | [+2.20, +3.80] | 5/5 |

El criterio preregistrado para sostener H-A era media positiva con ≥4 de 5 semillas a favor.
**Se obtuvo 0 de 5 con el intervalo enteramente por debajo de cero. H-A queda refutada.**

**Sensibilidad a λ** (el orden no cambia en ningún valor probado):

| λ | vanilla | protocol | full |
|---|---|---|---|
| 0 (score puro) | 41.37 | 43.20 | 25.97 |
| 2 (primario) | 29.53 | 31.79 | 15.61 |
| 5 | 11.77 | 14.67 | 0.07 |
| 10 | −17.83 | −13.87 | −25.83 |

---

## 3. Qué hace `full` para perder: escala en vez de actuar

| acción | vanilla | protocol | full |
|---|---|---|---|
| `escalate_to_human` | 29 % | 28 % | **57 %** |
| `restart` | 25 % | 25 % | 10 % |
| `scale` | 15 % | 14 % | **1 %** |
| `rollback` | 18 % | 18 % | 13 % |

Y el desglose por causa muestra el trueque exacto:

| causa | vanilla | protocol | full | Δ (full − vanilla) |
|---|---|---|---|---|
| `db_saturated` | −60.2 | −55.6 | **−6.2** | **+54.0** |
| `false_alarm` | 82.2 | 89.8 | 90.0 | +7.8 |
| `dependency_down` | 19.4 | 19.8 | 18.4 | −1.0 |
| `bad_deploy` | 60.6 | 61.4 | 32.4 | −28.2 |
| `memory_leak` | 67.0 | 71.4 | 25.8 | −41.2 |
| `traffic_spike` | 79.2 | 72.4 | **−4.6** | **−83.8** |

**La única causa que la capa mejora es exactamente la que motivó los cambios de arquitectura
durante el desarrollo.** `db_saturated` era el caso que el análisis de fallas de la corrida 1
señaló, y para el que se introdujeron la clave de escena gruesa y el ajuste del gate. Mejora 54
puntos. Las otras tres causas resolubles se derrumban. Es la definición de sobreajuste, y se ve
a simple vista en la tabla.

---

## 4. Qué mecanismo lo causa

En 150 episodios de `full`: **5 vetos, 1 disparo de hábito, 205 episodios escritos en memoria**.
El gate y los hábitos estuvieron prácticamente inertes. La interocepción tampoco disparó sus
avisos: con 2.3 pasos promedio por episodio, la señal de estancamiento nunca se activó (0 veces) y
el tono se mantuvo alto (~0.86). Como `protocol` ≈ `vanilla`, la resta deja **la recuperación de
memoria como la única variable activa** que distingue `full` de los otros dos brazos.

Un diagnóstico offline (sin LLM, reproduciendo el recall sobre la semilla 101) muestra el
mecanismo: **el 13 % de los precedentes recuperados provienen de una causa distinta** a la del
incidente actual, porque `recall_min_overlap = 2` cuenta features genéricas —`service`,
`hour_bucket`, `traffic`— que se comparten entre causas. El par contaminado más frecuente es
`memory_leak ← db_saturated`: en `db_saturated`, `restart` empeora y se escribe un episodio
marcado como fracaso; ese fracaso se filtra a escenas de `memory_leak`, **donde `restart` es la
acción correcta**, y aparece primero en el contexto por la prioridad de recuperación de fracasos.
El agente lee "restart falló" y escala. Limitar el recall a la misma clase de escena elimina la
contaminación por completo (100 % de precedentes de la causa correcta) en el mismo diagnóstico.

Esto es exactamente lo que anticipó la revisión de literatura neurocientífica
(`docs/superpowers/research/2026-09-06-auditoria-neurociencia.md`): nuestro recall hace **lo
contrario** a la separación de patrones del giro dentado — prioriza el máximo solapamiento de
features en vez de volver más distinguibles las escenas parecidas.

---

## 5. Lo que la capa sí logra, y por qué no alcanza

`full` **reduce el daño de 4.0 a 0.6 acciones dañinas por trayectoria (5/5 semillas a favor)** y
usa **0.74 llamadas menos por episodio (5/5)**. Las dos afirmaciones de eficiencia y seguridad del
paper se sostienen en datos nuevos. El problema es el precio: paga esa prudencia con **26 puntos
de tasa de resolución** (0.81 → 0.55).

En la frontera de Pareto (utilidad ↑, daño ↓), los brazos no dominados son `protocol` y `full`:
`full` sobrevive únicamente por su mínimo daño, y `vanilla` queda **dominado por `protocol`**.
Con el λ declarado antes de correr, el intercambio de `full` no compensa.

---

## 6. Calibración: una mejora aparente que es, en buena parte, un artefacto

| brazo | n | Brier ↓ | NLL ↓ | ECE ↓ | acierto ↑ | confianza media |
|---|---|---|---|---|---|---|
| `protocol` | 372 | 1.087 | 2.044 | 0.513 | 0.296 | 0.804 |
| `full` | 300 | **0.695** | **1.338** | **0.238** | **0.534** | 0.772 |

A primera vista `full` predice mucho mejor las consecuencias. Pero **el 51 % de sus predicciones
son sobre `escalate_to_human`**, la acción más fácil de anticipar (acierto 0.71 contra 0.38 en
`protocol`, que la usa la mitad de veces). Es precisamente el desplazamiento de distribución de
acciones que la auditoría advirtió que podía disfrazarse de mejor modelo del mundo.

Controlando por herramienta queda una mejora **real pero modesta**: `resolve` 0.44 vs 0.25,
`rollback` 0.43 vs 0.39, `restart` 0.27 vs 0.15. O sea: la memoria **sí** mejora algo la capacidad
de anticipar consecuencias — y aun así el agente convierte esa mejor anticipación en peores
decisiones. Anticipar que una acción puede fallar lo lleva a no actuar, en vez de a elegir la
acción correcta.

---

## 7. Qué concluimos y qué no

**Concluimos:**
1. La arquitectura congelada **degrada el desempeño** en dominios nuevos del mismo mundo: −13.9 de
   utilidad, −26 puntos de resolución, 0/5 semillas a favor.
2. El efecto de las corridas de desarrollo era, en buena medida, **sobreajuste**: la única causa
   que mejora es la que guió el rediseño.
3. **El contrato de predicción obligatoria es neutro** en desempeño (ΔU +2.26, intervalo cruzando
   cero) — no era el confound que temíamos, pero tampoco aporta por sí solo.
4. Las afirmaciones de **menos daño y menos llamadas se sostienen** (5/5 ambas). Lo que no se
   sostiene es que salgan gratis.
5. El mecanismo responsable es, por eliminación y por diagnóstico directo, **la recuperación de
   memoria por solapamiento de features**.

**No concluimos:**
- Que la idea de una capa de control externa no sirva. Este experimento refuta **esta**
  arquitectura con **estos** umbrales en **este** mundo.
- Que la memoria episódica no ayude a los agentes. Refuta este mecanismo de recuperación.
- Nada sobre otros modelos: se corrió con Sonnet 5 únicamente.
- Nada biológico.

---

## 8. Qué sigue

1. **Etapas B y C** con la misma arquitectura congelada: `retrieval` (contrato + memoria)
   confirmaría directamente la atribución a la memoria; `cache`, `gate` y `telemetry` completan el
   desglose. Son datos baratos y ya están implementados.
2. **Rediseño v2** con el fix candidato —limitar el recall a la clase de escena, y no priorizar
   fracasos de otra clase—, que **cambia `CONFIG_SHA` y exige una corrida confirmatoria nueva**.
3. **Separación permanente dev/test.** Las semillas 101–505 quedan quemadas: a partir de ahora son
   datos de desarrollo. v2 se valida contra un conjunto de test fresco que no se mira hasta el
   final.
4. **Corregir el paper**: §6.6 con este resultado, y el resumen ejecutivo con el alcance real.

---

## 9. Nota metodológica

Dos observaciones sobre la propia corrida, para que nadie las descubra después:

- **Enmienda E1** (registrada en el preregistro): al arrancar el brazo `protocol` se detectó que
  el conteo de llamadas vivía en el plugin del gate, así que los brazos sin gate reportaban costo
  cero. Se movió al plugin de predicción, presente en todas las configuraciones, y se descartó el
  único episodio afectado. No cambia la arquitectura ni el hash.
- **Los tokens no son estrictamente comparables entre brazos**: `vanilla` los cuenta por eventos
  del runner y los brazos con capa por el plugin. Las medianas son del mismo orden (202k, 186k,
  181k), pero no reportamos diferencias de tokens como hallazgo.
