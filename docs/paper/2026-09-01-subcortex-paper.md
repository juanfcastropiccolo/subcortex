# Subcortex: una capa subcortical para agentes LLM

## Diseño a partir de la anatomía de una decisión y validación en tres mundos

**Juan F. Castro Piccolo** · con asistencia de Claude (Anthropic) · 1 de septiembre de 2026

---

## Resumen

Un agente basado en un modelo de lenguaje decide en un solo disparo: el modelo propone y el
sistema ejecuta. En el cerebro, la corteza propone y todo lo demás —ganglios basales, cerebelo,
hipocampo, amígdala, ínsula, habénula— autoriza, predice, recuerda, compila y aprende. Partimos de
un recorrido neuroanatómico de una decisión simple ("Anatomía de un sí") y lo usamos como mapa
de lo que le falta a un agente donde el LLM es solo la corteza. Implementamos ese mapa como
`subcortex`, una librería de cinco plugins para Google ADK que se adjunta a cualquier `LlmAgent`
con una línea, sin modificar el modelo ni el prompt: predicción obligatoria antes de actuar
(cerebelo), veto por defecto con desinhibición de una sola acción (ganglios basales), memoria
episódica por situación que escribe ante sorpresa o éxito (hipocampo, amígdala, habénula),
hábitos compilados que responden sin llamar al modelo (caudado → putamen), interocepción
traducida a lenguaje (ínsula) y consolidación offline (sueño). Lo evaluamos con el mismo agente
Gemini, con y sin la capa, en tres mundos: un simulador de operaciones (`opsworld`), bugs reales
inyectados en una librería Python (`bugworld`) y un replay histórico del paper trader de momentum
del autor (`marketworld`). En `opsworld` la capa sube el score 22 %, lleva la resolución al
100 % y reduce las llamadas al modelo 34 % en el último tercio, con hábitos compilados. En
`marketworld`, sobre 40 decisiones reales de mercado, cinco trayectorias terminan en 89 ± 20 de
equity contra 50.5 del mismo agente sin la capa (5/5 por encima), y con cadencia semanal supera
también a BTC; la ventaja proviene de un comportamiento estable en régimen bajista. En
`bugworld` la capa no aporta nada en tres corridas, y explicamos por qué: sin repetición de
situaciones ni acciones reutilizables, ningún mecanismo subcortical tiene palanca. Documentamos
ocho lecciones de diseño surgidas de los fracasos, los costos (unos 9 USD de API en total) y los
límites del estudio.

---

## 1. Introducción

Los frameworks de agentes actuales tratan al modelo de lenguaje como el sistema completo de
decisión: perciben mediante herramientas, deliberan en el contexto del modelo y ejecutan lo que
el modelo emite. El aprendizaje entre episodios, cuando existe, suele reducirse a resumir
conversaciones o a recuperar texto por similitud con la pregunta. Nada distingue "la acción no se
ejecutó" de "la acción no funcionó"; nada compila una decisión repetida; nada le dice al modelo
cómo está.

La hipótesis de este trabajo es que esa arquitectura reproduce solo la corteza y omite lo que en
un cerebro hace que una decisión sea segura, barata y mejorable con el uso. Lo que un LLM hace
mejor que cualquier otro artefacto es exactamente lo que hace la corteza orbitofrontal: convertir
argumentos heterogéneos a una moneda común. Lo que no tiene es lo subcortical.

Tomamos como punto de partida un ensayo previo del autor que recorre, paso a paso, una decisión
mínima —salir a correr o no— desde la retina hasta la médula y de vuelta al aprendizaje, con
énfasis en tres conclusiones: no hay un lugar donde se decide sino un bucle que converge; la
emoción no interfiere con la razón sino que la habilita, porque traduce el estado corporal a una
moneda comparable; y decidir es, mecánicamente, dejar de inhibir. Este paper convierte ese mapa
en software y lo somete a experimentos A/B con el mismo modelo en tres dominios.

Contribuciones:

1. Un mapeo explícito de quince pasos neuroanatómicos a mecanismos implementables en un
   agente, con las fórmulas concretas de cada uno (sección 3).
2. `subcortex`, una implementación sobre Google ADK que envuelve cualquier `LlmAgent` sin
   tocarlo, con 82 tests que corren sin red (sección 4).
3. Tres bancos de prueba y un protocolo A/B con referencias sin LLM, replicación y ablaciones
   (sección 5), con resultados positivos en dos de ellos y un resultado nulo explicado (sección 6).
4. Ocho lecciones de diseño que salieron de las corridas que no funcionaron, incluidas dos
   incompatibilidades con la API de Gemini 3 y un bucle infinito de hábitos (sección 7).

---

## 2. Del cerebro al agente

La tabla resume el mapeo. La columna "ausente en un agente hoy" describe el estado de los
frameworks de agentes LLM que conocemos; la columna "subcortex" nombra el componente que lo
implementa.

| Paso del ensayo | Mecanismo cerebral | Ausente en un agente hoy | subcortex |
|---|---|---|---|
| 2, 4 — tálamo, formación reticular | Ganancia según estado de alerta | Todo entra al contexto con igual peso | `tone` modula cuántos precedentes se recuperan |
| 5 — amígdala | Valencia antes de deliberar; recuerdos con etiqueta emocional | Memorias neutras | Episodios con `valence`, `habenula` |
| 6–7 — hipotálamo, ínsula | Estado corporal traducido a sentimiento comparable | El agente no sabe cómo está | `InteroceptionPlugin`: presupuesto, fallos, bloqueos → texto + `tone` |
| 8 — hipocampo, parahipocampal | Recuperación por escena; simulación de futuros | Recuperación por similitud con la pregunta | Recall por features de la situación; la escena se enriquece con lo que el diagnóstico descubre |
| 9 — cingulado anterior | Detección de conflicto; costo del esfuerzo | Sin router rápido/lento; sin costo de acción | `reconsider` ante valor cercano al umbral; costo por clase de riesgo |
| 10 — prefrontal | Moneda común, intención | Es lo que el LLM hace bien | El LLM, sin cambios |
| 11 — ganglios basales | Veto universal; desinhibir una sola acción; vía hiperdirecta | El default es actuar | `GatePlugin`: valor ≥ umbral, freno para irreversibles, una acción por turno |
| 13 — cerebelo | Copia eferente, predicción de consecuencias, error | No declara expectativa | `PredictionPlugin`: `expected_effect`, `confidence` obligatorios; error con signo |
| 15.1–15.2 — habénula, dopamina | Error de predición como señal de aprendizaje; canal negativo separado | Se escribe memoria por volumen | Escritura solo ante sorpresa o éxito; `habenula=True` con prioridad de recuperación; dopamina por (escena, acción) |
| 15.4 — caudado → putamen | Hábito: estímulo-respuesta que saltea la deliberación | Todo pasa por el LLM | `HabitPlugin`: compila tras 3 éxitos de la misma acción; responde sin LLM; se des-habitúa |
| 0.4, 15.3 — microglía, sueño | Poda, refuerzo, consolidación episódico → semántico | La memoria crece monótona | `consolidate()`: decaimiento, poda, refuerzo, reglas destiladas |

Dos decisiones del mapeo merecen comentario. Primera: el bucle de tool-calls que ADK ya tiene
—el modelo propone una llamada, la herramienta responde, el modelo vuelve a mirar— es
estructuralmente el bucle córtico-estriado-tálamo-cortical; por eso un veto no rompe nada:
devuelve un resultado de herramienta y el modelo re-itera. Segunda: el hábito se implementa como
un `LlmResponse` sintético que el plugin devuelve antes de llamar al modelo; el modelo no se
invoca, y esa es la única forma de que "compilar" signifique de verdad abaratar.

---

## 3. Diseño de subcortex

### 3.1 Punto de entrada

```python
app = App(name="ops", root_agent=my_agent)
subcortex.attach(app, risk={"restart": "costly", "rollback": "irreversible", "resolve": "free"},
                 diagnostic_tools={"inspect_service"}, coarse_features=("symptom", "finding"))
```

`attach()` envuelve las herramientas de acción del agente (les agrega dos parámetros
obligatorios), registra cinco plugins en `App.plugins` en un orden fijo y abre un
`EpisodicStore` SQLite. ADK ejecuta los plugins en orden y corta la cadena en el primer valor
no nulo; el orden `[Prediction, Gate, Memory, Habit, Interoception]` garantiza que en
`before_tool` la validación preceda al veto, y en `after_tool` el error de predicción se
calcule antes de que memoria, hábito e interocepción lo consuman.

### 3.2 Escena a dos niveles

Una escena es el contexto observable de la decisión. Tiene dos claves: `key`, hash de todas las
features (la usa el recuerdo episódico) y `coarse_key`, hash de un subconjunto configurado
(`coarse_features`) que usan la dopamina, el gate y los hábitos. La distinción salió de un
fracaso: con una sola clave fina, en 40 episodios ninguna situación se repetía y nada aprendía.
La escena además se enriquece durante el episodio con lo que las herramientas diagnósticas
descubren (`discover_fn` → campo `finding`), de modo que "síntoma latencia alta" y "síntoma
latencia alta + pool de base de datos agotado" son clases distintas.

### 3.3 Predicción y error

Toda herramienta de acción exige `expected_effect ∈ {resolves, improves, no_change, worsens}` y
`confidence ∈ [0, 1]`. Las herramientas devuelven `observed_effect` en la misma escala. Con
rangos `worsens = −1, no_change = 0, improves = 1, resolves = 2`:

```
error = clamp((rank(observed) − rank(expected)) / 2, −1, 1) × confidence
```

Equivocarse con alta confianza pesa más. Un `error < 0` es la señal de la habénula.

### 3.4 Gate

Por defecto todo está vetado. Una acción se autoriza si

```
value = confidence × dopamina(coarse_key, tool) − costo(riesgo) ≥ 0.10
```

con `costo = {free: 0, costly: 0.10, irreversible: 0.20}` y `dopamina = (éxitos + 0.6·2) /
(éxitos + fallos + 2)` (prior 0.6). Una acción irreversible tiene además vía hiperdirecta: se veta
si `confidence < 0.6` o si el `tone` es bajo, salvo que la acción ya haya resuelto esa clase de
escena dos veces sin fallar ("confianza ganada"). Si el modelo emite varias acciones en paralelo,
solo pasa la de mayor valor (winner-take-all). Tras tres bloqueos seguidos, solo se autorizan las
acciones de escape (`escalate`, `resolve`). Cada veto devuelve `reason` y `hint` con las acciones
que sí funcionaron en esa clase de escena.

Como extensión opcional (frente 2), la confianza que pesa el gate puede mezclar la declarada por
el modelo con la dopamina según cuánta historia haya: `w = n/(n+2)`; con cero observaciones
manda el modelo, con seis el historial pesa tres a uno.

### 3.5 Memoria

Un episodio se escribe cuando `|error| ≥ 0.25` (sorpresa), cuando la herramienta falló, o cuando
la acción resolvió (éxito, incorporado tras el segundo mundo). Guarda escena, acción, argumentos,
expectativa, resultado, error, valencia, `habenula` y una `strength`. El recuerdo, antes de cada
llamada al modelo, recupera hasta cuatro episodios con la misma escena exacta o con al menos dos
features en común, fracasos primero, y los inyecta como "Precedentes". La dopamina se actualiza
en toda acción evaluada, con una definición de éxito que incluye cumplir una expectativa de
`no_change`: mantener en un mercado que cae es acertar, no fallar.

### 3.6 Hábitos

Tras tres éxitos **de la misma acción con los mismos argumentos** (plantillados: un valor que
coincide con una feature se guarda como `$feature`) en la misma clase de escena, sin fallos, se
compila un hábito con fuerza 0.8. Al empezar un episodio cuya clase tiene un hábito fuerte, el
plugin devuelve la llamada directamente; el modelo no se invoca. Un hábito se intenta a lo sumo
una vez por episodio y se debilita a la mitad si su resultado es negativo, inválido o vetado.
Nunca se compilan hábitos sobre acciones irreversibles.

### 3.7 Interocepción

Cuenta pasos, fallos reales, bloqueos, llamadas inválidas seguidas y acciones evaluadas. Produce
un escalar `tone ∈ [0.05, 1]` que baja con fallos y con presupuesto consumido —no con vetos, para
no realimentarlos— y un bloque de texto traducido ("llevás 2 fallos seguidos", "sin progreso:
consumiste el 60 % del presupuesto sin ningún resultado evaluado; si no tenés un cambio concreto,
escalá"). Es la ínsula: el estado interno en la misma moneda que la meta.

### 3.8 Consolidación

`consolidate()` decae la fuerza de los episodios no accedidos, refuerza los recuperados a menudo,
poda los débiles y viejos, y destila reglas: para cada (acción, resultado) con al menos tres
episodios, la intersección de sus features es el patrón y se genera un texto ("en incidentes con
`dispersion=wide`, `follow_momentum` tiende a `worsens` (n=5)"). Opcionalmente un LLM escribe
reglas más ricas, pero solo se conservan las que al menos tres episodios respaldan: el modelo no
puede inventar reglas sin evidencia. Una función auxiliar sugiere qué features separan mejor
éxitos de fracasos (ganancia de información), candidatas a `coarse_features`.

### 3.9 Contrato de estados

Un resultado de herramienta lleva `status`. Cuatro estados no tocan el mundo y ningún plugin
aprende de ellos: `rejected` (falta predicción), `vetoed` (gate), `invalid` (argumentos mal
formados, episodio terminado) y `reconsider` (cingulado). La distinción entre `invalid` y `error`
resultó decisiva (sección 7).

---

## 4. Implementación

`subcortex` son ocho módulos Python (~1 100 líneas) sobre `google-adk` 2.8 y `pydantic`. Los
plugins heredan de `BasePlugin` y usan seis hooks: `before_model`, `after_model`,
`before_tool`, `after_tool`, `on_tool_error` y el estado de sesión. Todo el estado transitorio
vive en `session.state["subcortex.*"]`; lo persistente, en SQLite (`episodes`, `dopamine`,
`habits`, `habit_candidates`, `rules`). Cada plugin atrapa sus excepciones y degrada a
comportamiento vanilla. Hay 82 tests unitarios y de integración que corren sin red; los de
integración usan el `Runner` real de ADK con un modelo programado (`ScriptedLlm`) y verifican,
entre otras cosas, que un veto hace re-iterar al modelo, que un hábito ejecuta sin llamarlo y que
el historial que ve el modelo tras un hábito no contiene llamadas sintéticas.

Dos mundos sintéticos y uno real acompañan a la librería (~1 900 líneas): un simulador de
incidentes, un inyector de bugs por mutación AST sobre una librería vendorizada, y un replay de
mercado sobre velas diarias reales. Un runner A/B genérico (`demo/ab.py`) ejecuta el mismo
agente con y sin la capa sobre la misma secuencia, con reintentos ante errores transitorios,
tope de llamadas y timeout por episodio, guardado incremental y `--resume`.

---

## 5. Metodología

**Diseño A/B.** En cada mundo se construye un único `LlmAgent` (Gemini 3 Flash, instrucción
fija, herramientas reales del mundo) y se corre dos veces sobre la misma secuencia de episodios:
*baseline* (ADK vanilla) y *subcortex* (`attach()`). La única diferencia entre variantes es la
línea de `attach()`. Cada episodio es una sesión ADK nueva; el `EpisodicStore` persiste entre
episodios.

**Métricas.** Score del mundo por episodio, tasa de resolución, llamadas al LLM, tokens, pasos,
acciones dañinas (`worsens`), vetos, episodios escritos, disparos de hábito, error de predicción
medio. Todas se reportan por tercios de la secuencia para ver aprendizaje.

**Referencias sin LLM.** En `marketworld`, además, la regla momentum pura a dos cadencias y BTC
buy & hold sobre la misma ventana.

**Replicación y ablaciones.** En `marketworld` se corrieron cinco trayectorias de subcortex (el
baseline resultó determinista) y ablaciones desactivando un plugin por vez.

**Mundos.**

*opsworld* — operador de guardia sobre una flota simulada. Seis causas ocultas (memory leak,
deploy roto, pico de tráfico, base saturada, dependencia caída, falsa alarma), síntomas
observables, tres herramientas diagnósticas y seis de acción con riesgo declarado. Dinámica con
trampas: `restart` arregla el leak pero empeora la base saturada; `rollback` es inútil y caro en
un pico de tráfico; `failover_db` en una falsa alarma causa daño. Score +100 por resolver,
−40 por empeorar, −5 por paso, costos por acción. 40 incidentes, seed fijo.

*bugworld* — 40 bugs reales inyectados por mutación AST (operadores aritméticos y de comparación
invertidos, `and`↔`or`, condiciones negadas, off-by-one, `return None`, y bugs en el propio test
donde lo correcto es escalar) en `toolz` 1.1.0 (181 tests, ~3 s). Herramientas reales: correr
pytest, leer, buscar, editar, reescribir, revertir, cerrar, escalar. Cada edición reejecuta la
suite completa. Presupuesto 10 llamadas.

*marketworld* — replay del paper trader de momentum del autor: 10 majors, velas diarias reales de
Binance, regla exacta (top-2 por retorno 30d entre los que cotizan sobre su SMA-100, 0.15 % por
lado). 40 decisiones cada 21 días entre marzo de 2024 y junio de 2026 (fase alcista y bajista),
consecuencia medida a 21 días, cartera persistente; variante semanal con 120 decisiones. El agente
no ve fechas. Escena: tendencia de BTC, amplitud, volatilidad, dispersión, cartera actual;
`finding` = régimen. Acciones: `follow_momentum`, `hold`, `go_cash`, `rotate`.

---

## 6. Resultados

### 6.1 opsworld: la capa funciona, en la segunda iteración

| métrica | baseline | subcortex v1 | **subcortex v2** |
|---|---|---|---|
| score medio | 46.3 | 37.1 | **56.3** (+22 %) |
| tasa de resolución | 0.93 | 0.80 | **1.00** |
| llamadas al LLM / episodio | 6.7 | 7.4 | **5.0** |
| llamadas al LLM, último tercio | 7.2 | 9.1 | **4.8** (−34 %) |
| tokens / episodio | 11 574 | 24 814 | 12 066 |
| acciones dañinas | 4 | 3 | **2** |
| vetos | — | 32 | 0 |
| hábitos compilados / disparos | — | 0 / 0 | **4 / 9** |
| error de predicción por tercio | — | 0.62 → 0.57 | 0.55 → 0.46 |

La primera iteración fue peor que el baseline. El diagnóstico —clave de escena demasiado fina,
bucle veto → tono → veto, y un valor que multiplicaba el tono y hacía imposible autorizar una
acción irreversible a mitad de episodio— produjo la iteración 2, que valida la hipótesis: mismo
score en el primer tercio (no hay memoria todavía), 34 % menos llamadas en el último (cuatro
decisiones ya compiladas en hábitos), la mitad de daño, tokens a la par. Por causa, la ganancia
se concentra donde el baseline se equivoca repetidamente: `db_saturated` (−16 → +37) y
`dependency_down` (−45 → +2.5). Un bug de Gemini 3 (rechaza function calls sin
`thought_signature`) apareció en el primer disparo de hábito y se resolvió reescribiendo el par
llamada sintética → resultado como texto en el historial, de forma independiente del proveedor.

### 6.2 bugworld: empate tres veces, explicado

| | baseline | subcortex v1 | subcortex v2 (+éxito, +no-progreso) |
|---|---|---|---|
| score medio | 22.5 | 25.0 | 19.0 |
| resueltos | 28/40 | 29/40 | 26/40 |
| llamadas al LLM / bug | 12.6 | 11.8 | 11.8 |
| episodios escritos | — | 0 | 26 (20 escenas distintas) |
| escalaciones en `broken_test` | 0/5 | 0/5 | 0/5 |

Las tres corridas caen dentro del ruido del modelo (±5). La fila decisiva es "episodios
escritos": en la primera versión, cero. La única acción evaluable por episodio era la edición
válida, y cuando una edición se aplica casi siempre es la correcta —esperaba `resolves`, obtuvo
`resolves`, error cero, sin sorpresa—; todo lo demás eran llamadas inválidas que por contrato no
son resultados. Agregar la escritura por éxito produjo 26 episodios en 20 escenas distintas: nada
se repite a la granularidad en que se decide, y un precedente de otro bug con el mismo síntoma
describe otra función y otro fix. La señal de no-progreso llegó al modelo y no cambió su
conducta: con el problema delante, prefiere seguir probando a rendirse. Conclusión: donde cada
episodio es único, la acción no es reutilizable y la conducta errónea es *no actuar*, ningún
mecanismo subcortical tiene palanca. Es un límite del enfoque, no un ajuste pendiente.

### 6.3 marketworld: validación en el problema real

**Referencias sin LLM (misma ventana):** regla momentum diaria 78.6; regla al ritmo del agente
(cada 21 días) 50.5; BTC 95.3. Ventana dura para momentum.

**Cinco trayectorias de subcortex, baseline determinista:**

| trayectoria | equity final | régimen bajista: media por decisión | `hold` / `follow` en bajista |
|---|---|---|---|
| run 1 | 90.9 | −1.22 % | 12 / 4 |
| run 2 | 121.0 | −1.29 % | 13 / 2 |
| run 3 | 83.2 | −1.21 % | 13 / 3 |
| run 4 | 68.3 | −1.29 % | 12 / 4 |
| run 5 | 83.2 | −1.21 % | 13 / 3 |
| **subcortex, media ± desvío** | **89.3 ± 19.5** | **−1.24 %** | |
| baseline (40/40 decisiones idénticas en dos corridas) | 50.5 | −3.51 % | 9 / 5 |

Cinco de cinco por encima del baseline; cuatro por encima de la regla diaria; dos por encima de
BTC. El baseline es exactamente la regla a 21 días: siguió `follow_momentum` 26 veces y sus
desvíos no cambiaron nada. La conducta de subcortex en régimen bajista es casi idéntica en las
cinco trayectorias (12–13 `hold` de 19, −1.2 a −1.3 % por decisión): ese es el mecanismo, y es
estable; la dispersión del equity (68–121) proviene de las decisiones en régimen alcista, donde
la varianza del modelo domina. La capa destiló reglas correctas sin etiquetas —"con BTC en
tendencia, `follow_momentum` tiende a `resolves` (n=5)", "con dispersión amplia, tiende a
`worsens` (n=5)", "con BTC bajista, `go_cash` tiende a `no_change` (n=4)"— y en varias
trayectorias compiló un hábito `hold` para el régimen bajista dominante; en una, un hábito de
`follow_momentum` falló una vez y se des-habituó.

**Variante semanal (120 decisiones, horizonte 7):**

| corredor | equity final | régimen bajista (58 decisiones) |
|---|---|---|
| baseline | 91.4 | −0.62 % |
| **subcortex** | **101.6** | **+0.16 %** |
| regla al ritmo del agente | 90.9 | |
| regla diaria | 78.6 | |
| BTC | 95.3 | |

Con tres veces más muestras, los hábitos se dispararon 11 veces (2 compilados, 6 reglas) y
subcortex quedó por encima de BTC. Una sola trayectoria.

**Ablaciones (baseline reutilizado; una trayectoria por variante, mismas 40 decisiones):**

| variante | equity final | régimen bajista: media por decisión |
|---|---|---|
| capa completa (5 trayectorias) | 89.3 ± 19.5 | −1.24 % |
| sin memoria | 69.2 | **−3.50 %** (= baseline) |
| sin interocepción | 69.7 | −2.27 % |
| sin hábitos | 74.9 | −1.22 % |
| sin gate | 91.5 | −1.28 % |
| baseline | 50.5 | −3.51 % |

Con la cautela de que cada ablación es una sola trayectoria contra una media con σ ≈ 20, la señal
conductual es nítida: **sin memoria, la conducta en régimen bajista vuelve exactamente a la del
baseline** (−3.50 % vs −3.51 %); la interocepción aporta prudencia adicional; los hábitos algo; el
gate, nada en este dominio —consistente con los cero vetos de todas las corridas: no hay acciones
irreversibles que frenar. Las variantes de confianza por historial y reconsider quedaron dentro
del rango de la capa completa (92.3 y 69.8; no concluyente con una trayectoria), y la corrida de
reglas con LLM quedó incompleta por presupuesto (14/40).

### 6.4 Costos

Toda la investigación —tres mundos, unas 25 corridas A/B, ~20 millones de tokens registrados—
costó alrededor de 9 USD de API a 0.46 USD por millón de tokens. Una corrida de 40 decisiones de
una variante en `marketworld` cuesta 0.20 USD; un A/B completo en `opsworld`, 0.45; en
`bugworld`, 1.5 (cada edición reejecuta 181 tests y las lecturas de archivo pesan). La capa no
encarece por episodio: en `opsworld` y `marketworld` los tokens quedan a la par y las llamadas
bajan.

---

## 7. Lecciones de diseño

Cada una salió de una corrida que no funcionó y quedó en el código con su test.

1. **La escena necesita dos niveles.** Con una sola clave fina nada se repite y nada aprende;
   con una gruesa el recuerdo pierde precisión. Dopamina, gate y hábitos usan la gruesa; el
   recuerdo, la fina.
2. **La escena se enriquece con lo que el diagnóstico descubre.** El precedente correcto es el que
   comparte lo que se supo *después* de mirar, no solo lo que se vio al entrar.
3. **El tono no multiplica el valor ni baja con los vetos.** Multiplicarlo hacía imposible autorizar
   una irreversible a mitad de episodio; bajarlo con vetos producía un bucle que forzaba
   escalaciones innecesarias.
4. **Cumplir una expectativa de "no cambio" es éxito.** Con recompensa absoluta, `hold` en régimen
   bajista tenía 0 éxitos y 8 fallos; con error de predicción, acierta. Es el paso 15 del ensayo,
   no el paso 5.
5. **Una llamada inválida no es un resultado.** Sin el estado `invalid`, tres argumentos mal
   copiados hundían el tono a 0.05 y la dopamina de `edit_file` a 0.2, y el gate vetaba ediciones
   con confianza 0.8.
6. **Un hábito es la misma acción con los mismos argumentos**, se intenta una vez por episodio y
   se debilita si falla. Sin eso se compilaban ediciones de otro bug y el agente entraba en un
   bucle infinito sin llamar al modelo (54 minutos colgado).
7. **El horizonte de consecuencia debe cubrir hasta la decisión siguiente.** Medir a 7 días con
   decisiones a 21 hacía que el score y el equity se contradijeran.
8. **Las function calls sintéticas se reescriben como texto.** Gemini 3 rechaza en el historial
   llamadas que el modelo no generó; no hay firma dummy documentada. Marcar la llamada del hábito y
   convertir el par llamada → resultado a texto antes de cada invocación es independiente del
   proveedor y cubierto por tests.

---

## 8. Límites y amenazas a la validez

- **Tamaño de muestra.** 40 episodios por mundo (120 en la variante semanal). Las direcciones son
  robustas (replicación 5/5 en `marketworld`); las magnitudes tienen desvíos del orden de la
  mitad de la ventaja.
- **Un solo modelo.** Todo se corrió con Gemini 3 Flash. La comparación con un segundo modelo se
  planificó y se descartó por presupuesto; la afirmación es sobre la arquitectura con este modelo.
- **No determinismo dependiente del camino.** El baseline resultó determinista en `marketworld`;
  subcortex no, porque una decisión distinta cambia qué episodios existen después. La varianza
  observada es una propiedad del sistema, no solo ruido de muestreo.
- **Mundos.** `opsworld` es sintético y sus trampas las diseñó quien diseñó la capa. `bugworld`
  usa un solo repositorio. `marketworld` usa una ventana elegida por su mezcla de regímenes, no
  por ser favorable a la capa (la regla pierde en ella), y las 34 primeras filas de baseline de
  `bugworld` se reconstruyeron del log sin tokens.
- **Calibración de la confianza.** El gate pesa la confianza que el modelo declara, y los modelos
  están mal calibrados. La mezcla con el historial se implementó pero su medición estaba en curso
  al cierre.
- **Las herramientas devuelven `observed_effect`.** En un despliegue real ese campo debe
  producirlo el mundo (tests, métricas, retorno), no el modelo; los tres mundos lo hacen, pero es
  una exigencia de integración que no siempre se cumple.

---

## 9. Trabajo futuro

Tres a cinco trayectorias por variante en todos los mundos y un segundo modelo, para reportar
medias con desvío. Un cuarto mundo con acciones irreversibles reales (operaciones sobre una
instancia de automatización, en sandbox) donde el veto por defecto pueda mostrar su valor, que en
los tres mundos fue marginal. Aprendizaje de la escena: `coarse_features` se eligió a mano por
mundo; la sugerencia por ganancia de información existe, pero aplicarla sin invalidar dopamina y
hábitos requiere una migración de claves. Y el paso natural del proyecto de origen: correr la
capa al lado del paper trader real como segunda opinión, sin tocar la estrategia, empezando por el
hábito `hold` en régimen bajista y las reglas de dispersión que la regla pura no tiene.

---

## 10. Reproducibilidad

Repositorio `memory-tests`, rama `worktree-subcortex-poc`. `uv sync` instala todo; `uv run
pytest` corre los 87 tests (5 lentos en `bugworld`) sin red. `demo/run_ab.py`,
`demo/run_bugs_ab.py` y `demo/run_market_ab.py` corren los A/B con `GOOGLE_API_KEY` en
`demo/.env`; aceptan `--baseline-from`, `--resume`, `--ablate`, `--spacing/--horizon`,
`--history-confidence`, `--reconsider`, `--llm-rules`. Los datos de `marketworld` son 10 CSV de
velas diarias incluidos en el repositorio. Los resultados crudos de cada corrida están en
`results-*.json`; los reportes con diagnóstico, en `docs/superpowers/results/`; las
especificaciones, en `docs/superpowers/specs/`; el ensayo de origen, en
`documentation/anatomia-de-una-decision.md`.

---

## Referencias

- Castro Piccolo, J. F. (2026). *Anatomía de un "sí": recorrido completo de una decisión simple, de la primera sinapsis a la última.* Ensayo, documento del repositorio.
- Damasio, A. R. (1994). *Descartes' Error: Emotion, Reason, and the Human Brain.* Putnam. (Hipótesis del marcador somático.)
- Schultz, W., Dayan, P., & Montague, P. R. (1997). A neural substrate of prediction and reward. *Science*, 275(5306).
- Redgrave, P., Prescott, T. J., & Gurney, K. (1999). The basal ganglia: a vertebrate solution to the selection problem? *Neuroscience*, 89(4).
- Ratcliff, R. (1978). A theory of memory retrieval. *Psychological Review*, 85(2). (Modelos de difusión-deriva.)
- Anderson, J. R., et al. (2004). An integrated theory of the mind. *Psychological Review*, 111(4). (ACT-R.)
- Laird, J. E. (2012). *The Soar Cognitive Architecture.* MIT Press.
- Shinn, N., et al. (2023). Reflexion: language agents with verbal reinforcement learning. *NeurIPS*.
- Wang, G., et al. (2023). Voyager: an open-ended embodied agent with large language models. *arXiv*.
- Park, J. S., et al. (2023). Generative agents: interactive simulacra of human behavior. *UIST*.
- Google. *Agent Development Kit (ADK) — Python*, versión 2.8. Documentación en línea.
