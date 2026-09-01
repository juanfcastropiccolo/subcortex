# subcortex

Capa subcortical para agentes [ADK](https://google.github.io/adk-docs/). Se engancha a cualquier
`LlmAgent` sin modificarlo y le agrega lo que un LLM solo no tiene: veto por defecto, predicción
antes de actuar, interocepción, memoria episódica que escribe solo ante sorpresa, hábitos que
saltean al modelo y consolidación offline.

El mapa viene del ensayo [`documentation/anatomia-de-una-decision.md`](documentation/anatomia-de-una-decision.md):
un LLM es la corteza; esto es el resto.

| Componente | Cerebro | Hook ADK | Qué hace |
|---|---|---|---|
| `InteroceptionPlugin` | ínsula / hipotálamo / reticular | `before_model`, `after_tool` | Traduce el estado interno (presupuesto, fallos, vetos) a texto y a un escalar `tone` que modula al resto |
| `PredictionPlugin` | cerebelo | wrapping de tools, `before_tool`, `after_tool` | Exige `expected_effect` y `confidence` en cada acción; calcula el error de predicción |
| `GatePlugin` | ganglios basales + cingulado | `before_tool`, `after_model` | Veto por defecto; vía hiperdirecta para irreversibles; una sola acción por turno |
| `MemoryPlugin` | hipocampo / amígdala / habénula | `before_model`, `after_tool` | Recuerda por escena, fracasos primero; escribe solo si hubo sorpresa |
| `HabitPlugin` | caudado → putamen | `before_model`, `after_tool` | Compila decisiones repetidas y responde sin llamar al LLM; se des-habitúa si falla |
| `consolidate()` | sueño / microglía | offline | Poda, refuerza y destila episodios en reglas |

## Uso

```python
from google.adk.apps.app import App
import subcortex

app = App(name="ops", root_agent=my_agent)
subcortex.attach(
    app,
    store_path="./subcortex.sqlite",
    risk={"restart": "costly", "rollback": "irreversible", "resolve": "free"},
    diagnostic_tools={"inspect_service"},
)
```

`attach()` envuelve las tools de acción (les agrega los parámetros de predicción) y registra los
plugins en `app.plugins`. La escena se lee de `session.state["subcortex.features"]` por defecto;
pasá `scene_fn=` para cambiarlo.

## Instalación y tests

```bash
uv sync
uv run pytest            # unitarios + integración con un LLM falso; no usa red
```

## Demo A/B

`opsworld/` es un simulador determinista de incidentes con dinámica oculta (p. ej. `restart` arregla
un memory leak pero empeora una DB saturada). `demo/run_ab.py` corre el mismo agente Gemini con y
sin subcortex sobre la misma secuencia y compara score, llamadas al LLM, vetos, daño, error de
predicción, escrituras de memoria y hábitos.

```bash
cp .env.example demo/.env   # y poné tu GOOGLE_API_KEY
uv run python -m demo.run_ab --n 40 --seed 42 --consolidate-every 10
adk web .                   # inspección manual del agente `demo`
```

## Resultados (iteración 2, 40 incidentes, gemini-3-flash-preview)

| métrica | baseline | subcortex |
|---|---|---|
| score medio | 46.3 | **56.3** |
| tasa de resolución | 0.93 | **1.00** |
| llamadas al LLM / episodio (último tercio) | 7.2 | **4.8** |
| tokens / episodio | 11.6 k | 12.1 k |
| acciones dañinas | 4 | **2** |
| hábitos compilados / disparos | — | 4 / 9 |

Detalle, diagnóstico de la corrida 1 (que no funcionó) y lo que queda pendiente en
[`docs/superpowers/results/`](docs/superpowers/results/).

## Fuera del simulador: bugworld

`bugworld/` inyecta 40 bugs reales por mutación AST en `toolz` (vendorizado) y el agente los
arregla con tools reales (pytest, leer, buscar, editar). `demo/run_bugs_ab.py` corre el A/B.
Resultado: **empate en tres corridas** (baseline 22.5 · subcortex 25.0 · subcortex con escritura
por éxito y señal de no-progreso 19.0, todo dentro de ±5 de ruido). En una tarea de un solo golpe
las escenas no se repiten (20 distintas en 26 episodios), la acción no es reutilizable y la conducta
errónea es *no actuar*, que el veto no alcanza. El reporte documenta el límite y por qué.
