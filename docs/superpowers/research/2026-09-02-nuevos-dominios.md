# Nuevos dominios candidatos para subcortex (investigación, 2026-09-02)

Investigación con búsqueda web sobre dónde probar la POC a continuación, evaluando cada candidato
contra los seis criterios que dejaron los tres mundos ya corridos:

1. **Repetición** de clases de escena entre episodios (sin repetición no hay aprendizaje).
2. **Acciones reutilizables** (mismos args o plantillables; texto libre no compila hábitos).
3. **Resultado medible y rápido** (observed_effect objetivo en segundos/minutos).
4. **Riesgo asimétrico** (el gate aún no mostró su valor en ningún dominio probado).
5. **Replay determinista** (A/B justo, offline, sin credenciales).
6. **Barato** (pocas llamadas por episodio).

## Ranking

| # | Candidato | Qué es | C1 | C2 | C3 | C4 | C5 | C6 | Integración |
|---|---|---|---|---|---|---|---|---|---|
| 1 | [tau2-bench](https://github.com/sierra-research/tau2-bench) (Sierra) | Customer service con policies por dominio (airline/retail/telecom/mock), tools Python, user simulado por LLM, reward programático | Alto | Alto | Alto | **Alto** | Medio | Medio | Baja (pip) |
| 2 | [AIOpsLab](https://github.com/microsoft/AIOpsLab) (Microsoft) | Microservicios en k8s local (kind), 48 fallas inyectables, telemetría, evaluación detección→localización→mitigación | Alto | Alto | Alto | **Alto** | Medio | Medio | Media (Docker+kind) |
| 3 | [AppWorld](https://github.com/StonyBrookNLP/appworld) | 9 apps simuladas, 457 APIs, 750 tareas, backend determinista, checker que penaliza daño colateral | Alto/Medio | Alto | Alto | Medio | **Alto** | Medio | Baja/media (pip) |
| 4 | [WorkBench](https://github.com/olly-styles/WorkBench) | Sandbox de oficina, 26 tools, 690 tareas por plantillas, métrica nativa de acción dañina | Alto | Alto | Alto | Medio/Alto | Alto | Alto | Baja — pero saturado por modelos frontera (útil con modelo chico) |
| 5 | [BALROG](https://github.com/balrog-ai/BALROG) (Crafter/BabyAI/TextWorld) | Harness unificado de juegos por texto, seeds deterministas | Alto | Alto | Alto | Medio | Alto | Bajo/Medio | Baja/media |
| 6 | [ALFWorld](https://github.com/alfworld/alfworld) | Tareas domésticas en texto, 6 clases repetidas | Alto | Alto | Alto | **Bajo** | Alto | Medio | Baja — casi saturado, nada que vetar |
| 7 | [OR-Gym inventario](https://github.com/r2barati/or-gym-inventory) | Decisión de inventario secuencial (Gym) | Alto | Alto | Alto | Medio | Alto | Alto | Trivial — pero estructuralmente ≈ marketworld: confirma, no informa |
| 8 | [AuctionNet](https://github.com/alimama-tech/AuctionNet) (Alibaba) | Auto-bidding con env generativo calibrado con datos reales | Alto | Alto | Alto | Medio/Alto | Alto | Alto | Media/alta (orientado a RL, dataset 80 GB) |
| 9 | [Factorio LE](https://github.com/JackHopkins/factorio-learning-environment) | Factorio headless para LLMs (NeurIPS 2025) | Alto | Alto | Alto | Medio | Alto | **Bajo** | Media — caro en llamadas |

**Descartados con razón**: Vending-Bench (benchmark oficial cerrado; 60–100M tokens por rollout), ITBench (mismo nicho que AIOpsLab con más fricción; SRE-Agent archivado), AgentBench (modo mantenimiento, poca repetición intra-entorno), NetHack solo (episodios eternos, ~0–2 % incluso para frontera), WebArena/OSWorld (setup pesado, acciones de UI frágiles → hábitos poco compilables).

## Top-3 razonado

1. **tau2-bench** — el único con **C4 alto y esfuerzo bajo**: las policies definen explícitamente qué
   acciones están prohibidas por contexto, así que el veto tiene por primera vez semántica clara y
   medible (violación de policy = fallo objetivo). Su punto flojo (user simulado por LLM) se mitiga
   fijando modelo/seed y promediando 3 corridas. Empezar por `mock`/`retail`.
2. **AIOpsLab** — el dominio donde subcortex mejor funcionó (incidentes), pero real y diseñado por
   terceros: fallas inyectadas en k8s local (sin GPU ni credenciales), acciones genuinamente
   destructivas (delete/scale/patch) que le dan al gate su segunda oportunidad seria. Costo: una
   tarde de setup Docker/kind y promediar corridas por ruido de timing.
3. **AppWorld** — el mejor replay determinista de la lista (offline, pip, checker programático que
   penaliza daño colateral): si subcortex mejora acá, la atribución es indiscutible. Verificar
   primero cuántas familias de tareas repiten escena en el split elegido.

**Menciones**: WorkBench como cuarto barato si se quiere la métrica nativa de "harmful action"
con un modelo chico; OR-Gym solo como test unitario del mecanismo de dopamina.
