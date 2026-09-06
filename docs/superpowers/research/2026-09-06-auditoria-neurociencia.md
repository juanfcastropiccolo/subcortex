# Auditoría neurocientífica de las analogías de subcortex (2026-09-06)

Revisión componente por componente con literatura verificada, motivada por la auditoría PhD que
calificó los mapeos como "metáforas funcionales, no modelos". El objetivo es separar tres niveles:
**inspiración funcional** (defendible), **analogía algorítmica** (a veces defendible) y **modelo
mecanicista** (no defendible con la evidencia actual), y dejar por escrito qué afirmación sobrevive.

Escala de veredicto usada abajo: 🟢 defendible con salvedad · 🟡 defendible solo como inspiración
de principio · 🔴 hay que renombrar o borrar la afirmación.

---

## 1. Cerebelo → contrato de predicción 🟡

**Neurociencia.** El modelo Marr–Albus–Ito, formalizado por Wolpert & Kawato, postula un *modelo
interno hacia adelante*: dado el comando motor y el estado, el cerebelo predice la consecuencia
sensorial; las fibras trepadoras llevan el error a las células de Purkinje y disparan LTD. La
evidencia moderna complica la versión de manual: las fibras trepadoras codifican algo más parecido
a un **error de diferencia temporal** que a un error post-hoc (Ohmae & Medina 2015), llevan
señales **predictivas anticipatorias** y no solo de fallo (Heffley et al. 2018), y codifican
incluso **magnitud de recompensa esperada** (Larry et al. 2019).

**Nuestra implementación.** El LLM declara `expected_effect` en 4 categorías y `confidence` en
[0,1]; el error es `clamp((rank(obs) − rank(esp))/2) × confianza`.

**Veredicto.** La estructura funcional —predecir, comparar, aprender del error— sí es isomórfica al
ciclo predicción→fibra trepadora→LTD, y eso alcanza para citar el principio. Pero hay un **defecto
estructural sin análogo biológico**: en el cerebelo la señal de error viene de un canal sensorial
*independiente* de la predicción, mientras que acá la predicción **y** la ponderación de su propio
error (`× confianza`) salen del mismo modelo. Un modelo sistemáticamente sobreconfiado distorsiona
su propia señal de aprendizaje de forma correlacionada. Además, 4 categorías discretas están lejos
de la estimación continua de estado latente que sugiere la evidencia.

**Redacción honesta.** «Inspirado en el principio de aprendizaje supervisado por error del cerebelo,
con dos diferencias centrales: la predicción y la evaluación de confianza provienen del mismo
modelo (en el cerebelo el error llega por un canal independiente), y opera sobre categorías
discretas verbalizadas en vez de una estimación continua de estado.» Borrar cualquier "implementamos
un cerebelo".

**Experimento que lo pondría a prueba.** Predecir el **estado siguiente** (`ŝ_{t+1}` estructurado)
independientemente de la confianza verbal, medir el error por distancia y comparar contra la señal
actual. Test decisivo: error puro `f(pred, obs)` vs. error actual `f(pred, obs) × autoconfianza`.
Si el primero calibra mejor cuando el modelo está mal calibrado, el acoplamiento actual es una
debilidad, no fidelidad.

**Citas.** Wolpert & Kawato (1998) *Neural Networks* 11(7-8):1317; Ohmae & Medina (2015)
*Nat Neurosci* 18:1798 (nn.4167); Heffley et al. (2018) *Nat Neurosci* 21:1431 (s41593-018-0228-8);
Larry et al. (2019) *eLife* 8:e46870.

---

## 2. Ganglios basales → gate de veto 🟢

**Neurociencia.** El modelo Albin–DeLong (vía directa D1 "Go" / indirecta D2 "No-Go" / hiperdirecta
córtex→STN como freno rápido; Frank 2006 "hold your horses") es el de manual. La evidencia lo
complica seriamente: las vías directa e indirecta **se coactivan** antes del movimiento en vez de
oponerse (Cui et al. 2013), y codifican valor de forma **continua y relativa** entre poblaciones —
dSPN sube con el valor, iSPN baja— con roles finos y dependientes de contexto (Shin, Kim & Jung
2018), hasta el punto de que existe una revisión formal titulada "una reevaluación crítica" del
esquema de dos vías (Calabresi et al. 2014).

**Nuestra implementación.** `valor = confianza × dopamina − costo(riesgo) ≥ 0.10`, más freno para
irreversibles con confianza < 0.6 o tono < 0.4, más winner-take-all entre acciones paralelas.

**Veredicto.** Es el mapeo **más defendible de los tres** a nivel de arquitectura de control: gate
de umbral + freno de alta prioridad + selección competitiva es genuinamente isomórfico a
"acumulación de evidencia + freno STN + WTA". El freno hiperdirecto es la pieza mejor sostenida.
Pero: (i) si el paper usa la dicotomía Go/No-Go como justificación, está apoyándose en evidencia
que la contradice; (ii) el WTA sobre un escalar se parece más a un argmax de utilidades que a la
competencia entre canales corticoestriatales con inhibición lateral; (iii) los ganglios basales
reales tienen **múltiples circuitos paralelos segregados** (motor, asociativo, límbico) con
umbrales distintos, no un gate escalar global.

**Redacción honesta.** «Inspirado en la arquitectura funcional de selección de acción de los
ganglios basales —umbral de utilidad neta con un freno de alta prioridad análogo a la vía
hiperdirecta— sin reproducir la codificación poblacional continua entre vías (Shin et al. 2018),
la coactivación concurrente (Cui et al. 2013) ni la segregación en circuitos paralelos.»

**Experimento.** Reemplazar el escalar por **dos señales poblacionales** (pro-acción que crece con
valor, anti-acción que crece con riesgo/incertidumbre, actualizadas de forma relativa) y comparar
calibración de decisiones. Si no hay diferencia medible, el nombre es decorativo.

**Citas.** Cui et al. (2013) *Nature* 494:238 (nature11846); Shin, Kim & Jung (2018)
*Nat Commun* 9:404 (s41467-017-02817-1); Calabresi et al. (2014) *Nat Neurosci* 17(8):1022
(nn.3743); Frank (2006) *Neural Networks* 19(8):1120.

---

## 3. Dopamina → estimador de éxito 🔴 (renombrar)

**Neurociencia.** Schultz, Dayan & Montague (1997) establecieron que las neuronas dopaminérgicas
codifican un **error de predicción de recompensa** con la forma del error TD:
`δ = r + γV(s_{t+1}) − V(s_t)` — una señal **con signo**, dependiente de la expectativa previa y
del valor del estado siguiente. La evidencia moderna va más lejos: el código es **distribucional**
(neuronas con distinta asimetría optimista/pesimista reconstruyen la distribución completa de
recompensa futura; Dabney et al. 2020), la población es **funcionalmente heterogénea** (Engelhard
et al. 2019) y hay señales de **novedad/saliencia** independientes del RPE en estriado posterior
(Menegas et al. 2017).

**Nuestra implementación.** `p̂ = (éxitos + 2·0.6)/(éxitos + fracasos + 2)` por par
(clase de escena, acción). Esto es, literalmente, un **estimador Beta-Bernoulli con prior
informativo**: la forma canónica de estimar una tasa de éxito en un bandit contextual.

**Veredicto.** El más débil de los tres, y el que un comité va a atacar primero. No comparte con la
dopamina real ni la matemática (frecuencia marginal vs. bootstrapping temporal), ni el contenido
informacional (tasa acumulada vs. señal con signo respecto a la expectativa), ni la estructura
(escalar único vs. código poblacional distribucional). No hay propagación de crédito: es
aprendizaje de bandit de un paso, no RL secuencial. Lo único compartido —"usar el historial para
modular la confianza"— describe también a UCB, Thompson sampling y cualquier estimador bayesiano
de tasa.

**Redacción honesta.** Renombrar en el paper a **estimador bayesiano de confiabilidad contextual**
(o "tasa de éxito Beta-Bernoulli por clase de escena"), y dejar la dopamina como inspiración lejana
citada una vez, nunca como nombre del mecanismo.

**Experimento.** Reemplazar `p̂` por un error TD real con `V(escena)` y usar `δ` con signo como el
historial que multiplica la confianza; correr ambas variantes en una tarea donde la recompensa
esperada dependa del contexto secuencial. Si el TD no gana, la etiqueta era decorativa.

**Citas.** Schultz, Dayan & Montague (1997) *Science* 275:1593; Dabney et al. (2020) *Nature*
577:671 (s41586-019-1924-6); Engelhard et al. (2019) *Nature* 570:509 (s41586-019-1261-9);
Menegas et al. (2017) *eLife* 6:e21886.

---

## 4. Hipocampo → memoria episódica por escena 🟡

**Neurociencia.** El circuito trisináptico reparte trabajo: el **giro dentado** hace *separación de
patrones* (esparce entradas solapadas para que terminen en códigos casi ortogonales, minimizando
interferencia), **CA3** hace *completado de patrones* vía dinámica atractora recurrente, y la
**teoría del índice hipocampal** (Teyler & Rudy 2007) sostiene que el hipocampo no guarda el
contenido sino un índice que reactiva el patrón neocortical del episodio. Hay además codificación
de **secuencia y tiempo** (células de tiempo, Eichenbaum 2014), y el marco de *Complementary
Learning Systems* (McClelland et al. 1995) explica por qué hace falta un aprendiz rápido y
ortogonalizante junto a uno lento.

**Nuestra implementación.** SQLite indexado por hash de features; recall por escena exacta o ≥2
features en común, ordenado por (misma escena, overlap, fracaso primero, fuerza).

**Veredicto.** Es **recuperación asociativa estándar** (kNN simbólico sobre features discretas),
arquitectónicamente cercana al *case-based reasoning* clásico. El punto más filoso: la separación
de patrones haría que estímulos parecidos se vuelvan *más* distintos para evitar interferencia;
nuestro recall hace **exactamente lo contrario** —más features compartidas, más prioridad—, o sea
la operación inversa a la de DG. Tampoco hay completado (se devuelven registros exactos, no un
patrón reconstruido) ni secuencias/tiempo (las memorias son bolsas de features sin orden). Lo único
defendible es el paralelo de alto nivel con la teoría del índice.

**Redacción honesta.** «Almacén episódico indexado por contexto situacional, inspirado en la idea
de índice hipocampal.» Borrar toda mención a separación/completado de patrones, CA3, DG o células
de lugar/tiempo como si el sistema las implementara.

**Experimento.** Construir un set de *casi-colisiones*: episodios con ≥2 features compartidas y
resultado opuesto, y episodios con <2 features literales pero misma escena latente. Medir
interferencia y fallos de recuperación; comparar contra un recall por embeddings con pérdida
contrastiva. Si no hay diferencia, la etiqueta no aporta nada computacional.

**Citas.** Yassa & Stark (2011) *Trends Neurosci* 34(10):515; Teyler & Rudy (2007) *Hippocampus*;
Eichenbaum (2014) *Nat Rev Neurosci* (nrn3827); McClelland, McNaughton & O'Reilly (1995)
*Psychol Rev*; *Nat Rev Neurosci* (2023) s41583-023-00710-z.

---

## 5. Amígdala / habénula → valencia y prioridad 🔴 (renombrar)

**Neurociencia.** La **amígdala** basolateral *no almacena* memoria: modula la consolidación que
ocurre en otras estructuras, vía noradrenalina y glucocorticoides, en una ventana post-codificación
(McGaugh 2004). Es control de ganancia neuromodulatorio ligado a *arousal*, no un cómputo de signo
de error. La **habénula lateral** (Matsumoto & Hikosaka 2007) se excita ante estímulos que predicen
ausencia de recompensa e inhibe a las dopaminérgicas: es parte de un circuito que computa un **RPE
con signo, gradual**, acoplado a plasticidad. La escritura gateada por sorpresa sí tiene sustento
sólido: el bucle hipocampo-VTA (Lisman & Grace 2005) y los errores de predicción que actualizan
memoria episódica (Bein et al. 2021).

**Nuestra implementación.** `valence` = error con signo; `habenula = (error < 0)` da prioridad de
recuperación; se escribe solo si |error| ≥ 0.25 o si la acción resolvió.

**Veredicto.** Mezcla una pieza buena con dos etiquetas malas. **El write-gate por sorpresa es
defendible** y bien citable. Pero el escalar `valence` llamado "amígdala" no captura un control de
ganancia neuromodulatorio, y **el flag booleano `habenula` es el mapeo más débil de todo el
sistema**: la LHb real codifica un RPE gradual acoplado a dopamina que modula la tasa de
aprendizaje; nuestro `if error < 0` reordena un `ORDER BY`. Confunde dos roles computacionales
distintos: "cuánto ajustar la plasticidad" vs. "qué recuerdo sale primero". Lo que sí sostiene
"los fracasos se recuerdan primero" es la literatura de **sesgo de negatividad**, no la habénula.

**Redacción honesta.** «Escritura de memoria gateada por sorpresa (Lisman & Grace 2005)» y
«priorización de fracasos en la recuperación, consistente con el sesgo de negatividad». Sacar la
palabra habénula del nombre del mecanismo.

**Experimento.** Tres condiciones: (1) flag booleano actual; (2) ablación con orden aleatorio o por
recencia manteniendo el write-gate; (3) `|valence|` como **modulador de peso en la destilación**
(episodios con más error pesan más al construir la regla), imitando el rol real de un RPE graduado.
Si (1) ≈ (2), el flag es decorativo; si (3) gana, la señal con signo aporta algo real.

**Citas.** McGaugh (2004) *Annu Rev Neurosci* 27:1; Matsumoto & Hikosaka (2007) *Nature* 447:1111
(nature05860); Lisman & Grace (2005) *Neuron* 46:703; Bein et al. (2021) *PNAS*.

---

## 6. Sueño / microglía → `consolidate()` 🔴 en 3 de 4 pasos

**Neurociencia.** El replay hipocampal durante *sharp-wave ripples* es **reactivación activa y
estructurada de secuencias**, además **priorizada** por recompensa (Ambrose, Pfeiffer & Foster
2016) y usada para *seleccionar* qué experiencias se preservan (*Science* 2024). La consolidación
sistémica (Diekelmann & Born 2010) transfiere gradualmente del hipocampo a la neocorteza por
reactivación repetida. La hipótesis de homeostasis sináptica (Tononi & Cirelli 2014) describe un
*downscaling* global multiplicativo ligado a onda lenta. La poda microglial (Schafer et al. 2012)
elimina **sinapsis dentro de una red viva** por fagocitosis dependiente de complemento — y su
generalidad está en debate. La extracción de *gist* por solapamiento de información (IOtA; Lewis &
Durrant 2011) es gradual y probabilística, sobre muchos eventos de reactivación.

**Nuestra implementación.** Decaimiento `0.9^días`, refuerzo ×1.2 tras ≥3 accesos, poda por umbral,
y destilación por intersección de features con soporte ≥3.

**Veredicto.** En conjunto es **un job de mantenimiento de base de datos** con un solo componente
cognitivamente real. El decaimiento es una curva de olvido tipo Ebbinghaus, no SHY. El refuerzo por
frecuencia no tiene nada de replay (sin compresión temporal, sin estructura de secuencia, sin
priorización por valor). "Microglía" pegado a un `DELETE WHERE strength < umbral AND access_count
= 0` no sobrevive el escrutinio. **La destilación sí es defendible**: la intersección de features
sobre un conjunto de soporte instancia, de forma tosca, la lógica inductiva de IOtA — con la
salvedad de que en el cerebro es gradual y probabilística, y acá es un AND determinista de una
pasada, probablemente frágil ante qué 3 episodios cruzaron el umbral.

**Redacción honesta.** «La destilación está inspirada en la extracción de *gist* por solapamiento
de información (Lewis & Durrant 2011)». Describir decaimiento, refuerzo y poda como lo que son:
ingeniería de caché. Borrar "microglía" y "sueño" como afirmaciones de mecanismo.

**Experimento.** Comparar `consolidate()` contra una línea base despojada (eviction LRU + minería
genérica de itemsets frecuentes con soporte ≥3) sobre tareas cuya estructura profunda se mantiene
pero cuyas features superficiales cambian. Medir transferencia y **estabilidad de la regla** al
remuestrear qué episodios cruzan el umbral. Si la línea base empata, la maquinaria con sabor neuro
es decorativa.

**Citas.** Diekelmann & Born (2010) *Nat Rev Neurosci* 11:114 (nrn2762); Tononi & Cirelli (2014)
*Neuron*; Schafer et al. (2012) *Neuron* 74(4):691; Lewis & Durrant (2011) *Trends Cogn Sci*;
Ambrose, Pfeiffer & Foster (2016) *Neuron* 91:1124.

---

## 7. Caudado → putamen (hábitos) 🔴 (renombrar: caché procedural)

**Neurociencia.** El marco Balleine & Dickinson define el hábito por **insensibilidad**: tras
devaluar el resultado (saciedad específica, aversión) o degradar la contingencia acción-resultado,
la conducta habitual **sigue** y la dirigida a objetivo cae. Yin & Knowlton (2006) mapearon la
transición DMS→DLS. Tres complicaciones: el arbitraje entre sistemas es **por incertidumbre y
continuo**, no un interruptor por número de repeticiones (Daw, Niv & Dayan 2005); el sustrato del
lado dirigido a objetivo cambia con el tiempo de entrenamiento (Bradfield et al. 2020, *Nat
Neurosci* 23:1194); y **cinco experimentos fallaron en inducir hábito humano por sobre-entrenamiento**
(de Wit et al. 2018, *J Exp Psychol Gen* 147(7):1043). El constructo está bajo fuego en su propia casa.

**Nuestra implementación.** 3 éxitos + 0 fracasos de la misma acción → se compila y ejecuta sin
LLM; fuerza 0.8; un intento por episodio; nunca sobre irreversibles; **al primer fallo la fuerza
cae a la mitad**.

**Veredicto.** El hallazgo más filoso de toda la revisión: **falla por el lado contrario**. La
firma de un hábito real es la insensibilidad al valor del resultado; nuestro mecanismo se
des-habitúa en un solo trial fallido, o sea es **hipersensible a la contingencia** — el rasgo que
define el control *dirigido a objetivo*, no el habitual. Si corriéramos el test canónico de
devaluación, el sistema fallaría por exceso de sensibilidad. Es una **caché procedural gatillada
por frecuencia**; ni siquiera es un buen análogo model-free, que acumula valor gradualmente en vez
de contar "3 éxitos y listo".

**Redacción honesta.** «Caché procedural inspirada de forma laxa en la compilación
estímulo-respuesta», con la advertencia explícita de que, a diferencia de un hábito biológico,
**es más sensible al resultado** que el proceso deliberativo que reemplaza. Borrar toda mención a
DMS→DLS como si se modelara.

**Experimento (implementable acá).** Devaluación sin fallo mecánico: dejar compilar el hábito, y
en un **trial de sonda en extinción** cambiar el objetivo de modo que la acción ya no sea deseable
—sin que la herramienta devuelva error ni veto— y medir si el hábito dispara igual. Complementar
con degradación de contingencia (que el buen resultado ocurra sin ejecutar la acción). Hoy el
diseño no tiene camino para eso: el único gatillo de debilitamiento es "la herramienta devolvió
algo malo", nunca "el objetivo cambió".

**Citas.** Yin & Knowlton (2006) *Nat Rev Neurosci* 7:464 (nrn1919); Balleine & Dickinson (1998)
*Neuropharmacology* 37:407; Daw, Niv & Dayan (2005) *Nat Neurosci* 8:1704 (nn1560); Bradfield et al.
(2020) *Nat Neurosci* 23:1194 (s41593-020-0693-8); de Wit et al. (2018) *J Exp Psychol Gen* 147(7):1043.

---

## 8. Ínsula → tono 🔴 (renombrar: monitoreo de recursos)

**Neurociencia.** La interocepción es la representación aferente del estado fisiológico vía lámina I
y vago hasta la ínsula posterior, re-representada hacia la anterior (Craig). Critchley et al. (2004)
mostraron que la precisión en detectar latidos correlaciona con actividad **y volumen de materia
gris** en ínsula anterior derecha. Tan et al. (2018) disocian sub-regiones (la ínsula media liga
precisión interoceptiva con ansiedad): "ínsula" no es monolítica. El marco moderno (Seth 2013) la
describe como **comparador predictivo**: el error entre la predicción descendente del estado
corporal y la señal aferente real es lo que constituye el sentimiento. Y la hipótesis del marcador
somático de Damasio tiene una crítica dura y replicada (Maia & McClelland 2004, *PNAS*
101(45):16075): en el Iowa Gambling Task los participantes tenían conocimiento explícito suficiente
para explicar su desempeño, sin necesidad de invocar un marcador inconsciente.

**Nuestra implementación.** `tone = clip(1 − 0.15·fallos − 0.5·pasos/presupuesto, 0.05, 1)`,
verbalizado con frases de coaching y usado para modular cuántos recuerdos entran.

**Veredicto.** **Telemetría con nombre bonito**, y la ruptura es estructural, no estilística: la
interocepción real es **inferencia sobre un estado oculto y ruidoso** —por eso existe la "precisión
interoceptiva" como variable individual medible—, mientras que `steps`, `failures` y `blocks` son
contadores exactos que el sistema ya posee con certeza total. No hay nada que estimar, no hay
comparador, no hay error de predicción. Es **monitoreo metacognitivo de recursos**; el encuadre
corporal es *framing* lingüístico sobre información que el modelo podría derivar leyendo su propio
historial.

**Redacción honesta.** «Escalar de monitoreo de recursos verbalizado con lenguaje de estado
interno», sin citar a Damasio/Craig/Critchley como si describieran el mecanismo. Y **correr el
brazo `telemetry` antes de publicar** cualquier afirmación fuerte: el control ya está implementado
y el propio comentario del código admite qué probaría.

**Experimento.** (1) El ya armado: `telemetry` vs `full` vs `protocol` con el protocolo congelado.
(2) El que tocaría algo estructural: exponer al modelo solo un derivado **ruidoso o retrasado** del
estado real, dejar la cuenta verdadera fuera del contexto, y medir si el desempeño se degrada
específicamente cuando el tono deja de trackear el estado real — análogo del paradigma de precisión
interoceptiva.

**Citas.** Critchley et al. (2004) *Nat Neurosci* 7(2):189 (nn1176); Tan et al. (2018) *Sci Rep*
8:17280 (s41598-018-35635-6); Maia & McClelland (2004) *PNAS* 101(45):16075; Seth (2013)
*Trends Cogn Sci* 17(11):565.

---

## 9. Tálamo/reticular y cingulado anterior 🟡 / 🔴 (mal etiquetado)

**Neurociencia.** El núcleo reticular talámico inhibe selectivamente canales talámicos; McAlonan,
Cavanaugh & Wurtz (2006) dieron evidencia directa en monos de modulación atencional opuesta entre
geniculado y TRN. Halassa & Kastner (2017) muestran que el tálamo **construye redes corticales
específicas de tarea**, no es una perilla de volumen. El sistema locus coeruleus-noradrenalina
(Aston-Jones & Cohen 2005) sí fija una **ganancia global** ligada a utilidad de tarea y al balance
exploración/explotación. Para el cingulado: Botvinick et al. (2001) definen conflicto como
**co-activación de respuestas incompatibles**, y Shenhav et al. (2013, EVC) describen al ACC dorsal
integrando beneficio esperado del control contra su costo.

**Nuestra implementación.** Ganancia: `k = min(4, round(2 + 2·tone))` — o sea 2, 3 o 4 precedentes.
Cingulado: `reconsider` cuando `|valor − umbral| < 0.08`.

**Veredicto.** La ganancia es **la analogía de forma más limpia** del sistema (un estado global
regula cuánta información entra, en línea con la ganancia adaptativa de LC-NE), pero con rango
dinámico trivial (tres valores) y **sin selectividad por canal**, que es justamente lo distintivo
del TRN. El cingulado está **mal etiquetado**, y este es un hallazgo propio de la revisión:
"conflicto" en la literatura es competencia entre respuestas simultáneas — y el sistema **sí tiene**
ese mecanismo: el **winner-take-all sobre acciones paralelas**. El paper le puso el nombre
"cingulado" al chequeo de proximidad al umbral, que mide **incertidumbre**, no conflicto. El
mecanismo con mejor derecho al nombre es el otro.

**Redacción honesta.** «Modulación de ganancia de contexto de grano grueso, inspirada en la
ganancia adaptativa (Aston-Jones & Cohen 2005)»; y mover la referencia de conflicto al arbitraje
winner-take-all, describiendo el `reconsider` por proximidad como lo que es: un paso extra de
deliberación bajo incertidumbre, emparentado con EVC.

**Experimento.** Ganancia: romper el acoplamiento `tone→k` (permutar o fijar `k=3`) y ver si cae el
desempeño; probar gating **selectivo por canal** (recortar primero los recuerdos de menor solape)
contra el corte parejo. Conflicto: tres brazos — reconsider por umbral (actual), reconsider
disparado por competencia real entre acciones de valor cercano, y ambos — midiendo en cuál el paso
extra corrige errores más seguido.

**Citas.** McAlonan, Cavanaugh & Wurtz (2006) *J Neurosci* 26(16):4444; Halassa & Kastner (2017)
*Nat Neurosci* 20:1669; Aston-Jones & Cohen (2005) *Annu Rev Neurosci* 28:403; Botvinick et al.
(2001) *Psychol Rev* 108(3):624; Shenhav, Botvinick & Cohen (2013) *Neuron* 79:217.

---

## Síntesis provisoria (componentes 1–3)

Ranking de honestidad del mapeo, de más a menos defendible: **ganglios basales** (analogía
arquitectónica razonable con salvedades citables) → **cerebelo** (principio correcto, con un
defecto estructural propio que hay que declarar) → **dopamina** (renombrar: es un estimador
Beta-Bernoulli estándar).

Recomendación transversal: agregar al paper una sección explícita de **límites de la analogía** con
estas citas, en vez de dejar que el lector infiera fidelidad mecanicista del nombre del componente.
Eso es exactamente lo que separa una metáfora funcional honesta de la sobreventa.
