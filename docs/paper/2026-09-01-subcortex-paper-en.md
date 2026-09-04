# Subcortex: a subcortical layer for LLM agents

## Design from the anatomy of a decision, and validation across three worlds

**Juan F. Castro Piccolo** · September 1, 2026

---

## Abstract

An agent built on a language model decides in a single shot: the model proposes and the system
executes. In the brain, the cortex proposes and everything else — basal ganglia, cerebellum,
hippocampus, amygdala, insula, habenula — authorizes, predicts, remembers, compiles and learns.
We start from a neuroanatomical walkthrough of a simple decision ("Anatomy of a Yes") and use it
as a map of what is missing from an agent in which the LLM is only the cortex. We implement that
map as `subcortex`, a library of five plugins for Google ADK that attaches to any `LlmAgent`
with one line, without modifying the model or the prompt: mandatory prediction before acting
(cerebellum), veto-by-default with disinhibition of a single action (basal ganglia), episodic
memory by situation that writes on surprise or success (hippocampus, amygdala, habenula),
compiled habits that respond without calling the model (caudate → putamen), interoception
translated into language (insula), and offline consolidation (sleep). We evaluate it with the
same Gemini agent, with and without the layer, in three worlds: an operations simulator
(`opsworld`), real bugs injected into a Python library (`bugworld`), and a historical replay of
the author's momentum paper trader (`marketworld`). In `opsworld` the layer raises the score by
22 %, brings resolution to 100 % and cuts model calls by 34 % in the last third, with compiled
habits. In `marketworld`, over 40 real market decisions, five trajectories end at 89 ± 20 equity
against 50.5 for the same agent without the layer (5/5 above), and at weekly cadence it also
beats BTC; the advantage comes from stable behavior in bear regimes. In `bugworld` the layer
contributes nothing across three runs, and we explain why: without recurring situations or
reusable actions, no subcortical mechanism has leverage. A replication with a second engine (Claude Sonnet 5,
through an adapter that translates tool-calling into a JSON contract) reproduces the direction
in `marketworld` and shifts `opsworld`'s gain from score to efficiency and safety: with a
stronger base model, the layer cuts calls by 21 % and harmful actions by two thirds without
giving up score. We document nine design lessons that emerged from the failures, the per-episode operating cost, and the limits of the study.

---

## 1. Introduction

Today's agent frameworks treat the language model as the complete decision system: they perceive
through tools, deliberate inside the model's context, and execute whatever the model emits.
Learning across episodes, where it exists at all, usually reduces to summarizing conversations
or retrieving text by similarity to the question. Nothing distinguishes "the action did not run"
from "the action did not work"; nothing compiles a repeated decision; nothing tells the model how
it is doing.

The hypothesis of this work is that this architecture reproduces only the cortex and omits what,
in a brain, makes a decision safe, cheap and improvable with use. What an LLM does better than
any other artifact is exactly what the orbitofrontal cortex does: convert heterogeneous arguments
into a common currency. What it lacks is the subcortical machinery.

Our starting point is a previous essay by the author that walks, step by step, through a minimal
decision — going for a run or not — from the retina to the spinal cord and back to learning,
emphasizing three conclusions: there is no single place where the decision happens, only a loop
that converges; emotion does not interfere with reason but enables it, by translating bodily
state into a comparable currency; and deciding is, mechanically, ceasing to inhibit. This paper
turns that map into software and subjects it to A/B experiments with the same model in three
domains.

Contributions:

1. An explicit mapping from fifteen neuroanatomical steps to mechanisms implementable in an
   agent, with the concrete formulas for each (section 3).
2. `subcortex`, an implementation on Google ADK that wraps any `LlmAgent` without touching it,
   with 82 tests that run without network access (section 4).
3. Three test beds and an A/B protocol with LLM-free references, replication and ablations
   (section 5), with positive results in two of them and an explained null result (section 6).
4. Eight design lessons that came out of the runs that did not work, including two
   incompatibilities with the Gemini 3 API and an infinite habit loop (section 7).

---

## 2. From the brain to the agent

The table summarizes the mapping. The column "missing from agents today" describes the state of
the LLM agent frameworks we know; the "subcortex" column names the component that implements it.

| Essay step | Brain mechanism | Missing from agents today | subcortex |
|---|---|---|---|
| 2, 4 — thalamus, reticular formation | Gain set by alertness | Everything enters the context with equal weight | `tone` modulates how many precedents are recalled |
| 5 — amygdala | Valence before deliberation; memories carry emotional tags | Neutral memories | Episodes with `valence`, `habenula` |
| 6–7 — hypothalamus, insula | Bodily state translated into a comparable feeling | The agent doesn't know how it is doing | `InteroceptionPlugin`: budget, failures, blocks → text + `tone` |
| 8 — hippocampus, parahippocampal | Retrieval by scene; simulation of futures | Retrieval by similarity to the question | Recall by situation features; the scene is enriched by what diagnosis discovers |
| 9 — anterior cingulate | Conflict detection; effort cost | No fast/slow router; no action cost | `reconsider` when value is near the threshold; cost per risk class |
| 10 — prefrontal | Common currency, intention | This is what the LLM does well | The LLM, unchanged |
| 11 — basal ganglia | Universal veto; disinhibit a single action; hyperdirect pathway | The default is to act | `GatePlugin`: value ≥ threshold, brake on irreversibles, one action per turn |
| 13 — cerebellum | Efference copy, predicted consequences, error | No declared expectation | `PredictionPlugin`: mandatory `expected_effect`, `confidence`; signed error |
| 15.1–15.2 — habenula, dopamine | Prediction error as the learning signal; a separate negative channel | Memory written by volume | Writing only on surprise or success; `habenula=True` with retrieval priority; dopamine per (scene, action) |
| 15.4 — caudate → putamen | Habit: stimulus-response that skips deliberation | Everything goes through the LLM | `HabitPlugin`: compiles after 3 successes of the same action; responds without the LLM; de-habituates |
| 0.4, 15.3 — microglia, sleep | Pruning, reinforcement, episodic → semantic consolidation | Memory grows monotonically | `consolidate()`: decay, pruning, reinforcement, distilled rules |

{{fig:f1-arquitectura}}

Two mapping decisions deserve comment. First: the tool-calling loop ADK already has — the model
proposes a call, the tool responds, the model looks again — is structurally the
cortico-striato-thalamo-cortical loop; that is why a veto breaks nothing: it returns as a tool
result and the model re-iterates. Second: a habit is implemented as a synthetic `LlmResponse`
the plugin returns before calling the model; the model is not invoked, and that is the only way
"compiling" can genuinely mean getting cheaper.

---

## 3. Design of subcortex

### 3.1 Entry point

```python
app = App(name="ops", root_agent=my_agent)
subcortex.attach(app, risk={"restart": "costly", "rollback": "irreversible", "resolve": "free"},
                 diagnostic_tools={"inspect_service"}, coarse_features=("symptom", "finding"))
```

`attach()` wraps the agent's action tools (adding two mandatory parameters), registers five
plugins in `App.plugins` in a fixed order, and opens a SQLite `EpisodicStore`. ADK executes
plugins in order and short-circuits the chain at the first non-null value; the order
`[Prediction, Gate, Memory, Habit, Interoception]` guarantees that in `before_tool` validation
precedes the veto, and in `after_tool` the prediction error is computed before memory, habit and
interoception consume it.

### 3.2 A two-level scene

A scene is the observable context of the decision. It has two keys: `key`, a hash of all
features (used by episodic recall), and `coarse_key`, a hash of a configured subset
(`coarse_features`) used by dopamine, the gate and habits. The distinction came out of a
failure: with a single fine key, no situation recurred within 40 episodes and nothing learned.
The scene is also enriched during the episode with what diagnostic tools discover
(`discover_fn` → a `finding` field), so that "symptom: high latency" and "symptom: high latency
+ database pool exhausted" are different classes.

### 3.3 Prediction and error

Every action tool requires `expected_effect ∈ {resolves, improves, no_change, worsens}` and
`confidence ∈ [0, 1]`. Tools return `observed_effect` on the same scale. With ranks
`worsens = −1, no_change = 0, improves = 1, resolves = 2`:

```
error = clamp((rank(observed) − rank(expected)) / 2, −1, 1) × confidence
```

Being wrong with high confidence weighs more. An `error < 0` is the habenula's signal.

### 3.4 Gate

By default everything is vetoed. An action is authorized if

```
value = confidence × dopamine(coarse_key, tool) − cost(risk) ≥ 0.10
```

with `cost = {free: 0, costly: 0.10, irreversible: 0.20}` and `dopamine = (successes + 0.6·2) /
(successes + failures + 2)` (prior 0.6). An irreversible action also has a hyperdirect pathway:
it is vetoed if `confidence < 0.6` or if `tone` is low, unless the action has already resolved
that scene class twice without failing ("earned trust"). If the model emits several actions in
parallel, only the highest-value one passes (winner-take-all). After three consecutive blocks,
only the escape actions (`escalate`, `resolve`) are authorized. Every veto returns a `reason`
and a `hint` listing the actions that did work in that scene class.

As an optional extension (front 2), the confidence the gate weighs can blend the model's declared
confidence with dopamine according to how much history exists: `w = n/(n+2)`; with zero
observations the model rules, with six the history outweighs it three to one.

### 3.5 Memory

An episode is written when `|error| ≥ 0.25` (surprise), when the tool failed, or when the action
resolved (success, added after the second world). It stores scene, action, arguments,
expectation, outcome, error, valence, `habenula` and a `strength`. Recall, before every model
call, retrieves up to four episodes with the exact same scene or at least two shared features,
failures first, and injects them as "Precedents". Dopamine is updated on every evaluated action,
with a definition of success that includes meeting a `no_change` expectation: holding steady in a
falling market is being right, not failing.

### 3.6 Habits

After three successes of **the same action with the same arguments** (templated: a value that
matches a feature is stored as `$feature`) in the same scene class, without failures, a habit is
compiled with strength 0.8. At the start of an episode whose class has a strong habit, the plugin
returns the call directly; the model is not invoked. A habit is attempted at most once per
episode and is halved in strength if its outcome is negative, invalid or vetoed. Habits are never
compiled over irreversible actions.

### 3.7 Interoception

It counts steps, real failures, blocks, consecutive invalid calls, and evaluated actions. It
produces a scalar `tone ∈ [0.05, 1]` that decreases with failures and with consumed budget — not
with vetoes, to avoid feeding them back — and a translated text block ("2 consecutive failures";
"NO PROGRESS: you used 60 % of the budget without a single evaluated result; if you don't have a
concrete change, escalate"). It is the insula: internal state in the same currency as the goal.

### 3.8 Consolidation

`consolidate()` decays the strength of unaccessed episodes, reinforces frequently recalled ones,
prunes the weak and old, and distills rules: for each (action, outcome) with at least three
episodes, the intersection of their features is the pattern and a text is generated ("in
incidents with `dispersion=wide`, `follow_momentum` tends to `worsens` (n=5)"). Optionally an
LLM writes richer rules, but only those backed by at least three episodes are kept: the model
cannot invent rules without evidence. A helper function suggests which features best separate
successes from failures (information gain), candidates for `coarse_features`.

### 3.9 Status contract

A tool result carries a `status`. Four statuses do not touch the world and no plugin learns from
them: `rejected` (missing prediction), `vetoed` (gate), `invalid` (malformed arguments, episode
already over) and `reconsider` (cingulate). The distinction between `invalid` and `error` turned
out to be decisive (section 7).

---

## 4. Implementation

`subcortex` is eight Python modules (~1,100 lines) on `google-adk` 2.8 and `pydantic`. The
plugins inherit from `BasePlugin` and use six hooks: `before_model`, `after_model`,
`before_tool`, `after_tool`, `on_tool_error` and session state. All transient state lives in
`session.state["subcortex.*"]`; persistent state in SQLite (`episodes`, `dopamine`, `habits`,
`habit_candidates`, `rules`). Every plugin catches its own exceptions and degrades to vanilla
behavior. There are 82 unit and integration tests that run without network access; the
integration tests use ADK's real `Runner` with a scripted model (`ScriptedLlm`) and verify,
among other things, that a veto makes the model re-iterate, that a habit executes without
calling it, and that the history the model sees after a habit contains no synthetic calls.

Two synthetic worlds and one real one accompany the library (~1,900 lines): an incident
simulator, an AST-mutation bug injector over a vendored library, and a market replay over real
daily candles. A generic A/B runner (`demo/ab.py`) executes the same agent with and without the
layer over the same sequence, with retries on transient errors, a per-episode call cap and
wall-clock timeout, incremental saving and `--resume`.

---

## 5. Methodology

**A/B design.** In each world a single `LlmAgent` is built (Gemini 3 Flash, fixed instruction,
the world's real tools) and run twice over the same episode sequence: *baseline* (vanilla ADK)
and *subcortex* (`attach()`). The only difference between variants is the `attach()` line. Each
episode is a fresh ADK session; the `EpisodicStore` persists across episodes.

**Metrics.** World score per episode, resolution rate, model calls, tokens, steps, harmful
actions (`worsens`), vetoes, episodes written, habit firings, mean prediction error. All are
reported by thirds of the sequence to expose learning.

**LLM-free references.** In `marketworld`, additionally, the pure momentum rule at two cadences
and BTC buy & hold over the same window.

**Replication and ablations.** In `marketworld`, five subcortex trajectories were run (the
baseline turned out deterministic) plus ablations disabling one plugin at a time.

**Worlds.**

*opsworld* — an on-call operator over a simulated fleet. Six hidden causes (memory leak, bad
deploy, traffic spike, saturated database, dependency down, false alarm), observable symptoms,
three diagnostic tools and six action tools with declared risk. Dynamics with traps: `restart`
fixes the leak but worsens the saturated database; `rollback` is useless and expensive in a
traffic spike; `failover_db` on a false alarm causes real damage. Score +100 for resolving,
−40 per worsening, −5 per step, per-action costs. 40 incidents, fixed seed.

*bugworld* — 40 real bugs injected by AST mutation (inverted arithmetic and comparison
operators, `and`↔`or`, negated conditions, off-by-one, `return None`, and bugs in the test
itself, where the right move is to escalate) into `toolz` 1.1.0 (181 tests, ~3 s). Real tools:
run pytest, read, search, edit, rewrite, revert, finish, escalate. Every edit re-runs the whole
suite. Budget: 10 calls.

*marketworld* — a replay of the author's momentum paper trader: 10 majors, real Binance daily
candles, the exact rule (top-2 by 30-day return among those above their SMA-100, 0.15 % per
side). 40 decisions every 21 days between March 2024 and June 2026 (bull and bear phases),
consequences measured at 21 days, persistent portfolio; a weekly variant with 120 decisions. The
agent never sees dates. Scene: BTC trend, breadth, volatility, dispersion, current holdings;
`finding` = regime. Actions: `follow_momentum`, `hold`, `go_cash`, `rotate`.

---

## 6. Results

### 6.1 opsworld: the layer works — on the second iteration

| metric | baseline | subcortex v1 | **subcortex v2** |
|---|---|---|---|
| mean score | 46.3 | 37.1 | **56.3** (+22 %) |
| resolution rate | 0.93 | 0.80 | **1.00** |
| LLM calls / episode | 6.7 | 7.4 | **5.0** |
| LLM calls, last third | 7.2 | 9.1 | **4.8** (−34 %) |
| tokens / episode | 11,574 | 24,814 | 12,066 |
| harmful actions | 4 | 3 | **2** |
| vetoes | — | 32 | 0 |
| habits compiled / fired | — | 0 / 0 | **4 / 9** |
| prediction error by third | — | 0.62 → 0.57 | 0.55 → 0.46 |

{{fig:f2-opsworld}}

The first iteration was worse than the baseline. The diagnosis — a scene key that was too fine,
a veto → tone → veto loop, and a value formula that multiplied by tone and made it impossible to
authorize an irreversible action mid-episode — produced iteration 2, which validates the
hypothesis: equal score in the first third (no memory yet), 34 % fewer calls in the last (four
decisions already compiled into habits), half the damage, tokens at par. By cause, the gain
concentrates where the baseline fails repeatedly: `db_saturated` (−16 → +37) and
`dependency_down` (−45 → +2.5). A Gemini 3 issue (it rejects function calls lacking a
`thought_signature`) surfaced at the first habit firing and was solved by rewriting the
synthetic-call/result pair as text in the history, in a provider-independent way.

### 6.2 bugworld: a three-way tie, explained

| | baseline | subcortex v1 | subcortex v2 (+success, +no-progress) |
|---|---|---|---|
| mean score | 22.5 | 25.0 | 19.0 |
| solved | 28/40 | 29/40 | 26/40 |
| LLM calls / bug | 12.6 | 11.8 | 11.8 |
| episodes written | — | 0 | 26 (20 distinct scenes) |
| escalations on `broken_test` | 0/5 | 0/5 | 0/5 |

All three runs fall within the model's noise (±5). The decisive row is "episodes written": in
the first version, zero. The only evaluable action per episode was the valid edit, and when an
edit applies it is almost always the right one — expected `resolves`, observed `resolves`, zero
error, no surprise; everything else was invalid calls which, by contract, are not outcomes.
Adding success-writing produced 26 episodes across 20 distinct scenes: nothing recurs at the
granularity where decisions happen, and a precedent from another bug with the same symptom
describes another function and another fix. The no-progress signal reached the model and did not
change its behavior: with the problem in front of it, it prefers to keep trying over giving up.
Conclusion: where every episode is unique, the action is not reusable, and the wrong behavior is
*not acting*, no subcortical mechanism has leverage. This is a limit of the approach, not a
pending tweak.

### 6.3 marketworld: validation on the real problem

**LLM-free references (same window):** daily momentum rule 78.6; the rule at the agent's cadence
(every 21 days) 50.5; BTC 95.3. A hard window for momentum.

**Five subcortex trajectories, deterministic baseline:**

| trajectory | final equity | bear regime: mean per decision | `hold` / `follow` in bear |
|---|---|---|---|
| run 1 | 90.9 | −1.22 % | 12 / 4 |
| run 2 | 121.0 | −1.29 % | 13 / 2 |
| run 3 | 83.2 | −1.21 % | 13 / 3 |
| run 4 | 68.3 | −1.29 % | 12 / 4 |
| run 5 | 83.2 | −1.21 % | 13 / 3 |
| **subcortex, mean ± sd** | **89.3 ± 19.5** | **−1.24 %** | |
| baseline (40/40 identical decisions across two runs) | 50.5 | −3.51 % | 9 / 5 |

{{fig:f3-marketworld}}

Five out of five above the baseline; four above the daily rule; two above BTC. The baseline is
exactly the 21-day rule: it followed `follow_momentum` 26 times and its deviations changed
nothing. Subcortex's bear-regime behavior is nearly identical across the five trajectories
(12–13 `hold` out of 19, −1.2 to −1.3 % per decision): that is the mechanism, and it is stable;
the equity dispersion (68–121) comes from bull-regime decisions, where the model's variance
dominates. The layer distilled correct rules with no labels — "with BTC trending,
`follow_momentum` tends to `resolves` (n=5)", "with wide dispersion, it tends to `worsens`
(n=5)", "with BTC bearish, `go_cash` tends to `no_change` (n=4)" — and in several trajectories
compiled a `hold` habit for the dominant bear regime; in one, a `follow_momentum` habit failed
once and de-habituated.

**Weekly variant (120 decisions, 7-day horizon):**

| runner | final equity | bear regime (58 decisions) |
|---|---|---|
| baseline | 91.4 | −0.62 % |
| **subcortex** | **101.6** | **+0.16 %** |
| rule at the agent's cadence | 90.9 | |
| daily rule | 78.6 | |
| BTC | 95.3 | |

With three times more samples, habits fired 11 times (2 compiled, 6 rules) and subcortex ended
above BTC. A single trajectory.

**Ablations (baseline reused; one trajectory per variant, same 40 decisions):**

| variant | final equity | bear regime: mean per decision |
|---|---|---|
| full layer (5 trajectories) | 89.3 ± 19.5 | −1.24 % |
| no memory | 69.2 | **−3.50 %** (= baseline) |
| no interoception | 69.7 | −2.27 % |
| no habits | 74.9 | −1.22 % |
| no gate | 91.5 | −1.28 % |
| baseline | 50.5 | −3.51 % |

{{fig:f4-ablaciones}}

With the caveat that each ablation is a single trajectory against a mean with σ ≈ 20, the
behavioral signal is sharp: **without memory, bear-regime behavior returns exactly to the
baseline's** (−3.50 % vs −3.51 %); interoception adds prudence; habits add something; the gate
adds nothing in this domain — consistent with the zero vetoes across all runs: there are no
irreversible actions to brake. The history-confidence and reconsider variants landed within the
full layer's range (92.3 and 69.8; inconclusive with one trajectory), and the LLM-rules run was
left incomplete (14/40), so it is not reported.

### 6.4 Operating efficiency

The layer is not expensive by design: its cost depends on whether it saves calls (habits) faster
than it adds context (precedents). Figure 5 compares model calls, tokens and wall-clock time per
episode. In `opsworld`, subcortex is cheaper than the baseline on all three dimensions (5.0 vs
6.7 calls, tokens at par, 10.2 vs 12.2 s). In `marketworld` it pays a token overhead (~2×) and
extra seconds for in-context precedents and per-decision evaluation, with comparable calls. The
general rule we observed: where there is repetition, habits end up paying for the memory; where
there is not, memory is pure overhead.

{{fig:f5-eficiencia}}

### 6.5 A second engine: Claude

To separate the architecture from the model that runs it, we repeated the `opsworld` and
`marketworld` A/Bs with Claude Sonnet 5 as the engine, through an adapter that implements ADK's
`BaseLlm` on top of the local Claude Code CLI: tools travel as a schema-validated JSON contract
and the reply comes back as a native `FunctionCall`, so the five plugins run without a single
change. Same methodology, n=40 per arm.

| metric | ops: baseline | ops: subcortex | market: baseline | market: subcortex |
|---|---|---|---|---|
| mean score | 42.0 | 41.0 | 1.7 | **9.6** |
| resolution rate | 0.82 | 0.72 | — | — |
| LLM calls / episode | 6.0 | **4.8** | 3.1 | 2.8 |
| harmful actions | 6 | **2** | 11 | **6** |
| habit firings | 0 | 6 | 0 | **11** |

{{fig:f6-motores}}

Three readings. In `marketworld` the direction reproduces — subcortex ends above the baseline,
within the range of the five Gemini trajectories — and for the first time habits fired in this
world (11 times, with one correct dehabituation when the regime changed): Claude declares higher
confidences and its repeated successes compile earlier. In `opsworld`, Claude's baseline already
solves the causes that cost Gemini dearly and the score margin disappears; what remains is the
structural part — 21 % fewer calls, a third of the harmful actions, a final third with zero
worsening actions — which is what the analogy predicts: the basal ganglia do not make the cortex
smarter, they make it cheaper and less dangerous. The honest reading: resolution dropped ten
points, concentrated in episodes where a similar-but-not-identical precedent anchored the agent
into closing early; the stronger the base model, the finer the recall threshold must be so that
memory does not compete with cold judgment that was already good (§9). The adapter left a lesson
of its own (lesson 9): with contract-based tool-calling, the obligation to act must be written
down. Operationally the engine is ~2× slower (one CLI process per call) and its token counters
are not comparable with the API's, so figure 6 compares calls, harm and score.

---

## 7. Design lessons

Each one came out of a run that did not work, and stayed in the code with its test.

1. **The scene needs two levels.** With a single fine key nothing recurs and nothing learns;
   with a coarse one, recall loses precision. Dopamine, gate and habits use the coarse key;
   recall, the fine one.
2. **The scene is enriched by what diagnosis discovers.** The right precedent is the one that
   shares what was learned *after* looking, not only what was visible on entry.
3. **Tone neither multiplies value nor drops with vetoes.** Multiplying made it impossible to
   authorize an irreversible action mid-episode; dropping it with vetoes produced a loop that
   forced needless escalations.
4. **Meeting a "no change" expectation is success.** With absolute reward, `hold` in a bear
   regime had 0 successes and 8 failures; with prediction error, it is a hit. This is step 15 of
   the essay, not step 5.
5. **An invalid call is not an outcome.** Without the `invalid` status, three miscopied
   arguments sank the tone to 0.05 and `edit_file`'s dopamine to 0.2, and the gate vetoed edits
   made with 0.8 confidence.
6. **A habit is the same action with the same arguments**, attempted once per episode, weakened
   when it fails. Without that, edits from another bug were being compiled and the agent entered
   an infinite loop without calling the model (hung for 54 minutes).
7. **The consequence horizon must reach the next decision.** Measuring at 7 days with decisions
   every 21 made score and equity contradict each other.
8. **Synthetic function calls are rewritten as text.** Gemini 3 rejects history entries with
   calls the model did not generate; there is no documented dummy signature. Marking the habit's
   call and converting the call → result pair to text before each invocation is
   provider-independent and covered by tests.
9. **An output contract must oblige action.** With native tool-calling, the channel pushes the
   model to call tools; over a JSON-in-text contract, "answering without acting" is a valid
   output, and it appeared in 25 % of the first episodes with the second engine. The explicit
   prohibition — never final text without having executed at least one tool — removed it
   entirely (0 in 160 episodes).

---

## 8. Limits and threats to validity

- **Sample size.** 40 episodes per world (120 in the weekly variant). Directions are robust
  (5/5 replication in `marketworld`); magnitudes carry deviations on the order of half the
  advantage.
- **Two engines, one run per pair.** The main runs use Gemini 3 Flash; the Claude Sonnet 5
  replication (§6.5) confirms the direction in `marketworld` and the efficiency reading in
  `opsworld`, but it is one trajectory per world, and magnitudes are not directly comparable
  across engines: the tool-calling contracts differ.
- **Path-dependent non-determinism.** The baseline turned out deterministic in `marketworld`;
  subcortex did not, because a different decision changes which episodes exist afterwards. The
  observed variance is a property of the system, not just sampling noise.
- **Worlds.** `opsworld` is synthetic and its traps were designed by whoever designed the layer.
  `bugworld` uses a single repository. `marketworld` uses a window chosen for its mix of
  regimes, not for favoring the layer (the rule loses money in it), and the first 34 baseline
  rows of `bugworld` were reconstructed from logs without token counts.
- **Confidence calibration.** The gate weighs the confidence the model declares, and models are
  poorly calibrated. The blend with history was implemented but its measurement was still in
  progress at the time of writing.
- **Tools must return `observed_effect`.** In a real deployment that field must be produced by
  the world (tests, metrics, returns), not by the model; all three worlds do this, but it is an
  integration requirement that is not always met.

---

## 9. Future work

Three to five trajectories per variant in all worlds and per engine, to report means with
deviations. The Claude replication also leaves a question of its own: with a stronger base
model, episodic recall can over-anchor (ten resolution points in `opsworld`); the natural fix is
a recall threshold adaptive to the model's own hit rate. A fourth world with genuinely irreversible actions (operations on an automation
instance, sandboxed) where veto-by-default can show its value, which was marginal in all three
worlds. Scene learning: `coarse_features` was hand-picked per world; the information-gain
suggestion exists, but applying it without invalidating dopamine and habits requires a key
migration. And the natural next step of the originating project: running the layer alongside the
real paper trader as a second opinion, without touching the strategy, starting with the `hold`
habit in bear regimes and the dispersion rules the pure rule does not have.

---

## 10. Reproducibility

Repository `memory-tests`, branch `worktree-subcortex-poc`. `uv sync` installs everything;
`uv run pytest` runs the 91 tests (5 slow ones in `bugworld`) without network access.
`demo/run_ab.py`, `demo/run_bugs_ab.py` and `demo/run_market_ab.py` run the A/Bs with
`GOOGLE_API_KEY` in `demo/.env`; they accept `--baseline-from`, `--resume`, `--ablate`,
`--spacing/--horizon`, `--history-confidence`, `--reconsider`, `--llm-rules`. The
`SUBCORTEX_MODEL` environment variable picks the engine: a Gemini model string, or
`claude-code[:model]` for the `adapters/claude_code_llm.py` adapter over the local Claude Code
CLI, with no API key; the raw results for the second engine live in `results-ops-cc.json` and
`results-market-cc.json`. The `marketworld`
data are 10 daily-candle CSVs included in the repository. Raw results for every run live in
`results-*.json`; the reports with diagnoses in `docs/superpowers/results/`; the specifications
in `docs/superpowers/specs/`; the source essay in `documentation/anatomia-de-una-decision.md`.

---

## References

- Castro Piccolo, J. F. (2026). *Anatomía de un "sí": recorrido completo de una decisión simple, de la primera sinapsis a la última.* Essay, repository document.
- Damasio, A. R. (1994). *Descartes' Error: Emotion, Reason, and the Human Brain.* Putnam. (Somatic marker hypothesis.)
- Schultz, W., Dayan, P., & Montague, P. R. (1997). A neural substrate of prediction and reward. *Science*, 275(5306).
- Redgrave, P., Prescott, T. J., & Gurney, K. (1999). The basal ganglia: a vertebrate solution to the selection problem? *Neuroscience*, 89(4).
- Ratcliff, R. (1978). A theory of memory retrieval. *Psychological Review*, 85(2). (Drift-diffusion models.)
- Anderson, J. R., et al. (2004). An integrated theory of the mind. *Psychological Review*, 111(4). (ACT-R.)
- Laird, J. E. (2012). *The Soar Cognitive Architecture.* MIT Press.
- Shinn, N., et al. (2023). Reflexion: language agents with verbal reinforcement learning. *NeurIPS*.
- Wang, G., et al. (2023). Voyager: an open-ended embodied agent with large language models. *arXiv*.
- Park, J. S., et al. (2023). Generative agents: interactive simulacra of human behavior. *UIST*.
- Google. *Agent Development Kit (ADK) — Python*, version 2.8. Online documentation.
