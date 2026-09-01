# marketworld — Proyecto Momentum en replay histórico

**Fecha:** 2026-09-01 · **Estado:** aprobado ("vayamos con momentum") · **Depende de:** spec de subcortex §10

## 1. Por qué este dominio

Tercer banco de pruebas, elegido porque tiene lo que bugworld no tenía: **situaciones que se
repiten** (regímenes de mercado), **acciones reutilizables** (`follow_momentum`, `hold`,
`go_cash`) y **resultado medible por decisión** (retorno a 7 días). Y porque es el proyecto real
de Juan: la regla del paper trader (`trading-bots/backend/scripts/momentum_paper.py`) se replica exacta.

## 2. El mundo

- **Datos:** velas diarias reales de Binance para los 10 majors del proyecto (`marketworld/data/`,
  1095 días hasta 2026-07-12, copiadas de la caché del repo). Sin red.
- **Regla de la casa (exacta):** top-2 por retorno 30d entre los que cotizan sobre su SMA-100,
  pesos iguales (mitad cash si califica uno; todo cash si ninguno), costo 0.15 % por lado,
  rebalanceo solo si cambia el conjunto.
- **Episodio = un día de decisión `t`.** 40 decisiones cada 21 días (abr-2024 → jul-2026: incluye
  la fase alcista 2024-25 y la bajista 2025-26). El agente **no ve la fecha**: evita que el modelo
  recuerde el mercado. La cartera persiste entre decisiones (replay secuencial).
- **Resultado:** retorno de la cartera resultante a 7 días, costos incluidos.
  `observed_effect`: > +3 % `resolves`; 0..+3 % `improves`; −3..0 `no_change`; < −3 % `worsens`.
  `score = round(1000 × retorno)` (+5 % → +50). `resolved` = retorno > 0.
- **Escena (entrada):** `btc_trend` (BTC vs SMA-100), `breadth` (cuántos de 10 sobre SMA-100:
  low ≤ 3, mid ≤ 6, high), `vol` (vol 30d de BTC vs su mediana de 1 año), `dispersion`
  (top-1 30d − mediana > 15 %), `holding` (cash/one/two). Descubierto por `market_snapshot`:
  `finding = f"{btc_trend}_{breadth}_{vol}"`. `coarse_features = ("finding", "holding")`.

## 3. Tools

Diagnósticas (gratis): `market_snapshot()` (ranking 30d/7d, sobre-SMA, régimen, target de la
regla, `finding`), `asset_detail(symbol)`, `portfolio()`.
Acciones (una por semana; cierran el episodio): `follow_momentum()` costly, `hold()` free,
`go_cash()` costly, `rotate(symbols)` costly (1–2 símbolos del universo; si no, `invalid`).
Presupuesto 5 llamadas; al agotarse sin decidir, `hold` forzado.

## 4. Referencias sin LLM

En la misma ventana: **regla pura → 129.4** y **BTC buy&hold → 76.7** (equity final desde 100).
Un gestor LLM que solo siguiera la regla igualaría 129.4; la pregunta es si la capa subcortical
lo ayuda a desviarse bien (cash en régimen bajista, seguir la regla en alcista) o mal.

## 5. Expectativa honesta

Es el dominio más favorable para el mecanismo (7 regímenes, 4 acciones repetibles, 40 muestras)
y a la vez el más ruidoso: el retorno a 7 días de cripto tiene desvío ~10 %, así que una
diferencia de equity final entre variantes con 40 semanas puede ser suerte. Lo que sí es
medible: cuántas veces cada variante siguió/rompió la regla por régimen, y si subcortex forma
hábitos por régimen (`down_low_high → go_cash/hold`, `up_high_low → follow_momentum`).
