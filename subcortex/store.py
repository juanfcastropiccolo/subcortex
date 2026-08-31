"""Almacén episódico (hipocampo) + contadores de dopamina (estriado) + hábitos + reglas."""
from __future__ import annotations

import json
import logging
import sqlite3
import time

from .types import Episode, Habit, Rule, Scene

log = logging.getLogger("subcortex")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
  id INTEGER PRIMARY KEY AUTOINCREMENT, scene_key TEXT, features TEXT, tool TEXT, args TEXT,
  expected TEXT, observed TEXT, prediction_error REAL, valence REAL, habenula INTEGER,
  strength REAL, access_count INTEGER, created_at REAL, last_access REAL);
CREATE INDEX IF NOT EXISTS ep_scene ON episodes(scene_key);
CREATE TABLE IF NOT EXISTS dopamine (
  scene_key TEXT, tool TEXT, successes INTEGER, failures INTEGER, PRIMARY KEY (scene_key, tool));
CREATE TABLE IF NOT EXISTS habits (
  scene_key TEXT PRIMARY KEY, tool TEXT, args TEXT, typical_effect TEXT, strength REAL,
  successes INTEGER, failures INTEGER);
CREATE TABLE IF NOT EXISTS rules (
  pattern TEXT, tool TEXT, text TEXT, support INTEGER, PRIMARY KEY (pattern, tool));
"""


class EpisodicStore:
    def __init__(self, path: str = ":memory:") -> None:
        try:
            self.conn = sqlite3.connect(path, check_same_thread=False)
        except sqlite3.Error as e:  # degradación: nunca romper al agente
            log.warning("SQLite no disponible en %s (%s); usando :memory:", path, e)
            self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # --- episodios -----------------------------------------------------------
    def write(self, ep: Episode) -> int:
        cur = self.conn.execute(
            "INSERT INTO episodes (scene_key, features, tool, args, expected, observed, prediction_error,"
            " valence, habenula, strength, access_count, created_at, last_access)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ep.scene_key, json.dumps(ep.features, sort_keys=True), ep.tool,
             json.dumps(ep.args, sort_keys=True), ep.expected, ep.observed, ep.prediction_error,
             ep.valence, int(ep.habenula), ep.strength, ep.access_count, ep.created_at, ep.last_access))
        self.conn.commit()
        return int(cur.lastrowid)

    def recall(self, scene: Scene, k: int, now: float | None = None) -> list[Episode]:
        """Misma escena primero, luego por overlap de features; dentro: habénula, luego fuerza."""
        now = time.time() if now is None else now
        rows = [self._row_to_episode(r) for r in self.conn.execute("SELECT * FROM episodes")]
        feats = list(scene.features.items())

        def overlap(e: Episode) -> int:
            items = e.features.items()
            return sum(1 for kv in feats if kv in items)

        cands = [e for e in rows if e.scene_key == scene.key or overlap(e) > 0]
        cands.sort(key=lambda e: (e.scene_key != scene.key, -overlap(e), not e.habenula, -e.strength))
        chosen = cands[:k]
        for e in chosen:
            e.access_count += 1
            e.last_access = now
            self.conn.execute("UPDATE episodes SET access_count=?, last_access=? WHERE id=?",
                              (e.access_count, now, e.id))
        self.conn.commit()
        return chosen

    def all_episodes(self) -> list[Episode]:
        return [self._row_to_episode(r)
                for r in self.conn.execute("SELECT * FROM episodes ORDER BY id")]

    def update_strength(self, episode_id: int, strength: float) -> None:
        self.conn.execute("UPDATE episodes SET strength=? WHERE id=?", (strength, episode_id))
        self.conn.commit()

    def delete_episodes(self, ids: list[int]) -> None:
        if ids:
            self.conn.executemany("DELETE FROM episodes WHERE id=?", [(i,) for i in ids])
            self.conn.commit()

    @staticmethod
    def _row_to_episode(r: sqlite3.Row) -> Episode:
        return Episode(id=r["id"], scene_key=r["scene_key"], features=json.loads(r["features"]),
                       tool=r["tool"], args=json.loads(r["args"]), expected=r["expected"],
                       observed=r["observed"], prediction_error=r["prediction_error"],
                       valence=r["valence"], habenula=bool(r["habenula"]), strength=r["strength"],
                       access_count=r["access_count"], created_at=r["created_at"],
                       last_access=r["last_access"])

    # --- dopamina --------------------------------------------------------------
    def record_outcome(self, scene_key: str, tool: str, success: bool) -> None:
        self.conn.execute(
            "INSERT INTO dopamine (scene_key, tool, successes, failures) VALUES (?,?,?,?)"
            " ON CONFLICT(scene_key, tool) DO UPDATE SET successes=successes+excluded.successes,"
            " failures=failures+excluded.failures",
            (scene_key, tool, int(success), int(not success)))
        self.conn.commit()

    def outcome_counts(self, scene_key: str, tool: str) -> tuple[int, int]:
        r = self.conn.execute("SELECT successes, failures FROM dopamine WHERE scene_key=? AND tool=?",
                              (scene_key, tool)).fetchone()
        return (r["successes"], r["failures"]) if r else (0, 0)

    def dopamine(self, scene_key: str, tool: str, prior: float) -> float:
        s, f = self.outcome_counts(scene_key, tool)
        return (s + prior * 2) / (s + f + 2)

    # --- hábitos ---------------------------------------------------------------
    def upsert_habit(self, h: Habit) -> None:
        self.conn.execute(
            "INSERT INTO habits (scene_key, tool, args, typical_effect, strength, successes, failures)"
            " VALUES (?,?,?,?,?,?,?) ON CONFLICT(scene_key) DO UPDATE SET tool=excluded.tool,"
            " args=excluded.args, typical_effect=excluded.typical_effect, strength=excluded.strength,"
            " successes=excluded.successes, failures=excluded.failures",
            (h.scene_key, h.tool, json.dumps(h.args, sort_keys=True), h.typical_effect, h.strength,
             h.successes, h.failures))
        self.conn.commit()

    def get_habit(self, scene_key: str) -> Habit | None:
        r = self.conn.execute("SELECT * FROM habits WHERE scene_key=?", (scene_key,)).fetchone()
        return self._row_to_habit(r) if r else None

    def habits(self) -> list[Habit]:
        return [self._row_to_habit(r) for r in self.conn.execute("SELECT * FROM habits")]

    @staticmethod
    def _row_to_habit(r: sqlite3.Row) -> Habit:
        return Habit(scene_key=r["scene_key"], tool=r["tool"], args=json.loads(r["args"]),
                     typical_effect=r["typical_effect"], strength=r["strength"],
                     successes=r["successes"], failures=r["failures"])

    # --- reglas ----------------------------------------------------------------
    def upsert_rule(self, rule: Rule) -> None:
        self.conn.execute(
            "INSERT INTO rules (pattern, tool, text, support) VALUES (?,?,?,?)"
            " ON CONFLICT(pattern, tool) DO UPDATE SET text=excluded.text, support=excluded.support",
            (json.dumps(rule.scene_pattern, sort_keys=True), rule.tool, rule.text, rule.support))
        self.conn.commit()

    def rules_for(self, scene: Scene) -> list[Rule]:
        out = []
        for r in self.conn.execute("SELECT * FROM rules ORDER BY support DESC"):
            pattern = json.loads(r["pattern"])
            if all(scene.features.get(k) == v for k, v in pattern.items()):
                out.append(Rule(scene_pattern=pattern, tool=r["tool"], text=r["text"],
                                support=r["support"]))
        return out

    def stats(self) -> dict:
        def count(table: str) -> int:
            return self.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]

        return {"episodes": count("episodes"), "habits": count("habits"), "rules": count("rules")}
