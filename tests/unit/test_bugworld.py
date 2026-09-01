"""Tests del mundo real. Cada BugWorld corre la suite de toolz (~2 s), por eso son pocos."""

import pytest

from bugworld.mutate import VENDOR, Bug, enumerate_sites
from bugworld.world import BugWorld


def first_bug(kind="arith_swap", file="toolz/itertoolz.py", bug_kind=None, index=None) -> Bug:
    sites = enumerate_sites((VENDOR / file).read_text(), kind, file)
    return Bug(id=0, kind=bug_kind or kind, file=file, site_kind=kind,
               site_index=sites[0].index if index is None else index, description=sites[0].description,
               failing_tests=[])


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> BugWorld:
    """Un bug de constant_shift en itertoolz (línea temprana) que rompe algún test."""
    root = tmp_path_factory.mktemp("bw")
    src = (VENDOR / "toolz/itertoolz.py").read_text()
    for s in enumerate_sites(src, "constant_shift", "toolz/itertoolz.py"):
        bug = first_bug("constant_shift", index=s.index)
        w = BugWorld(bug, root)
        if 1 <= w.baseline["failed"] <= 8:
            return w
    pytest.skip("no se encontró un constant_shift que rompa 1–8 tests")


def test_baseline_features_and_intro(world):
    f = world.features()
    assert set(f) == {"module", "n_failing", "error_type"} and f["module"].startswith("test_")
    assert "fallo" in world.intro() and world.steps == 0 and not world.done


def test_diagnostics_cost_a_step_and_run_tests_has_finding(world):
    r = world.diagnose("run_tests", pattern="")
    assert r["failed"] == world.baseline["failed"] and r["finding"] in (
        "numeric_mismatch", "boundary", "exception", "none_result", "bool_flip", "order", "unknown")
    r = world.diagnose("read_file", path="toolz/itertoolz.py", start=1, end=5)
    assert "   1|" in r["content"] and r["total_lines"] > 100
    assert world.diagnose("read_file", path="../secret.py", start=1, end=2)["status"] == "error"
    assert world.diagnose("search", pattern=r"def interleave")["hits"]
    assert world.steps == 4 and world.score == -20


def test_edit_guards_and_revert_resolves(world):
    assert world.act("edit_file", path="toolz/tests/test_itertoolz.py", old="x", new="y")["status"] == "error"
    assert world.act("edit_file", path="toolz/itertoolz.py", old="NO_EXISTE_ESTO", new="y")["status"] == "error"
    # deshacer el bug a mano: aplicar la mutación inversa vía el fuente limpio
    clean = (VENDOR / "toolz/itertoolz.py").read_text()
    mutated = (world.workdir / "toolz/itertoolz.py").read_text()
    import difflib
    diff = [ln for ln in difflib.ndiff(mutated.splitlines(), clean.splitlines()) if ln[:1] in "+-"]
    old = next(ln[2:] for ln in diff if ln.startswith("- "))
    new = next(ln[2:] for ln in diff if ln.startswith("+ "))
    r = world.act("edit_file", path="toolz/itertoolz.py", old=old, new=new)
    assert r["observed_effect"] == "resolves" and world.resolved and world.done
    assert world.score == -20 - 5 - 5 - 5 - 10 + 100  # 4 diag + 2 errores (paso c/u) + edición + resolve


def test_false_finish_and_escalate(tmp_path):
    bug = first_bug("compare_swap")
    w = BugWorld(bug, tmp_path)
    if not w.baseline["failed"]:
        pytest.skip("mutación inerte")
    r = w.act("finish")
    assert r["observed_effect"] == "worsens" and w.done and w.score == -5 - 30
    w2 = BugWorld(first_bug("compare_swap", file="toolz/tests/test_itertoolz.py", bug_kind="broken_test"),
                  tmp_path / "b")
    assert w2.act("escalate_to_human", reason="el test está mal")["observed_effect"] == "resolves"
    assert w2.score == 30 - 5 and w2.resolved
    w3 = BugWorld(bug, tmp_path / "c")
    assert w3.act("escalate_to_human", reason="x")["observed_effect"] == "no_change" and w3.score == -5


def test_rewrite_that_breaks_import_is_worsens(tmp_path):
    w = BugWorld(first_bug("arith_swap"), tmp_path)
    r = w.act("rewrite_file", path="toolz/recipes.py", content="def (broken syntax")
    assert r["observed_effect"] == "worsens" and r["collection_error"] and w.worsens == 1
    assert w.score == -5 - 25 - 40
    r = w.act("revert_file", path="toolz/recipes.py")
    assert r["observed_effect"] in ("improves", "no_change") and not r["collection_error"]
