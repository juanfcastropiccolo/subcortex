"""Mutador AST por posiciones: cambia un token en el código fuente sin tocar el resto.

Cada mutación es un "bug" con clase conocida (la causa oculta del episodio). Se localiza el
nodo con `ast` y se reemplaza solo el segmento de texto correspondiente, así comentarios y
formato quedan intactos y el agente ve código normal.
"""
from __future__ import annotations

import ast
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

VENDOR = Path(__file__).parent / "vendor"
SRC_MODULES = ("toolz/itertoolz.py", "toolz/dicttoolz.py", "toolz/functoolz.py", "toolz/recipes.py")
TEST_MODULES = ("toolz/tests/test_itertoolz.py", "toolz/tests/test_dicttoolz.py",
                "toolz/tests/test_functoolz.py", "toolz/tests/test_recipes.py")
SRC_KINDS = ("arith_swap", "compare_swap", "bool_swap", "negate_condition", "constant_shift", "return_none")
KINDS = SRC_KINDS + ("broken_test",)

_ARITH = {ast.Add: ("+", "-"), ast.Sub: ("-", "+"), ast.Mult: ("*", "//"), ast.FloorDiv: ("//", "*")}
_CMP = {ast.Lt: ("<", "<="), ast.LtE: ("<=", "<"), ast.Gt: (">", ">="), ast.GtE: (">=", ">"),
        ast.Eq: ("==", "!="), ast.NotEq: ("!=", "==")}
_BOOL = {ast.And: ("and", "or"), ast.Or: ("or", "and")}


@dataclass
class Site:
    kind: str
    file: str
    index: int
    lineno: int
    description: str


@dataclass
class Bug:
    id: int
    kind: str          # una de KINDS (broken_test usa una sub-mutación de SRC_KINDS en un test)
    file: str
    site_kind: str
    site_index: int
    description: str
    failing_tests: list[str]


class _Parents(ast.NodeVisitor):
    def __init__(self) -> None:
        self.parent: dict[ast.AST, ast.AST] = {}

    def visit(self, node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            self.parent[child] = node
        super().visit(node)


def _single_line(node: ast.AST) -> bool:
    return getattr(node, "lineno", None) == getattr(node, "end_lineno", None)


def _find_between(lines: list[str], a: ast.AST, b: ast.AST, token: str) -> tuple[int, int] | None:
    """Posición (línea 0-based, col) del token entre el fin de `a` y el inicio de `b` (misma línea)."""
    if a.end_lineno != b.lineno:
        return None
    row = a.end_lineno - 1
    seg = lines[row][a.end_col_offset:b.col_offset]
    j = seg.find(token)
    if j < 0:
        return None
    return row, a.end_col_offset + j


def enumerate_sites(source: str, kind: str, file: str = "") -> list[Site]:
    tree = ast.parse(source)
    parents = _Parents()
    parents.visit(tree)
    sites: list[Site] = []

    def add(node: ast.AST, desc: str) -> None:
        sites.append(Site(kind=kind, file=file, index=len(sites), lineno=node.lineno, description=desc))

    for node in ast.walk(tree):
        if kind == "arith_swap" and isinstance(node, ast.BinOp) and type(node.op) in _ARITH:
            if _find_between(source.splitlines(), node.left, node.right, _ARITH[type(node.op)][0]):
                add(node, f"{_ARITH[type(node.op)][0]} → {_ARITH[type(node.op)][1]} en línea {node.lineno}")
        elif kind == "compare_swap" and isinstance(node, ast.Compare) and len(node.ops) == 1 \
                and type(node.ops[0]) in _CMP:
            if _find_between(source.splitlines(), node.left, node.comparators[0], _CMP[type(node.ops[0])][0]):
                a, b = _CMP[type(node.ops[0])]
                add(node, f"{a} → {b} en línea {node.lineno}")
        elif kind == "bool_swap" and isinstance(node, ast.BoolOp) and len(node.values) >= 2:
            if _find_between(source.splitlines(), node.values[0], node.values[1], _BOOL[type(node.op)][0]):
                a, b = _BOOL[type(node.op)]
                add(node, f"{a} → {b} en línea {node.lineno}")
        elif kind == "negate_condition" and isinstance(node, ast.If) and _single_line(node.test):
            add(node, f"if … → if not (…) en línea {node.lineno}")
        elif kind == "constant_shift" and isinstance(node, ast.Constant) and type(node.value) is int \
                and node.value >= 0 and _single_line(node):
            p = parents.parent.get(node)
            if not isinstance(p, ast.arguments | ast.keyword) and not (
                    isinstance(p, ast.Expr)):  # ni defaults ni constantes sueltas
                add(node, f"{node.value} → {node.value + 1} en línea {node.lineno}")
        elif kind == "return_none" and isinstance(node, ast.Return) and node.value is not None \
                and not isinstance(node.value, ast.Constant) and _single_line(node.value):
            add(node, f"return … → return None en línea {node.lineno}")
    return sites


def apply_mutation(source: str, kind: str, index: int) -> tuple[str, str]:
    """Devuelve (fuente mutada, descripción). Lanza IndexError si no existe el sitio."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    raw = source.splitlines()
    sites = enumerate_sites(source, kind)
    site = sites[index]
    # Reconstruir el nodo objetivo recorriendo en el mismo orden que enumerate_sites.
    target = None
    count = -1
    parents = _Parents()
    parents.visit(tree)
    for node in ast.walk(tree):
        if _matches(node, kind, raw, parents):
            count += 1
            if count == index:
                target = node
                break
    assert target is not None

    def replace(row: int, col: int, old: str, new: str) -> None:
        line = lines[row]
        assert line[col:col + len(old)] == old, (line, col, old)
        lines[row] = line[:col] + new + line[col + len(old):]

    if kind == "arith_swap":
        old, new = _ARITH[type(target.op)]
        row, col = _find_between(raw, target.left, target.right, old)
        replace(row, col, old, new)
    elif kind == "compare_swap":
        old, new = _CMP[type(target.ops[0])]
        row, col = _find_between(raw, target.left, target.comparators[0], old)
        replace(row, col, old, new)
    elif kind == "bool_swap":
        old, new = _BOOL[type(target.op)]
        row, col = _find_between(raw, target.values[0], target.values[1], old)
        replace(row, col, old, new)
    elif kind == "negate_condition":
        t = target.test
        row = t.lineno - 1
        seg = raw[row][t.col_offset:t.end_col_offset]
        replace(row, t.col_offset, seg, f"not ({seg})")
    elif kind == "constant_shift":
        row = target.lineno - 1
        seg = raw[row][target.col_offset:target.end_col_offset]
        replace(row, target.col_offset, seg, str(target.value + 1))
    elif kind == "return_none":
        v = target.value
        row = v.lineno - 1
        seg = raw[row][v.col_offset:v.end_col_offset]
        replace(row, v.col_offset, seg, "None")
    return "".join(lines), site.description


def _matches(node: ast.AST, kind: str, raw: list[str], parents: _Parents) -> bool:
    if kind == "arith_swap":
        return isinstance(node, ast.BinOp) and type(node.op) in _ARITH and bool(
            _find_between(raw, node.left, node.right, _ARITH[type(node.op)][0]))
    if kind == "compare_swap":
        return isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMP and bool(
            _find_between(raw, node.left, node.comparators[0], _CMP[type(node.ops[0])][0]))
    if kind == "bool_swap":
        return isinstance(node, ast.BoolOp) and len(node.values) >= 2 and bool(
            _find_between(raw, node.values[0], node.values[1], _BOOL[type(node.op)][0]))
    if kind == "negate_condition":
        return isinstance(node, ast.If) and _single_line(node.test)
    if kind == "constant_shift":
        if not (isinstance(node, ast.Constant) and type(node.value) is int and node.value >= 0
                and _single_line(node)):
            return False
        p = parents.parent.get(node)
        return not isinstance(p, ast.arguments | ast.keyword) and not isinstance(p, ast.Expr)
    if kind == "return_none":
        return isinstance(node, ast.Return) and node.value is not None \
            and not isinstance(node.value, ast.Constant) and _single_line(node.value)
    return False


# --- generación de bugs -----------------------------------------------------------------------

def copy_template(dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(VENDOR, dst, ignore=shutil.ignore_patterns("__pycache__", "*.dist-info"))


_SUMMARY_PASSED = re.compile(r"(\d+) passed")
_SUMMARY_FAILED = re.compile(r"(\d+) failed")
_SUMMARY_ERRORS = re.compile(r"(\d+) errors?\b")


def run_suite(workdir: Path, pattern: str | None = None, timeout: int = 120) -> dict:
    """Corre la suite vendorizada en `workdir`.

    Devuelve {passed, failed, errors, failing: [ids], output, error: None | "collection" | "timeout"}.
    `error == "collection"` significa SyntaxError/ImportError: el código quedó roto."""
    cmd = [sys.executable, "-m", "pytest", "-q", "-rf", "--color=no", "-p", "no:cacheprovider",
           "--no-header", "--maxfail=50", "toolz/tests"]
    if pattern:
        cmd += ["-k", pattern]
    try:
        r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout,
                           check=False, env={"PYTHONPATH": str(workdir), "PATH": "/usr/bin:/bin",
                                             "PYTHONDONTWRITEBYTECODE": "1"})
    except subprocess.TimeoutExpired:
        return {"passed": 0, "failed": 0, "errors": 0, "failing": [], "output": "TIMEOUT",
                "error": "timeout"}
    out = r.stdout + r.stderr
    failing = [ln.split()[1] for ln in out.splitlines() if ln.startswith("FAILED ")]
    tail = "\n".join(out.splitlines()[-3:])
    passed = int(m.group(1)) if (m := _SUMMARY_PASSED.search(tail)) else 0
    failed = int(m.group(1)) if (m := _SUMMARY_FAILED.search(tail)) else 0
    errors = int(m.group(1)) if (m := _SUMMARY_ERRORS.search(tail)) else 0
    error = None
    if errors or "ERROR collecting" in out or "SyntaxError" in out or "ImportError" in out:
        error = "collection"
    return {"passed": passed, "failed": failed, "errors": errors, "failing": failing,
            "output": out, "error": error}


def generate_bugs(n: int, seed: int, out_path: Path | None = None,
                  min_failing: int = 1, max_failing: int = 8) -> list[Bug]:
    rng = random.Random(seed)
    bugs: list[Bug] = []
    attempts = 0
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "work"
        while len(bugs) < n and attempts < n * 15:
            attempts += 1
            kind = KINDS[len(bugs) % len(KINDS)]
            if kind == "broken_test":
                file = rng.choice(TEST_MODULES)
                site_kind = rng.choice(("compare_swap", "constant_shift", "arith_swap"))
                lo, hi = 1, 3
            else:
                file = rng.choice(SRC_MODULES)
                site_kind = kind
                lo, hi = min_failing, max_failing
            source = (VENDOR / file).read_text()
            sites = enumerate_sites(source, site_kind, file)
            if not sites:
                continue
            site = rng.choice(sites)
            copy_template(work)
            mutated, desc = apply_mutation(source, site_kind, site.index)
            (work / file).write_text(mutated)
            res = run_suite(work)
            if res["error"] or not (lo <= res["failed"] <= hi):
                continue
            bugs.append(Bug(id=len(bugs), kind=kind, file=file, site_kind=site_kind,
                            site_index=site.index, description=desc, failing_tests=res["failing"]))
            print(f"bug {len(bugs) - 1:02d} {kind:16s} {file:32s} {desc:40s} fallan {res['failed']}",
                  flush=True)
    if out_path:
        out_path.write_text(json.dumps([asdict(b) for b in bugs], indent=2, ensure_ascii=False))
    return bugs


def load_bugs(path: Path) -> list[Bug]:
    return [Bug(**b) for b in json.loads(path.read_text())]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).parent / "bugs.json"))
    a = ap.parse_args()
    generate_bugs(a.n, a.seed, Path(a.out))
