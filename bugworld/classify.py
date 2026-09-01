"""Clasificador heurístico de fallos de pytest → `finding` (lo que el diagnóstico descubre).

Trabaja sobre la salida, nunca sobre la causa: es lo que un operador humano infiere al leer
el primer fallo. Imperfecto a propósito.
"""
from __future__ import annotations

import ast
import re

FINDINGS = ("none_result", "exception", "bool_flip", "order", "numeric_mismatch", "boundary", "unknown")

_ASSERT_EQ = re.compile(r"^E\s+assert (.+?) == (.+)$", re.MULTILINE)
_EXC = re.compile(r"^E\s+([A-Za-z]+Error|StopIteration|KeyError|IndexError|ValueError|TypeError)\b", re.MULTILINE)


def _literal(s: str):
    try:
        return ast.literal_eval(s.strip())
    except Exception:  # noqa: BLE001 — cualquier literal no evaluable cuenta como texto
        return None


def classify_failure(output: str) -> str:
    if "NoneType" in output or re.search(r"^E\s+assert .*\bNone\b", output, re.MULTILINE):
        return "none_result"
    exc = _EXC.search(output)
    if exc and exc.group(1) != "AssertionError":
        return "exception"
    if re.search(r"^E\s+assert (not )?(True|False)\b", output, re.MULTILINE) or \
            re.search(r"^E\s+assert .* (is|==) (True|False)\b", output, re.MULTILINE):
        return "bool_flip"
    m = _ASSERT_EQ.search(output)
    if m:
        a, b = _literal(m.group(1)), _literal(m.group(2))
        if isinstance(a, bool) or isinstance(b, bool):
            return "bool_flip"
        if isinstance(a, int | float) and isinstance(b, int | float):
            return "numeric_mismatch"
        if isinstance(a, list | tuple) and isinstance(b, list | tuple):
            if len(a) == len(b):
                try:
                    if sorted(a) == sorted(b):
                        return "order"
                except TypeError:
                    pass
                return "numeric_mismatch" if all(isinstance(x, int | float) for x in list(a) + list(b)) \
                    else "unknown"
            return "boundary"
        if isinstance(a, dict | set) and isinstance(b, dict | set):
            return "boundary" if len(a) != len(b) else "unknown"
    if "assert" in output:
        return "unknown"
    return "unknown"
