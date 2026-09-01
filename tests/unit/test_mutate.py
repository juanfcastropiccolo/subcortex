import ast

import pytest

from bugworld.mutate import SRC_KINDS, SRC_MODULES, VENDOR, apply_mutation, enumerate_sites

SAMPLE = '''
def f(a, b, n=3):
    """doc 5"""
    if a and b:  # comentario
        return a + b
    x = [1, 2, 3]
    if len(x) < 3:
        return x[0]
    return None
'''


@pytest.mark.parametrize("kind", SRC_KINDS)
def test_each_kind_produces_valid_and_different_code(kind):
    sites = enumerate_sites(SAMPLE, kind)
    assert sites, kind
    mutated, desc = apply_mutation(SAMPLE, kind, 0)
    ast.parse(mutated)
    assert mutated != SAMPLE and desc
    assert "# comentario" in mutated and '"""doc 5"""' in mutated  # formato intacto


def test_specific_mutations():
    assert "a - b" in apply_mutation(SAMPLE, "arith_swap", 0)[0]
    assert "a or b" in apply_mutation(SAMPLE, "bool_swap", 0)[0]
    assert "if not (a and b):" in apply_mutation(SAMPLE, "negate_condition", 0)[0]
    assert "len(x) <= 3" in apply_mutation(SAMPLE, "compare_swap", 0)[0]
    assert "return None" in apply_mutation(SAMPLE, "return_none", 0)[0].splitlines()[4]
    shifted = apply_mutation(SAMPLE, "constant_shift", 0)[0]
    assert "n=3" in shifted and "[2, 2, 3]" in shifted  # no toca defaults; sí el primer literal


def test_sites_are_deterministic_and_indexed():
    a = enumerate_sites(SAMPLE, "constant_shift")
    b = enumerate_sites(SAMPLE, "constant_shift")
    assert [s.lineno for s in a] == [s.lineno for s in b]
    assert [s.index for s in a] == list(range(len(a)))
    with pytest.raises(IndexError):
        apply_mutation(SAMPLE, "arith_swap", 99)


@pytest.mark.parametrize("module", SRC_MODULES)
def test_vendor_modules_have_sites_and_mutations_parse(module):
    src = (VENDOR / module).read_text()
    kinds_with_sites = 0
    for kind in SRC_KINDS:
        sites = enumerate_sites(src, kind, module)
        if not sites:  # recipes.py es diminuto: no tiene aritmética
            continue
        kinds_with_sites += 1
        mutated, _ = apply_mutation(src, kind, sites[len(sites) // 2].index)
        ast.parse(mutated)
        assert mutated != src
    assert kinds_with_sites >= 3, module
