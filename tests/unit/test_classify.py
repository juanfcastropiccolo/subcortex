from bugworld.classify import classify_failure


def test_findings_from_representative_outputs():
    assert classify_failure("E       TypeError: 'NoneType' object is not iterable") == "none_result"
    assert classify_failure("E       assert None == [1, 2]") == "none_result"
    assert classify_failure("E       KeyError: 'a'") == "exception"
    assert classify_failure("E       assert False\nE        +  where False = f(1)") == "bool_flip"
    assert classify_failure("E       assert 3 == 4") == "numeric_mismatch"
    assert classify_failure("E       assert [1, 2] == [1, 2, 3]") == "boundary"
    assert classify_failure("E       assert [2, 1] == [1, 2]") == "order"
    assert classify_failure("E       assert [1, 5] == [1, 2]") == "numeric_mismatch"
    assert classify_failure("E       assert 'ab' == 'abc'") == "unknown"
    assert classify_failure("collected 0 items") == "unknown"
