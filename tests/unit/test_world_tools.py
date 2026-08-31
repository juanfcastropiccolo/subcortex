from types import SimpleNamespace

from opsworld.tools import ALL_TOOLS, registry, restart, inspect_service
from opsworld.world import Incident, World


def ctx(session_id="s1"):
    return SimpleNamespace(session=SimpleNamespace(id=session_id), state={})


def test_tools_route_to_session_world():
    registry.clear()
    w = World(Incident(id=0, service="api", cause="memory_leak", recent_deploy=False,
                       traffic="normal", hour_bucket="night"))
    registry.register("s1", w)
    assert inspect_service("api", ctx())["observed_effect"] == "diagnostic"
    assert restart("api", ctx())["observed_effect"] == "resolves"
    assert w.steps == 2


def test_unknown_session_returns_error():
    registry.clear()
    assert restart("api", ctx("nope"))["status"] == "error"


def test_all_tools_have_docstrings_and_names():
    names = {t.__name__ for t in ALL_TOOLS}
    assert {"restart", "rollback", "scale", "failover_db", "resolve",
            "escalate_to_human", "inspect_service", "check_deploys"} == names
    assert all(t.__doc__ for t in ALL_TOOLS)
