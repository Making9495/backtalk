"""The Claude-specific glue that can be tested without a signed-in CLI."""
from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

from backtalk import brain as shim
from backtalk.brains import Decision, make_brain
from backtalk.brains.claude import ClaudeBrain, occupied_tokens, sdk_gate
from backtalk.config import CFG


async def test_gate_adapter_maps_decisions_to_sdk_results():
    seen = []

    async def gate(tool, tool_input, ctx):
        seen.append((tool, tool_input))
        return Decision(allow=tool == "Read", message="no thanks")

    can_use_tool = sdk_gate(gate)
    ok = await can_use_tool("Read", {"file_path": "x"}, None)
    no = await can_use_tool("Bash", {"command": "rm"}, None)
    assert isinstance(ok, PermissionResultAllow)
    assert isinstance(no, PermissionResultDeny)
    assert no.message == "no thanks" and no.interrupt is False
    assert seen == [("Read", {"file_path": "x"}), ("Bash", {"command": "rm"})]
    assert sdk_gate(None) is None


def test_occupied_tokens_skips_free_space_and_buffer():
    usage = {"categories": [{"name": "System prompt", "tokens": 3000},
                            {"name": "Messages", "tokens": 1000},
                            {"name": "Free space", "tokens": 90000},
                            {"name": "Autocompact buffer", "tokens": 40000}]}
    assert occupied_tokens(usage) == 4000
    assert occupied_tokens(None) is None


def test_old_import_path_still_works():
    assert shim.WarmBrain is ClaudeBrain


def test_factory_picks_brain_from_config(monkeypatch):
    monkeypatch.setitem(CFG, "brain", "claude")
    assert make_brain().kind == "claude"
    monkeypatch.setitem(CFG, "brain", "local")
    assert make_brain().kind == "local"
    monkeypatch.setitem(CFG, "brain", "routed")
    b = make_brain(model="claude-x")
    assert b.kind == "routed" and b.claude.model == "claude-x"
    monkeypatch.setitem(CFG, "brain", "nonsense")
    try:
        make_brain()
    except ValueError as e:
        assert "nonsense" in str(e)
    else:
        raise AssertionError("unknown brain accepted")
