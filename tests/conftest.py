"""Point the config at a throwaway file BEFORE backtalk.config is
imported anywhere: CFG loads at import time, and tests must never read
the developer's real backtalk.json or their agent's CLAUDE.md."""
import json
import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="backtalk-tests-")
_agent = os.path.join(_tmp, "agent")
os.makedirs(_agent)
with open(os.path.join(_agent, "CLAUDE.md"), "w", encoding="utf-8") as f:
    f.write("You are Test Butler.")
with open(os.path.join(_tmp, "backtalk.json"), "w") as f:
    json.dump({"agent_dir": _agent, "signals_dir": _tmp}, f)
os.environ["BACKTALK_CONFIG"] = os.path.join(_tmp, "backtalk.json")
