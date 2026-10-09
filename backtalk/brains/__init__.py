# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pick the brain the config asks for (key "brain"):

  "claude"  the Claude Code session, tools and all (the default)
  "local"   a model on this machine, talk only (brains/local.py)
  "routed"  local by default, Claude for real work (brains/router.py)

Backends are imported lazily, so a local-only setup never needs a
signed-in Claude Code just to boot."""
from backtalk.brains.base import Brain, Decision, Gate  # noqa: F401
from backtalk.config import CFG

KINDS = ("claude", "local", "routed")


def make_brain(model: str | None = None, gate: Gate | None = None,
               resume_id: str | None = None) -> Brain:
    """model is the --model launch override; it applies to the brain the
    session leads with (Claude in "claude" and "routed", the local model
    in "local")."""
    kind = str(CFG.get("brain") or "claude").lower()
    if kind not in KINDS:
        raise ValueError(f'unknown brain {kind!r} in config, expected one '
                         f'of {", ".join(KINDS)}')
    if kind == "local":
        from backtalk.brains.local import LocalBrain
        return LocalBrain(model=model)
    from backtalk.brains.claude import ClaudeBrain
    claude = ClaudeBrain(model=model, gate=gate, resume_id=resume_id)
    if kind == "claude":
        return claude
    from backtalk.brains.local import LocalBrain
    from backtalk.brains.router import RoutedBrain
    return RoutedBrain(LocalBrain(), claude, CFG["router"])
