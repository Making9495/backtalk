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
"""What the voice loop needs from a brain, and nothing vendor-shaped.

main.py talks to this surface only. A brain that can't do something
leaves it out of `capabilities` and the voice console says so out loud,
instead of sending a command the backend can't understand."""
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Protocol

# Capability names. The console checks these before running a verb.
CAP_TOOLS = "tools"              # can act (files, shell, web): gate applies
CAP_COMPACT = "compact"          # "compact the session"
CAP_MODELS = "models"            # "switch to the deep model" / "fast"
CAP_EFFORT = "effort"            # "set effort to X"
CAP_PERMISSIONS = "permissions"  # live permission-mode flip


@dataclass(frozen=True)
class Decision:
    """The spoken gate's answer, backend-neutral. ClaudeBrain turns it
    into the SDK's PermissionResult types; a brain without tools never
    asks."""
    allow: bool
    message: str = ""


# (tool name, tool input, context) -> Decision. The context is whatever
# the backend has; the gate reads display_name / description off it with
# getattr, so any object (or None) works.
Gate = Callable[[str, dict, Any], Awaitable[Decision]]


class Brain(Protocol):
    kind: str                    # "claude", "local", "routed": logs, errors
    model: str                   # what's answering right now
    session: dict                # turns / out_tokens / in_tokens / cost
    capabilities: frozenset[str]

    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    def ask_stream(self, utterance: str) -> AsyncIterator[str]:
        """Yield complete sentences as the reply streams in."""
        ...

    async def interrupt(self) -> None: ...

    async def reset_turn(self, timeout: float = 8.0) -> None:
        """Make the brain safe for a new question after a cancelled
        turn. A no-op when the last turn finished clean."""
        ...

    # Console verbs. Each returns the backend's own reply text ("" when
    # it has nothing to say); text containing "error" or "invalid" is
    # spoken in place of the console's canned confirmation.
    async def clear(self) -> str: ...
    async def compact(self) -> str: ...
    async def deep(self) -> str: ...
    async def fast(self) -> str: ...
    async def set_effort(self, level: str) -> str: ...
    async def set_permission_mode(self, mode: str) -> None: ...

    async def context_usage(self) -> int | None:
        """Tokens currently occupying the context window, or None."""
        ...
