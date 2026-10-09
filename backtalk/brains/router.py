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
"""Routed mode: the local model answers by default, Claude does the work.

A turn goes to Claude when any of these is true:
  - the console said "switch to the deep model" (sticky, until "back to
    the fast model");
  - the utterance starts with an escalate phrase ("think harder ...");
  - the utterance contains an escalate word (file, run, email ...).

Plain rules, no classifier, ON PURPOSE: a router that guesses adds
latency to every turn and fails in ways nobody can predict. These fail
in ways you can read in the config. Every turn logs which brain took it.

Each brain sees the other's turns. Claude gets the recent local turns as
a short brief at the top of its next question (it keeps its own
history); the local model gets Claude's exchanges appended to its
history, so "and what about tomorrow?" works whichever one answers."""
import re

from backtalk.brains.base import (CAP_COMPACT, CAP_EFFORT, CAP_MODELS,
                                  CAP_PERMISSIONS, CAP_TOOLS)
from backtalk.vlog import log


class RoutedBrain:
    kind = "routed"
    capabilities = frozenset({CAP_TOOLS, CAP_COMPACT, CAP_MODELS,
                              CAP_EFFORT, CAP_PERMISSIONS})

    def __init__(self, local, claude, cfg: dict):
        self.local, self.claude = local, claude
        # one usage report for the whole conversation
        self.session = {"turns": 0, "out_tokens": 0, "in_tokens": 0,
                        "cost": 0.0}
        local.session = claude.session = self.session
        self._phrases = tuple(_norm(p) for p in cfg.get("escalate_phrases")
                              or ())
        words = [w.strip().lower() for w in cfg.get("escalate_words") or ()
                 if w.strip()]
        self._words = (re.compile(r"\b(" + "|".join(map(re.escape, words))
                                  + r")\b") if words else None)
        self._handoff_turns = int(cfg.get("handoff_turns", 6))
        self._sticky_claude = False
        self._active = local
        # local exchanges Claude hasn't heard yet: [(user, reply), ...]
        self._unseen_by_claude: list[tuple[str, str]] = []

    @property
    def model(self) -> str:
        return f"{self.local.model} (routes to {self.claude.model})"

    def route(self, utterance: str):
        """Which brain takes this turn, and why (for the log)."""
        if self._sticky_claude:
            return self.claude, "deep mode"
        norm = _norm(utterance)
        for p in self._phrases:
            if norm.startswith(p):
                return self.claude, f"phrase '{p}'"
        if self._words:
            m = self._words.search(norm)
            if m:
                return self.claude, f"word '{m.group(1)}'"
        return self.local, "default"

    def _brief(self, utterance: str) -> str:
        recent = self._unseen_by_claude[-self._handoff_turns:]
        if not recent:
            return utterance
        lines = []
        for user, reply in recent:
            lines += [f"Person: {user}", f"You (quick local voice): {reply}"]
        return ("[Earlier in this voice conversation, answered by your "
                "quick local model. Context only, don't repeat it:\n"
                + "\n".join(lines) + "]\n\n" + utterance)

    async def start(self):
        # Claude first: it's the slow, fallible connect, and main.py
        # says out loud which part failed.
        await self.claude.start()
        await self.local.start()

    async def stop(self):
        for b in (self.local, self.claude):
            try:
                await b.stop()
            except Exception:
                pass

    async def ask_stream(self, utterance: str):
        brain, why = self.route(utterance)
        self._active = brain
        log(f"[route]  {brain.kind} ({why})")
        question = self._brief(utterance) if brain is self.claude \
            else utterance
        said: list[str] = []
        completed = False
        try:
            async for sentence in brain.ask_stream(question):
                said.append(sentence)
                yield sentence
            completed = True
        finally:
            reply = " ".join(said) + ("" if completed else " [interrupted]")
            if brain is self.claude:
                self._unseen_by_claude.clear()
                self.local.note_exchange(utterance, reply.strip())
            else:
                self._unseen_by_claude.append((utterance, reply.strip()))
                del self._unseen_by_claude[:-self._handoff_turns]

    async def interrupt(self):
        await self._active.interrupt()

    async def reset_turn(self, timeout: float = 8.0):
        await self.claude.reset_turn(timeout)
        await self.local.reset_turn(timeout)

    async def clear(self) -> str:
        self._unseen_by_claude.clear()
        await self.local.clear()
        return await self.claude.clear()

    async def compact(self) -> str:
        resp = await self.local.compact()
        return (await self.claude.compact()) or resp

    async def deep(self) -> str:
        self._sticky_claude = True
        return ""

    async def fast(self) -> str:
        self._sticky_claude = False
        return ""

    async def set_effort(self, level: str) -> str:
        return await self.claude.set_effort(level)

    async def set_permission_mode(self, mode: str) -> None:
        await self.claude.set_permission_mode(mode)

    async def context_usage(self) -> int | None:
        return await self._active.context_usage()


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split())
