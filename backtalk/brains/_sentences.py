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
"""Streamed text in, complete sentences out. Shared by every brain so
the mouth gets the same chunking whichever model is talking.

Lifted unchanged from WarmBrain.ask_stream: a sentence ends at . ! or ?
followed by whitespace. That splits "Mr. Smith" in two, and always has;
the TTS doesn't care, and a smarter rule is a behavior change for the
Claude path that should be made on purpose, not in a refactor."""
import re

SENTENCE_END = re.compile(r"(?<=[.!?])\s")


class SentenceBuffer:
    def __init__(self):
        self._buf = ""

    def feed(self, text: str) -> list[str]:
        """Add streamed text; return every sentence it completed."""
        self._buf += text
        out = []
        while True:
            m = SENTENCE_END.search(self._buf)
            if not m:
                break
            sentence, self._buf = (self._buf[:m.end()].strip(),
                                   self._buf[m.end():])
            if sentence:
                out.append(sentence)
        return out

    def flush(self) -> str | None:
        """Whatever is left, as one final chunk (None when empty). Called
        at the end of a speech block, e.g. right before a tool call."""
        tail, self._buf = self._buf.strip(), ""
        return tail or None


class ThinkFilter:
    """Drops <think>...</think> spans from a streamed reply. Reasoning
    models served locally (Qwen3, DeepSeek-R1 and kin) often put their
    scratch work inline in the content stream; read aloud it is minutes
    of mumbling before the answer. Tags can straddle chunk boundaries,
    so a possible partial tag is held back until the next chunk."""
    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self._inside = False
        self._held = ""

    def feed(self, text: str) -> str:
        s, out = self._held + text, []
        self._held = ""
        while s:
            tag = self.CLOSE if self._inside else self.OPEN
            i = s.find(tag)
            if i >= 0:
                if not self._inside:
                    out.append(s[:i])
                s = s[i + len(tag):]
                self._inside = not self._inside
                continue
            # no full tag: hold back a tail that could be the start of one
            keep = 0
            for n in range(min(len(tag) - 1, len(s)), 0, -1):
                if tag.startswith(s[-n:]):
                    keep = n
                    break
            body, self._held = s[:len(s) - keep], s[len(s) - keep:]
            if not self._inside:
                out.append(body)
            break
        return "".join(out)

    def flush(self) -> str:
        held, self._held = self._held, ""
        return "" if self._inside else held
