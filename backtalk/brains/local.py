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
"""The local brain: a model on your own machine, over any
OpenAI-compatible chat endpoint (Ollama, llama.cpp's server, vLLM,
LM Studio).

TALK ONLY, ON PURPOSE. This brain has no tools: it can't touch files,
run commands or fetch pages, so the spoken permission gate never fires
and a voice in the room can't make it do anything but answer. It's for
quick conversation at zero cost and full privacy; "routed" mode hands
real work to Claude.

Same identity as the Claude brain: the system prompt is your agent's
CLAUDE.md (from agent_dir) plus the spoken-delivery discipline.
"""
import json
import os

import httpx

from backtalk.brains._sentences import SentenceBuffer, ThinkFilter
from backtalk.brains.base import CAP_COMPACT, CAP_MODELS
from backtalk.config import CFG, DISCIPLINE
from backtalk.vlog import log

# A whole CLAUDE.md can be long, and every character is prompt the local
# model re-reads on a cold cache. Enough for an identity, not a manual.
_PERSONA_MAX_CHARS = 12_000

_COMPACT_ASK = ("Summarize our conversation so far in one short paragraph: "
                "the facts, decisions and open threads worth keeping. "
                "Plain prose, no preamble.")

# Said instead of going silent when the server drops mid-conversation.
UNREACHABLE = ("I can't reach the local model right now. Check that its "
               "server is running.")


def persona_prompt(agent_dir: str) -> str:
    try:
        with open(os.path.join(agent_dir, "CLAUDE.md"),
                  encoding="utf-8") as f:
            persona = f.read(_PERSONA_MAX_CHARS).strip()
    except OSError:
        persona = ""
    parts = [persona] if persona else []
    parts.append(DISCIPLINE)
    parts.append("You are the quick local voice: you can talk, but you "
                 "can't open files, run anything or look things up. When "
                 "asked to do one of those, say so in a sentence and tell "
                 "the person to say 'think harder' to hand it to the "
                 "full agent.")
    return "\n\n".join(parts)


class LocalBrain:
    kind = "local"

    def __init__(self, cfg: dict | None = None, model: str | None = None,
                 session: dict | None = None,
                 transport: httpx.AsyncBaseTransport | None = None):
        self._cfg = cfg if cfg is not None else CFG["local"]
        self.model = model or self._cfg["model"]
        self._fast_model = self.model
        self._deep_model = str(self._cfg.get("deep_model") or "")
        self.capabilities = frozenset(
            {CAP_COMPACT} | ({CAP_MODELS} if self._deep_model else set()))
        self.session = session if session is not None else {
            "turns": 0, "out_tokens": 0, "in_tokens": 0, "cost": 0.0}
        self._transport = transport          # tests inject a fake server
        self._http: httpx.AsyncClient | None = None
        self._system = ""
        self.messages: list[dict] = []       # the conversation, no system
        self._last_context: int | None = None

    # ---- lifecycle

    async def start(self):
        headers = {}
        if self._cfg.get("api_key"):
            headers["Authorization"] = f"Bearer {self._cfg['api_key']}"
        self._http = httpx.AsyncClient(
            base_url=str(self._cfg["base_url"]).rstrip("/"),
            headers=headers, transport=self._transport,
            timeout=httpx.Timeout(float(self._cfg.get("timeout_s", 60)),
                                  connect=5.0))
        self._system = persona_prompt(CFG["agent_dir"])
        # Fail the launch loudly here rather than on the first question:
        # main.py turns a start() failure into a spoken explanation.
        try:
            r = await self._http.get("/models")
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise RuntimeError(
                f"local model server not reachable at "
                f"{self._cfg['base_url']}: {e!r}") from e
        log(f"[brain] local model {self.model} at {self._cfg['base_url']}")

    async def stop(self):
        if self._http:
            await self._http.aclose()
            self._http = None

    # ---- the turn

    def _payload(self, messages: list[dict], stream: bool) -> dict:
        body = {"model": self.model,
                "messages": [{"role": "system", "content": self._system}]
                + messages,
                "stream": stream}
        if stream:
            # usage on the final chunk; servers that don't know the option
            # ignore it and the tally simply stays at zero
            body["stream_options"] = {"include_usage": True}
        body.update(self._cfg.get("extra_body") or {})
        return body

    def _tally(self, usage: dict | None, count_turn=True):
        u = usage or {}
        if count_turn:
            self.session["turns"] += 1
        self.session["out_tokens"] += int(u.get("completion_tokens") or 0)
        self.session["in_tokens"] += int(u.get("prompt_tokens") or 0)
        if u:
            self._last_context = (int(u.get("prompt_tokens") or 0)
                                  + int(u.get("completion_tokens") or 0))

    def _trim(self):
        cap = int(self._cfg.get("max_history", 40))
        if cap > 0 and len(self.messages) > cap:
            self.messages = self.messages[-cap:]
            # never open the history on a dangling assistant line
            while self.messages and self.messages[0]["role"] != "user":
                self.messages.pop(0)

    def note_exchange(self, user: str, reply: str):
        """Record a turn another brain answered (routed mode), so a
        follow-up asked here still knows what was said."""
        self.messages += [{"role": "user", "content": user},
                          {"role": "assistant", "content": reply}]
        self._trim()

    async def ask_stream(self, utterance: str):
        """Yield complete sentences as they stream out of the model.

        Cancellation (the person talks over the reply) lands at one of
        the awaits below; the finally still records the turn, cut off
        where it was, so the next question has the right context."""
        sent = SentenceBuffer()
        think = ThinkFilter() if self._cfg.get("strip_think", True) else None
        reply: list[str] = []
        usage = None
        done = False
        failed = False
        try:
            async with self._http.stream(
                    "POST", "/chat/completions",
                    json=self._payload(
                        self.messages + [{"role": "user",
                                          "content": utterance}], True)
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices") or []:
                        text = (choice.get("delta") or {}).get("content")
                        if not text:
                            continue
                        if think:
                            text = think.feed(text)
                        reply.append(text)
                        for s in sent.feed(text):
                            yield s
            if think:
                tail = think.flush()
                reply.append(tail)
                for s in sent.feed(tail):
                    yield s
            done = True
        except httpx.HTTPError as e:
            # Mid-conversation the voice must never just go quiet: say
            # what happened, and keep the question out of the history.
            log(f"[brain] local model request failed: {e!r}")
            failed = True
        finally:
            if not failed:
                text = "".join(reply).strip()
                if not done:
                    text = (text + " [interrupted]").strip()
                self.messages += [{"role": "user", "content": utterance},
                                  {"role": "assistant", "content": text}]
                self._trim()
                self._tally(usage)
        if failed:
            yield UNREACHABLE
            return
        tail = sent.flush()
        if tail:
            yield tail

    async def interrupt(self):
        # The stream lives inside ask_stream; cancelling the task that
        # consumes it closes the connection, which is how llama.cpp and
        # Ollama stop generating. Nothing extra to send.
        pass

    async def reset_turn(self, timeout: float = 8.0):
        # No shared message pipe, so no off-by-one to repair (see
        # ClaudeBrain.reset_turn): every request carries its own reply.
        pass

    # ---- console verbs

    async def clear(self) -> str:
        self.messages = []
        self._last_context = None
        return ""

    async def compact(self) -> str:
        if not self.messages:
            return ""
        try:
            r = await self._http.post("/chat/completions", json=self._payload(
                self.messages + [{"role": "user", "content": _COMPACT_ASK}],
                False))
            r.raise_for_status()
            body = r.json()
            summary = body["choices"][0]["message"]["content"] or ""
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            log(f"[brain] local compact failed: {e!r}")
            return "error: the local model couldn't compact the session"
        if self._cfg.get("strip_think", True):
            f = ThinkFilter()
            summary = f.feed(summary) + f.flush()
        self._tally(body.get("usage"), count_turn=False)
        self.messages = [
            {"role": "user",
             "content": "Summary of our conversation so far: "
                        + summary.strip()},
            {"role": "assistant", "content": "Got it."}]
        return ""

    async def deep(self) -> str:
        self.model = self._deep_model or self.model
        return ""

    async def fast(self) -> str:
        self.model = self._fast_model
        return ""

    async def set_effort(self, level: str) -> str:
        return "error: effort isn't supported on the local model"

    async def set_permission_mode(self, mode: str) -> None:
        pass    # nothing to permit: this brain has no tools

    async def context_usage(self) -> int | None:
        return self._last_context
