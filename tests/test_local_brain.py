"""LocalBrain against a fake OpenAI-compatible server (httpx.MockTransport):
no Ollama needed, and the failure cases are reproducible."""
import asyncio
import json

import httpx
import pytest

from backtalk.brains.base import CAP_EFFORT, CAP_MODELS
from backtalk.brains.local import UNREACHABLE, LocalBrain

CFG = {"base_url": "http://fake/v1", "model": "small", "deep_model": "",
       "api_key": "k", "max_history": 40, "timeout_s": 5,
       "strip_think": True, "extra_body": {"temperature": 0.1}}


def sse(*deltas, usage=None):
    lines = [json.dumps({"choices": [{"delta": {"content": d}}]})
             for d in deltas]
    if usage:
        lines.append(json.dumps({"choices": [], "usage": usage}))
    return "".join(f"data: {l}\n\n" for l in lines) + "data: [DONE]\n\n"


class FakeServer:
    """Records requests; answers chat with queued SSE bodies."""

    def __init__(self):
        self.requests = []
        self.replies = []        # str (SSE body), dict (JSON), or Exception
        self.down = False

    async def __call__(self, request: httpx.Request):
        if self.down:
            raise httpx.ConnectError("refused", request=request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "small"}]})
        self.requests.append(json.loads(request.content))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, dict):
            return httpx.Response(200, json=reply)
        if callable(reply):
            return httpx.Response(200, content=reply())
        return httpx.Response(200, text=reply)


async def make(cfg=CFG):
    server = FakeServer()
    b = LocalBrain(cfg=dict(cfg), transport=httpx.MockTransport(server))
    await b.start()
    return b, server


async def collect(brain, text):
    return [s async for s in brain.ask_stream(text)]


async def test_start_fails_loudly_when_server_down():
    server = FakeServer()
    server.down = True
    b = LocalBrain(cfg=dict(CFG), transport=httpx.MockTransport(server))
    with pytest.raises(RuntimeError, match="local model server"):
        await b.start()


async def test_streams_sentences_and_records_history():
    b, server = await make()
    server.replies.append(sse("Hello there", ". How can I ", "help?"))
    assert await collect(b, "hi") == ["Hello there.", "How can I help?"]
    assert b.messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello there. How can I help?"}]
    req = server.requests[0]
    assert req["model"] == "small" and req["stream"] is True
    assert req["temperature"] == 0.1                       # extra_body
    assert req["messages"][0]["role"] == "system"
    assert "Test Butler" in req["messages"][0]["content"]  # CLAUDE.md
    assert req["messages"][-1] == {"role": "user", "content": "hi"}


async def test_history_is_sent_on_the_next_turn():
    b, server = await make()
    server.replies += [sse("One. "), sse("Two. ")]
    await collect(b, "first")
    await collect(b, "second")
    roles = [m["role"] for m in server.requests[1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


async def test_think_blocks_are_never_spoken():
    b, server = await make()
    server.replies.append(sse("<think>plan the", " answer</think>", "Four."))
    assert await collect(b, "two plus two") == ["Four."]
    assert b.messages[-1]["content"] == "Four."


async def test_usage_tallies_and_feeds_context_usage():
    b, server = await make()
    server.replies.append(sse("Hi. ", usage={"prompt_tokens": 120,
                                             "completion_tokens": 3}))
    await collect(b, "hi")
    assert b.session["turns"] == 1
    assert b.session["in_tokens"] == 120
    assert b.session["out_tokens"] == 3
    assert await b.context_usage() == 123


async def test_server_dropping_mid_conversation_is_spoken_not_silent():
    b, server = await make()
    server.replies.append(httpx.ReadTimeout("slow"))
    assert await collect(b, "are you there") == [UNREACHABLE]
    assert b.messages == []          # the failed question isn't kept


async def test_cancelled_turn_is_kept_as_interrupted():
    b, server = await make()
    gate = asyncio.Event()

    async def slow():
        yield b'data: {"choices":[{"delta":{"content":"Let me think. "}}]}\n\n'
        await gate.wait()            # never set: the reply stalls here
        yield b"data: [DONE]\n\n"
    server.replies.append(slow)

    got = []

    async def consume():
        async for s in b.ask_stream("tell me a story"):
            got.append(s)
    task = asyncio.create_task(consume())
    while not got:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert got == ["Let me think."]
    assert b.messages[-1] == {"role": "assistant",
                              "content": "Let me think. [interrupted]"}
    await b.reset_turn()             # nothing to repair; must not raise


async def test_clear_forgets_everything():
    b, server = await make()
    server.replies.append(sse("Hi. "))
    await collect(b, "hi")
    assert await b.clear() == ""
    assert b.messages == [] and await b.context_usage() is None


async def test_compact_replaces_history_with_summary():
    b, server = await make()
    server.replies += [sse("Hi. "), {
        "choices": [{"message": {"content":
                                 "<think>x</think>We said hi."}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 5}}]
    await collect(b, "hi")
    assert await b.compact() == ""
    assert b.messages[0]["content"].endswith("We said hi.")
    assert len(b.messages) == 2
    assert server.requests[1]["stream"] is False


async def test_compact_failure_is_reported_as_error_text():
    b, server = await make()
    server.replies += [sse("Hi. "), httpx.ConnectError("gone")]
    await collect(b, "hi")
    assert "error" in await b.compact()
    assert len(b.messages) == 2      # history untouched


async def test_history_trim_never_starts_on_assistant():
    b, _ = await make(dict(CFG, max_history=3))
    for i in range(3):
        b.note_exchange(f"q{i}", f"a{i}")
    assert b.messages[0]["role"] == "user"
    assert len(b.messages) <= 3


async def test_capabilities_and_model_switching():
    b, _ = await make()
    assert CAP_MODELS not in b.capabilities
    assert CAP_EFFORT not in b.capabilities
    assert "error" in await b.set_effort("high")

    b, _ = await make(dict(CFG, deep_model="big"))
    assert CAP_MODELS in b.capabilities
    await b.deep()
    assert b.model == "big"
    await b.fast()
    assert b.model == "small"
