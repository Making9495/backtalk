from backtalk.brains.router import RoutedBrain

RULES = {"escalate_phrases": ["think harder", "ask claude"],
         "escalate_words": ["file", "look up"],
         "handoff_turns": 2}


class FakeBrain:
    def __init__(self, kind, reply):
        self.kind, self.model, self.reply = kind, kind + "-model", reply
        self.session = {}
        self.asked, self.noted, self.calls = [], [], []

    async def ask_stream(self, text):
        self.asked.append(text)
        for s in self.reply:
            yield s

    def note_exchange(self, user, reply):
        self.noted.append((user, reply))

    async def interrupt(self):
        self.calls.append("interrupt")

    async def reset_turn(self, timeout=8.0):
        self.calls.append("reset")

    async def clear(self):
        self.calls.append("clear")
        return ""

    async def set_effort(self, level):
        self.calls.append(f"effort {level}")
        return ""

    async def context_usage(self):
        return {"local": 10, "claude": 20}[self.kind]


def make():
    local = FakeBrain("local", ["Sure.", "Easy one."])
    claude = FakeBrain("claude", ["Done."])
    return RoutedBrain(local, claude, RULES), local, claude


async def say(router, text):
    return [s async for s in router.ask_stream(text)]


def test_routing_rules():
    r, local, claude = make()
    assert r.route("how are you")[0] is local
    assert r.route("Think harder: why is the sky blue")[0] is claude
    assert r.route("can you open that FILE")[0] is claude
    assert r.route("please look up the weather")[0] is claude
    # whole words only: "profile" is not "file"
    assert r.route("nice profile picture")[0] is local
    # a phrase must START the utterance
    assert r.route("i never ask claude anything")[0] is local


async def test_sticky_deep_mode():
    r, local, claude = make()
    await r.deep()
    assert r.route("hello")[0] is claude
    await r.fast()
    assert r.route("hello")[0] is local


async def test_claude_is_briefed_on_recent_local_turns_once():
    r, local, claude = make()
    await say(r, "one")
    await say(r, "two")
    await say(r, "three")                 # only 2 kept (handoff_turns)
    assert await say(r, "think harder about it") == ["Done."]
    brief = claude.asked[0]
    assert "Person: two" in brief and "Person: three" in brief
    assert "Person: one" not in brief
    assert brief.endswith("think harder about it")
    await say(r, "ask claude again")
    assert claude.asked[1] == "ask claude again"   # nothing new to brief


async def test_local_hears_what_claude_said():
    r, local, claude = make()
    await say(r, "open the file")
    assert local.noted == [("open the file", "Done.")]
    await say(r, "thanks")
    assert local.asked == ["thanks"]


async def test_interrupt_goes_to_whoever_is_talking():
    r, local, claude = make()
    await say(r, "open the file")
    await r.interrupt()
    assert claude.calls == ["interrupt"] and local.calls == []


async def test_shared_session_and_forwarding():
    r, local, claude = make()
    assert local.session is claude.session is r.session
    await r.reset_turn()
    assert local.calls == ["reset"] and claude.calls == ["reset"]
    await r.set_effort("high")
    assert claude.calls[-1] == "effort high"
    await r.clear()
    assert "clear" in local.calls and "clear" in claude.calls
    await say(r, "hi")
    assert await r.context_usage() == 10
