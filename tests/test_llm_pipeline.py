"""Validate the provider-agnostic LLM pipeline without any network calls.

A fake provider implements ``_complete`` to exercise tool-call parsing, illegal
handling, retry-with-feedback, and the heuristic fallback.
"""

from toy_battle.agents.llm.base_llm import ContentPart, LLMAgent, LLMResponse, Tool
from toy_battle.engine import legal_moves, load_board, new_game
from toy_battle.engine.moves import MoveKind


class FakeLLM(LLMAgent):
    provider = "fake"

    def __init__(self, color, scripted_indices, **kw):
        self._script = list(scripted_indices)
        self._calls = 0
        super().__init__(color, model="fake-1", mode="tool", api_key="x", **kw)

    def default_model(self):
        return "fake-1"

    def api_key_env(self):
        return "FAKE_KEY"

    def _complete(self, system, content, tools, force_tool):
        idx = self._script[min(self._calls, len(self._script) - 1)]
        self._calls += 1
        return LLMResponse(
            tool_name=tools[0].name,
            arguments={"index": idx, "reasoning": "test"},
            prompt_tokens=10,
            completion_tokens=5,
        )


def _state():
    return new_game(load_board("plaine_des_chateaux"), first_player="red", seed=3)


def test_valid_tool_call_selects_legal_move():
    state = _state()
    agent = FakeLLM("red", scripted_indices=[0])
    move = agent.choose_move(state)
    assert move in legal_moves(state)
    assert agent.stats.prompt_tokens == 10
    assert agent.stats.completion_tokens == 5


def test_out_of_range_then_valid_retry():
    state = _state()
    n = len(legal_moves(state))
    # First an out-of-range index (counts as illegal), then a valid one.
    agent = FakeLLM("red", scripted_indices=[999, 1], max_retries=2)
    move = agent.choose_move(state)
    assert move in legal_moves(state)
    assert agent.stats.illegal_attempts == 1
    assert agent.stats.fallbacks == 0
    assert n >= 2


def test_persistent_illegal_falls_back_to_legal():
    state = _state()
    agent = FakeLLM("red", scripted_indices=[999], max_retries=1)
    move = agent.choose_move(state)
    assert move in legal_moves(state)
    assert agent.stats.fallbacks == 1
    assert agent.stats.illegal_attempts >= 1


def test_pricing_accumulates_cost():
    state = _state()
    agent = FakeLLM("red", scripted_indices=[0])
    agent.model = "claude-opus-4-8"  # known price
    agent.choose_move(state)
    assert agent.stats.cost_usd > 0
