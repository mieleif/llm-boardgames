"""A simple greedy 1-ply heuristic agent.

For each legal move it clones the state, applies the move (resolving its own
sub-decisions greedily), and scores the result from its own perspective. This
gives a non-trivial opponent that values winning, capturing medals, and board
presence — a useful skill baseline above ``random``.
"""

from __future__ import annotations

import random
from typing import Optional

from ..engine.moves import Decision, Move, legal_moves
from ..engine.rules import apply_move
from ..engine.state import GameState
from ..engine.troops import TROOPS
from .base import Agent

WIN = 10_000_000


class HeuristicAgent(Agent):
    def __init__(self, color: str, seed: Optional[int] = None, **kwargs):
        super().__init__(color, name=kwargs.pop("name", "heuristic"), **kwargs)
        self.rng = random.Random(seed)

    def _score(self, state: GameState) -> float:
        me, opp = self.color, state.opponent(self.color)
        if state.winner == me:
            return WIN
        if state.winner == opp:
            return -WIN
        score = 0.0
        score += 1000.0 * (state.players[me].medals - state.players[opp].medals)
        score += 8.0 * (len(state.on_board_nodes(me)) - len(state.on_board_nodes(opp)))
        # Light preference to keep options open (tiles available to play).
        score += 1.0 * len(state.players[me].rack)
        return score

    def choose_move(self, state: GameState) -> Move:
        self.stats.moves += 1
        moves = legal_moves(state)
        best, best_score = [], float("-inf")
        for mv in moves:
            sim = state.clone()
            try:
                apply_move(sim, mv, self._sim_decider)
            except Exception:
                continue
            s = self._score(sim)
            if s > best_score:
                best_score, best = s, [mv]
            elif s == best_score:
                best.append(mv)
        if not best:
            return self.rng.choice(moves)
        return self.rng.choice(best)

    # Greedy decisions while simulating / actually playing.
    def _sim_decider(self, state: GameState, decision: Decision) -> Optional[dict]:
        return self.decide(state, decision)

    def decide(self, state: GameState, decision: Decision) -> Optional[dict]:
        self.stats.decisions += 1
        opts = decision.options
        if not opts:
            return None
        if decision.kind == "mastok_target":
            # Discard the highest-force visible enemy troop.
            return max(opts, key=lambda o: _force(o["troop"]))
        if decision.kind == "captaine_extra":
            # Take any free extra placement (more board presence is good).
            return opts[0]
        if decision.kind == "plaine_return":
            return None  # keep board presence by default
        return None if decision.optional else opts[0]


def _force(troop_id: str) -> int:
    f = TROOPS[troop_id].force
    return f if f is not None else 0
