"""Uniformly random legal agent — the baseline opponent."""

from __future__ import annotations

import random
from typing import Optional

from ..engine.moves import Decision, Move, legal_moves
from ..engine.state import GameState
from .base import Agent


class RandomAgent(Agent):
    def __init__(self, color: str, seed: Optional[int] = None, **kwargs):
        super().__init__(color, name=kwargs.pop("name", "random"), **kwargs)
        self.rng = random.Random(seed)

    def choose_move(self, state: GameState) -> Move:
        self.stats.moves += 1
        return self.rng.choice(legal_moves(state))

    def decide(self, state: GameState, decision: Decision) -> Optional[dict]:
        self.stats.decisions += 1
        if decision.optional and self.rng.random() < 0.5:
            return None
        return self.rng.choice(decision.options) if decision.options else None
