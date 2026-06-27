"""Agent interface.

An agent makes two kinds of decisions:
- ``choose_move``: pick a top-level move (DRAW or PLACE) for the current turn.
- ``decide``: resolve an in-effect sub-decision (Cap'taine extra place, Mastok
  target, plaine return, ...). The engine only ever offers pre-validated options,
  so an agent can never produce an illegal sub-action.

Agents also expose lightweight telemetry (``stats``) that the benchmark harness
aggregates: illegal-move attempts, fallbacks, tokens, cost, latency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..engine.moves import Decision, Move, legal_moves
from ..engine.state import GameState


@dataclass
class AgentStats:
    moves: int = 0
    decisions: int = 0
    illegal_attempts: int = 0
    parse_errors: int = 0
    fallbacks: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0

    def merge(self, other: "AgentStats") -> None:
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(self, f) + getattr(other, f))


@dataclass
class AgentDecision:
    """Optional structured result an agent may return from choose_move, carrying
    reasoning for logging. Plain ``Move`` returns are also accepted."""

    move: Move
    reasoning: str = ""


class Agent:
    """Base agent. Subclass and implement ``choose_move`` (and optionally
    ``decide``). ``name`` is used in logs/leaderboards."""

    def __init__(self, color: str, name: Optional[str] = None, **_: object):
        self.color = color
        self.name = name or self.__class__.__name__
        self.stats = AgentStats()

    # -- required ----------------------------------------------------------
    def choose_move(self, state: GameState) -> Move:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- optional ----------------------------------------------------------
    def decide(self, state: GameState, decision: Decision) -> Optional[dict]:
        """Default: skip optional effects, take the first mandatory option."""
        if decision.optional:
            return None
        return decision.options[0] if decision.options else None

    # -- lifecycle hooks ---------------------------------------------------
    def on_game_start(self, state: GameState) -> None:
        pass

    def on_game_end(self, state: GameState) -> None:
        pass

    def __repr__(self) -> str:
        return f"<{self.name} ({self.color})>"


def choose_legal_or_raise(state: GameState, move: Move) -> Move:
    """Validate that ``move`` is among the current legal moves (by troop+target),
    returning a normalised move. Raises ``ValueError`` if not legal."""
    legal = legal_moves(state)
    from ..engine.moves import MoveKind

    if move.kind == MoveKind.DRAW:
        if any(m.kind == MoveKind.DRAW for m in legal):
            return move
        raise ValueError("DRAW is not legal")
    # Map to a legal place by (troop, target).
    p = state.players[state.current]
    troop = next((t.troop_id for t in p.rack if t.uid == move.tile_uid), None)
    for m in legal:
        if m.kind != MoveKind.PLACE or m.target != move.target:
            continue
        cand = next((t.troop_id for t in p.rack if t.uid == m.tile_uid), None)
        if troop is None or cand == troop:
            return m
    raise ValueError(f"PLACE on {move.target} is not legal")
