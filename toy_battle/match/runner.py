"""Play a single game between two agents, with structured turn logging."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..agents.base import Agent
from ..engine.board import Board
from ..engine.moves import Move, legal_moves
from ..engine.rules import IllegalMove, apply_move
from ..engine.state import GameState, new_game


@dataclass
class GameResult:
    board: str
    winner: Optional[str]
    end_reason: Optional[str]
    turns: int
    medals: dict
    agents: dict  # color -> agent name
    first_player: str
    seed: Optional[int]
    error: Optional[str] = None
    turn_log: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "board": self.board,
            "winner": self.winner,
            "end_reason": self.end_reason,
            "turns": self.turns,
            "medals": self.medals,
            "agents": self.agents,
            "first_player": self.first_player,
            "seed": self.seed,
            "error": self.error,
        }


def play_game(
    board: Board,
    agents: dict[str, Agent],
    *,
    first_player: Optional[str] = None,
    seed: Optional[int] = None,
    max_turns: int = 1000,
    log_path: Optional[str | Path] = None,
    verbose: bool = False,
) -> GameResult:
    """Play one game. ``agents`` maps colour -> Agent."""
    state = new_game(board, first_player=first_player, seed=seed)
    actual_first = state.current
    for a in agents.values():
        a.on_game_start(state)

    log_fh = open(log_path, "w", encoding="utf-8") if log_path else None
    error = None
    turn_log = []

    def write(record: dict) -> None:
        turn_log.append(record)
        if log_fh:
            log_fh.write(json.dumps(record) + "\n")
            log_fh.flush()

    write({"event": "start", "board": board.name, "first_player": state.current,
           "seed": seed, "agents": {c: a.name for c, a in agents.items()}})

    try:
        while state.winner is None and state.turn_count < max_turns:
            color = state.current
            agent = agents[color]
            if not legal_moves(state):
                # Engine resolves stuck on turn advance; this is a safety net.
                break
            t0 = time.time()
            move: Move = agent.choose_move(state)
            reasoning = ""
            if hasattr(move, "move"):  # AgentDecision-like
                reasoning = getattr(move, "reasoning", "")
                move = move.move
            try:
                apply_move(state, move, agent.decide)
            except IllegalMove as exc:
                # Should be rare; agents validate. Count and fall back to a legal move.
                agent.stats.illegal_attempts += 1
                fallback = legal_moves(state)[0]
                apply_move(state, fallback, agent.decide)
                move = fallback
                reasoning = f"[illegal: {exc}] fell back"
            dt = time.time() - t0
            rec = {
                "event": "move",
                "turn": state.turn_count,
                "color": color,
                "agent": agent.name,
                "move": move.describe(),
                "target": move.target,
                "reasoning": reasoning,
                "medals": {c: p.medals for c, p in state.players.items()},
                "seconds": round(dt, 3),
            }
            write(rec)
            if verbose:
                print(f"T{state.turn_count} {color}/{agent.name}: {move.describe()} "
                      f"| medals {rec['medals']}")
    except KeyboardInterrupt:
        error = "interrupted"
    except Exception as exc:  # pragma: no cover - defensive
        error = f"{type(exc).__name__}: {exc}"

    result = GameResult(
        board=board.name,
        winner=state.winner,
        end_reason=state.end_reason,
        turns=state.turn_count,
        medals={c: p.medals for c, p in state.players.items()},
        agents={c: a.name for c, a in agents.items()},
        first_player=actual_first,
        seed=seed,
        error=error,
        turn_log=turn_log,
    )
    write({"event": "end", **result.to_dict()})
    if log_fh:
        log_fh.close()
    for a in agents.values():
        a.on_game_end(state)
    return result
