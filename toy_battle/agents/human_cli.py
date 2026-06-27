"""Human agent: renders the board (text + optional PNG) and reads typed input."""

from __future__ import annotations

import os
from typing import Optional

from ..engine.moves import Decision, Move, MoveKind, legal_moves
from ..engine.state import GameState
from ..render.text import build_observation, observation_to_text
from .base import Agent


class HumanCliAgent(Agent):
    def __init__(self, color: str, render_image: bool = True, image_dir: str = ".", **kwargs):
        super().__init__(color, name=kwargs.pop("name", "human"), **kwargs)
        self.render_image = render_image
        self.image_dir = image_dir
        self._turn = 0

    def _maybe_render(self, state: GameState) -> None:
        if not self.render_image:
            return
        try:
            from ..render.image import render_state

            os.makedirs(self.image_dir, exist_ok=True)
            path = os.path.join(self.image_dir, f"board_turn_{state.turn_count:03d}.png")
            render_state(state, path)
            print(f"[board image: {path}]")
        except Exception as exc:  # rendering is best-effort for humans
            print(f"[image render unavailable: {exc}]")

    def choose_move(self, state: GameState) -> Move:
        self.stats.moves += 1
        obs = build_observation(state, self.color)
        print("\n" + observation_to_text(obs))
        self._maybe_render(state)
        moves = legal_moves(state)
        while True:
            raw = input(f"\n{self.color}, choose a move index [0-{len(moves) - 1}] (or 'q'): ").strip()
            if raw.lower() in ("q", "quit", "exit"):
                raise KeyboardInterrupt
            if raw.isdigit() and 0 <= int(raw) < len(moves):
                return moves[int(raw)]
            print("Invalid index, try again.")

    def decide(self, state: GameState, decision: Decision) -> Optional[dict]:
        self.stats.decisions += 1
        print(f"\n>> {decision.prompt}")
        for i, o in enumerate(decision.options):
            print(f"   [{i}] {o.get('label', o)}")
        skip_hint = " or 's' to skip" if decision.optional else ""
        while True:
            raw = input(f"Choose [0-{len(decision.options) - 1}]{skip_hint}: ").strip()
            if decision.optional and raw.lower() in ("s", "skip", ""):
                return None
            if raw.isdigit() and 0 <= int(raw) < len(decision.options):
                return decision.options[int(raw)]
            print("Invalid choice.")
