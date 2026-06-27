"""Provider-agnostic LLM agent.

Handles the whole decision pipeline; per-provider subclasses only implement
``_complete`` (one chat/tool call). The pipeline:

1. Build a system prompt (rules) + a user message describing the state.
   - tool mode: full structured text, no image (tests tool use + understanding).
   - vision mode: a rendered PNG + minimal text, withholding the per-base
     occupancy text (tests OCR / board reading).
2. Always include a numbered list of legal moves and a ``submit_move`` tool.
3. Parse the tool call -> a legal index. On parse/illegal failure, retry with
   feedback up to ``max_retries``; otherwise fall back to a heuristic legal move
   (counted as a fallback).
4. Capture tokens / cost / latency telemetry.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from ...engine.moves import Decision, Move, legal_moves
from ...engine.state import GameState
from ...engine.troops import TROOPS
from ...render.image import render_state
from ...render.text import build_observation, observation_to_text, legal_moves_view
from ..base import Agent
from .prompts import SYSTEM_PROMPT

# Approximate USD per 1M tokens (input, output). Unknown models default to 0.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "gpt-5": (5.0, 15.0),
    "gpt-4o": (2.5, 10.0),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.30, 2.50),
}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema


@dataclass
class ContentPart:
    kind: str  # "text" | "image"
    text: Optional[str] = None
    image_path: Optional[str] = None


@dataclass
class LLMResponse:
    tool_name: Optional[str]
    arguments: dict
    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: object = None


SUBMIT_MOVE = Tool(
    name="submit_move",
    description="Submit your chosen move by its index in the legal-moves list.",
    parameters={
        "type": "object",
        "properties": {
            "index": {"type": "integer", "description": "Index of the chosen legal move."},
            "reasoning": {"type": "string", "description": "Brief reason for the choice."},
        },
        "required": ["index"],
    },
)

SUBMIT_MOVE_VISUAL = Tool(
    name="submit_move",
    description=(
        "Submit the move you read from the board image, in game terms. "
        "Use action='draw' to draw tiles, or action='place' with the troop from "
        "your rack and the target base id labelled on the board."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["draw", "place"]},
            "troop": {
                "type": "string",
                "description": "For 'place': the troop from your rack (name like 'Roxy'/'XB-42', or its force number).",
            },
            "target": {
                "type": "string",
                "description": "For 'place': the base id exactly as labelled under the tile on the board (e.g. 'b_tl', 'br_c', 'hq_blue').",
            },
            "reasoning": {"type": "string", "description": "What you read from the board and why."},
        },
        "required": ["action"],
    },
)

SUBMIT_CHOICE = Tool(
    name="submit_choice",
    description="Submit your choice index for the current effect decision, or -1 to skip.",
    parameters={
        "type": "object",
        "properties": {
            "index": {"type": "integer", "description": "Index of the chosen option, or -1 to skip."},
            "reasoning": {"type": "string"},
        },
        "required": ["index"],
    },
)


class LLMAgent(Agent):
    provider = "llm"

    def __init__(
        self,
        color: str,
        model: Optional[str] = None,
        mode: str = "tool",
        *,
        api_key: Optional[str] = None,
        max_retries: int = 2,
        image_dir: str = "renders/llm_frames",
        temperature: float = 0.2,
        name: Optional[str] = None,
        **kwargs,
    ):
        self.model = model or self.default_model()
        self.mode = mode  # "tool" | "vision"
        super().__init__(color, name=name or f"{self.provider}:{self.model}:{mode}", **kwargs)
        self.api_key = api_key or os.environ.get(self.api_key_env())
        self.max_retries = max_retries
        self.image_dir = image_dir
        self.temperature = temperature

    # -- provider hooks (override) -----------------------------------------
    def default_model(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def api_key_env(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def _complete(
        self, system: str, content: list[ContentPart], tools: list[Tool], force_tool: Optional[str]
    ) -> LLMResponse:  # pragma: no cover - overridden / network
        raise NotImplementedError

    # -- telemetry ---------------------------------------------------------
    def _account(self, resp: LLMResponse, dt: float) -> None:
        self.stats.prompt_tokens += resp.prompt_tokens
        self.stats.completion_tokens += resp.completion_tokens
        self.stats.latency_s += dt
        pin, pout = PRICES.get(self.model, (0.0, 0.0))
        self.stats.cost_usd += (resp.prompt_tokens * pin + resp.completion_tokens * pout) / 1_000_000

    # -- move selection ----------------------------------------------------
    def choose_move(self, state: GameState) -> Move:
        self.stats.moves += 1
        if self.mode == "vision":
            return self._choose_visual(state)
        return self._choose_indexed(state)

    def _render_frame(self, state: GameState) -> str:
        os.makedirs(self.image_dir, exist_ok=True)
        path = os.path.join(self.image_dir, f"{self.color}_t{state.turn_count:03d}.png")
        render_state(state, path)
        return path

    # tool mode: full structured text + a numbered legal-move list + index pick.
    def _choose_indexed(self, state: GameState) -> Move:
        obs = build_observation(state, self.color)
        legal = legal_moves(state)
        base_content = [ContentPart("text", text=observation_to_text(obs))]
        feedback = ""
        for _ in range(self.max_retries + 1):
            content = list(base_content)
            instruction = (
                "Call submit_move with the index of your chosen move from the legal "
                f"list (0..{len(legal) - 1})."
            )
            if feedback:
                instruction = feedback + "\n" + instruction
            content.append(ContentPart("text", text=instruction))
            try:
                t0 = time.time()
                resp = self._complete(SYSTEM_PROMPT, content, [SUBMIT_MOVE], "submit_move")
                self._account(resp, time.time() - t0)
            except Exception as exc:
                self.stats.parse_errors += 1
                feedback = f"(previous attempt errored: {exc})"
                continue
            idx = _extract_index(resp)
            if idx is None:
                self.stats.parse_errors += 1
                feedback = "Your last response did not include a valid integer index."
                continue
            if not (0 <= idx < len(legal)):
                self.stats.illegal_attempts += 1
                feedback = f"Index {idx} is out of range; choose 0..{len(legal) - 1}."
                continue
            return legal[idx]
        self.stats.fallbacks += 1
        return self._fallback_move(state, legal)

    # vision mode: board IMAGE only (no occupancy text, no legal-move list). The
    # model must read the board and name its move in game terms (troop + base id),
    # which we validate against the rules.
    def _choose_visual(self, state: GameState) -> Move:
        obs = build_observation(state, self.color)
        legal = legal_moves(state)
        path = self._render_frame(state)
        base_brief = _vision_brief(obs)
        feedback = ""
        for _ in range(self.max_retries + 1):
            content = [
                ContentPart("text", text=base_brief if not feedback else feedback + "\n\n" + base_brief),
                ContentPart("image", image_path=path),
            ]
            try:
                t0 = time.time()
                resp = self._complete(SYSTEM_PROMPT, content, [SUBMIT_MOVE_VISUAL], "submit_move")
                self._account(resp, time.time() - t0)
            except Exception as exc:
                self.stats.parse_errors += 1
                feedback = f"(previous attempt errored: {exc})"
                continue
            move, reason = _resolve_visual_move(state, self.color, resp.arguments, legal)
            if move is None:
                self.stats.illegal_attempts += 1
                feedback = f"That move was not legal: {reason} Re-read the board image and try again."
                continue
            return move
        self.stats.fallbacks += 1
        return self._fallback_move(state, legal)

    def _fallback_move(self, state: GameState, legal: list[Move]) -> Move:
        from ..heuristic_agent import HeuristicAgent

        helper = HeuristicAgent(self.color)
        try:
            return helper.choose_move(state)
        except Exception:
            return legal[0]

    # -- in-effect decisions ----------------------------------------------
    def decide(self, state: GameState, decision: Decision) -> Optional[dict]:
        self.stats.decisions += 1
        lines = [f"Effect decision: {decision.prompt}", "Options:"]
        for i, o in enumerate(decision.options):
            lines.append(f"  [{i}] {o.get('label', json.dumps(o))}")
        if decision.optional:
            lines.append("You may skip by submitting index -1.")
        content = [ContentPart("text", text="\n".join(lines))]
        try:
            t0 = time.time()
            resp = self._complete(SYSTEM_PROMPT, content, [SUBMIT_CHOICE], "submit_choice")
            self._account(resp, time.time() - t0)
        except Exception:
            return None if decision.optional else (decision.options[0] if decision.options else None)
        idx = _extract_index(resp)
        if idx is None:
            return None if decision.optional else decision.options[0]
        if idx == -1:
            return None if decision.optional else decision.options[0]
        if 0 <= idx < len(decision.options):
            return decision.options[idx]
        return None if decision.optional else decision.options[0]


def _extract_index(resp: LLMResponse) -> Optional[int]:
    if resp.arguments and "index" in resp.arguments:
        try:
            return int(resp.arguments["index"])
        except (TypeError, ValueError):
            return None
    # Fallback: try to find an integer in free text.
    if resp.text:
        import re

        m = re.search(r"-?\d+", resp.text)
        if m:
            return int(m.group())
    return None


def _vision_brief(obs: dict) -> str:
    """Vision-mode prompt: NO board occupancy text and NO legal-move list — the
    model must read everything from the image. Only the player's own hidden
    information (their rack) and headline scores are given as text."""
    b = obs["board"]
    rack = ", ".join(f"{t['name']}(f{t['force']})" for t in obs["your_rack"]) or "(empty)"
    foe_last = obs["opponent"].get("last_action") or "(none yet)"
    return "\n".join(
        [
            f"Toy Battle — {b['display_name']}. You are {obs['you'].upper()}.",
            f"Medal objective: {b['medal_objective']}. "
            f"Your medals: {obs['your_medals']}, opponent: {obs['opponent']['medals']}.",
            f"Your rack (your hand, not on the board): {rack}.",
            f"Your reserve: {obs['your_reserve_count']} tiles.",
            f"Opponent's last action: {foe_last}.",
            "",
            "The ONLY view of the board is the attached image. Read it yourself: each "
            "base shows its top tile (troop code + force) coloured by owner, a small "
            "grey id under it (e.g. b_tl, br_c, s_ul, hq_blue), gold circles are the "
            "medals still available in each region, diamonds are the HQs.",
            "Decide your move and call submit_move: either action='draw', or "
            "action='place' with a troop from your rack and the target base id you read "
            "off the board. It must satisfy the rules (force/connection/region).",
        ]
    )


def _resolve_troop_id(value) -> Optional[str]:
    """Map a free-form troop reference (name, id, short code, or force number) to
    a troop id."""
    if value is None:
        return None
    s = str(value).strip().lower()
    for tid, t in TROOPS.items():
        if s in (tid, t.name.lower(), t.code.lower()):
            return tid
        if not t.is_joker and s == str(t.force):
            return tid
    # Tolerate things like "xb42" vs "xb-42".
    s2 = s.replace("-", "").replace("'", "").replace(" ", "")
    for tid, t in TROOPS.items():
        if s2 == t.name.lower().replace("-", "").replace("'", "").replace(" ", ""):
            return tid
    return None


def _resolve_visual_move(state: GameState, color: str, args: dict, legal):
    """Map a visual {action, troop, target} submission to a legal Move.

    Returns (move, reason). ``move`` is None when nothing legal matches; ``reason``
    is a short, crutch-free explanation for the retry feedback.
    """
    from ...engine.moves import MoveKind

    action = str(args.get("action", "")).strip().lower()
    if action == "draw":
        draw = next((m for m in legal if m.kind == MoveKind.DRAW), None)
        return (draw, "" if draw else "you cannot draw right now (rack full or reserve empty).")
    if action != "place":
        return (None, f"unknown action '{action}'; use 'draw' or 'place'.")

    target = str(args.get("target", "")).strip()
    if target not in state.board.nodes:
        return (None, f"'{target}' is not a base id on this board.")
    troop = _resolve_troop_id(args.get("troop"))
    rack = state.players[color].rack
    place_here = [m for m in legal if m.kind == MoveKind.PLACE and m.target == target]
    if not place_here:
        return (None, f"you have no legal placement on '{target}'.")
    if troop is None:
        return (place_here[0], "")
    for m in place_here:
        tile = next((t for t in rack if t.uid == m.tile_uid), None)
        if tile and tile.troop_id == troop:
            return (m, "")
    return (None, f"placing {TROOPS[troop].name} on '{target}' is not legal.")
