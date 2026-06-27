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
    def _state_message(self, state: GameState) -> list[ContentPart]:
        obs = build_observation(state, self.color)
        if self.mode == "vision":
            os.makedirs(self.image_dir, exist_ok=True)
            path = os.path.join(self.image_dir, f"{self.color}_t{state.turn_count:03d}.png")
            render_state(state, path)
            # Minimal text: withhold per-base occupancy to force board reading.
            brief = _vision_brief(obs)
            return [ContentPart("text", text=brief), ContentPart("image", image_path=path)]
        # tool mode: full structured text.
        return [ContentPart("text", text=observation_to_text(obs))]

    def choose_move(self, state: GameState) -> Move:
        self.stats.moves += 1
        legal = legal_moves(state)
        base_content = self._state_message(state)
        feedback = ""
        for attempt in range(self.max_retries + 1):
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
        # Exhausted retries: fall back to a heuristic legal move.
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
    b = obs["board"]
    rack = ", ".join(f"{t['name']}(f{t['force']})" for t in obs["your_rack"]) or "(empty)"
    foe_last = obs["opponent"].get("last_action") or "(none yet)"
    lines = [
        f"Toy Battle — {b['display_name']}. You are {obs['you'].upper()}.",
        f"Objective: {b['medal_objective']} medals. "
        f"Your medals: {obs['your_medals']}, opponent: {obs['opponent']['medals']}.",
        f"Your rack: {rack}.",
        f"Opponent's last action: {foe_last}.",
        "Read the attached board image to see which troops occupy which bases, "
        "their forces, and remaining region medals. Then pick from the legal moves:",
    ]
    for m in obs.get("legal_moves", []):
        lines.append(f"  [{m['index']}] {m['description']}")
    return "\n".join(lines)
