"""Moves, in-effect decisions, and legal-move generation.

A *turn* is one top-level action: DRAW or PLACE. Some troop/base effects then
require sub-decisions (e.g. Cap'taine placing an extra troop, Mastok choosing a
target). Those are resolved through the agent's ``decide`` callback against a
``Decision`` whose options are all engine-validated, so an agent can never pick
an illegal sub-action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .board import HQ
from .connectivity import is_connected
from .state import RACK_MAX, GameState
from .troops import TROOPS, can_cover


class MoveKind(str, Enum):
    DRAW = "draw"
    PLACE = "place"


@dataclass(frozen=True)
class Move:
    kind: MoveKind
    tile_uid: int | None = None  # for PLACE
    target: str | None = None  # node id, for PLACE

    def describe(self, state: GameState | None = None) -> str:
        if self.kind == MoveKind.DRAW:
            return "DRAW tiles from reserve"
        troop_name = ""
        if state is not None:
            p = state.players[state.current]
            tile = next((t for t in p.rack if t.uid == self.tile_uid), None)
            if tile:
                troop_name = f" {tile.name}(force {TROOPS[tile.troop_id].force_label})"
        return f"PLACE{troop_name} on {self.target}"


@dataclass
class Decision:
    """A sub-decision requested by an effect during a placement.

    ``options`` is a list of opaque option payloads (dicts) the agent chooses
    among by index. If ``optional`` is True, returning ``None`` skips the effect.
    """

    kind: str
    prompt: str
    options: list[dict]
    optional: bool = True
    context: dict = field(default_factory=dict)


def can_draw(state: GameState, color: str) -> bool:
    p = state.players[color]
    return len(p.rack) < RACK_MAX and len(p.reserve) > 0


def draw_count(state: GameState, color: str, requested: int) -> int:
    """How many tiles would actually be drawn for a request of ``requested``."""
    p = state.players[color]
    slots = RACK_MAX - len(p.rack)
    return max(0, min(requested, slots, len(p.reserve)))


def placement_allowed(
    state: GameState, color: str, troop_id: str, target: str, *, ignore_connection: bool = False
) -> bool:
    """Whether ``color`` may place a tile of ``troop_id`` on ``target`` right now.

    ``ignore_connection`` is set for the Crochet effect (which ignores the
    connection rule for bases, but not for capturing an enemy HQ).
    """
    board = state.board
    node = board.nodes.get(target)
    if node is None:
        return False

    if node.kind == HQ:
        # Can only target an *enemy* HQ, never your own.
        if node.owner == color:
            return False
    else:
        # Value restriction (e.g. piscine des tropiques) — checked before placing.
        if node.allowed_forces is not None:
            t = TROOPS[troop_id]
            if not t.is_joker and t.force not in node.allowed_forces:
                return False

    # Stacking rules against the current top tile.
    top = state.top_tile(target)
    if top is not None:
        if top.color == color:
            pass  # may always reinforce your own base
        else:
            if not can_cover(troop_id, top.troop_id):
                return False

    # Connection rule.
    if node.kind == HQ:
        # Capturing an enemy HQ always requires connection (even for Crochet).
        if not is_connected(state, color, target):
            return False
    else:
        if not ignore_connection and not is_connected(state, color, target):
            return False
    return True


def placeable_targets(state: GameState, color: str, troop_id: str) -> list[str]:
    ignore = TROOPS[troop_id].effect == "ignore_connection"
    out = []
    for nid, node in state.board.nodes.items():
        if node.kind == HQ and node.owner == color:
            continue
        if placement_allowed(state, color, troop_id, nid, ignore_connection=ignore):
            out.append(nid)
    return out


def legal_moves(state: GameState) -> list[Move]:
    """All legal top-level moves for the current player."""
    if state.winner is not None:
        return []
    color = state.current
    p = state.players[color]
    moves: list[Move] = []

    if can_draw(state, color):
        moves.append(Move(MoveKind.DRAW))

    # Deduplicate identical (troop_id, target) placements; keep one representative
    # tile uid so the action space stays compact.
    seen: set[tuple[str, str]] = set()
    for tile in p.rack:
        for target in placeable_targets(state, color, tile.troop_id):
            key = (tile.troop_id, target)
            if key in seen:
                continue
            seen.add(key)
            moves.append(Move(MoveKind.PLACE, tile_uid=tile.uid, target=target))
    return moves
