"""Connection-to-HQ and region-control logic."""

from __future__ import annotations

from .board import Board
from .state import GameState


def routable_nodes(state: GameState, color: str) -> set[str]:
    """The set of nodes a player can route *through*: their HQ plus every base
    they currently occupy that is reachable from the HQ via occupied bases.

    An empty base, or a base topped by an enemy tile, breaks the chain.
    """
    board = state.board
    start = board.hq_of(color)
    seen = {start}
    frontier = [start]
    while frontier:
        node = frontier.pop()
        for nb in board.neighbors(node):
            if nb in seen:
                continue
            if state.occupies(color, nb):
                seen.add(nb)
                frontier.append(nb)
    return seen


def is_connected(state: GameState, color: str, target: str) -> bool:
    """Whether ``target`` can be legally reached: it must be adjacent to the HQ
    or to some occupied base that itself chains back to the HQ."""
    board = state.board
    reach = routable_nodes(state, color)
    if target in reach:
        # Already occupied by you and connected (reinforcing your own base).
        return True
    neighbors = board.neighbors(target)
    return any(n in reach for n in neighbors)


def newly_controlled_regions(state: GameState, color: str) -> list[str]:
    """Region ids whose every border base is now occupied by ``color`` and that
    have not yet been claimed (medals still on the board)."""
    out = []
    for region in state.board.regions:
        if region.id in state.claimed_regions:
            continue
        if all(state.occupies(color, nid) for nid in region.border):
            out.append(region.id)
    return out
