"""Rule resolution: applying moves, effect ordering, region scoring, win/end."""

from __future__ import annotations

from typing import Optional

from .board import HQ, SPECIAL_BASE
from .connectivity import newly_controlled_regions
from .effects import (
    BASE_EFFECTS,
    TROOP_EFFECTS,
    EffectContext,
    default_decider,
    effects_suppressed,
)
from .moves import Decision, Move, MoveKind, draw_count, legal_moves
from .state import GameState, Tile
from .troops import TROOPS


class IllegalMove(Exception):
    pass


def apply_move(state: GameState, move: Move, decider=None) -> GameState:
    """Apply a legal top-level move, resolving all effects, scoring and win
    conditions. Mutates and returns ``state``."""
    if state.winner is not None:
        raise IllegalMove("game is already over")
    decider = decider or default_decider
    color = state.current

    move = _validate(state, move)

    pre_desc = move.describe(state)  # while the tile is still on the rack
    medals_before = state.players[color].medals
    hist_start = len(state.history)

    if move.kind == MoveKind.DRAW:
        _do_draw(state, color)
    else:
        _do_place_action(state, color, move, decider)

    _record_last_action(state, color, pre_desc, medals_before, hist_start)

    state.turn_count += 1
    if state.winner is None:
        state.current = state.opponent(color)
        _maybe_end_stuck(state)
    return state


# Keywords used to surface the notable consequences of a move in its summary.
_EFFECT_KEYWORDS = (
    "controls region",
    "captures",
    "discards",
    "returns",
    "draws",
    "places extra",
)


def _record_last_action(
    state: GameState, color: str, pre_desc: str, medals_before: int, hist_start: int
) -> None:
    """Store a one-line, opponent-visible summary of what ``color`` just did."""
    summary = pre_desc
    delta = state.players[color].medals - medals_before
    extras = [
        ln.strip()
        for ln in state.history[hist_start:]
        if any(k in ln for k in _EFFECT_KEYWORDS)
    ]
    if delta > 0:
        summary += f"  (+{delta} medals)"
    if extras:
        summary += " | " + "; ".join(extras[:3])
    state.last_moves[color] = summary
    state.last_action_by = color
    state.last_action_summary = summary


def _validate(state: GameState, move: Move) -> Move:
    """Check the move is legal; normalise the tile uid to an interchangeable
    copy actually present in the current player's rack."""
    color = state.current
    legal = legal_moves(state)
    if move.kind == MoveKind.DRAW:
        if not any(m.kind == MoveKind.DRAW for m in legal):
            raise IllegalMove("cannot draw")
        return move

    p = state.players[color]
    tile = next((t for t in p.rack if t.uid == move.tile_uid), None)
    troop_id = tile.troop_id if tile else None
    if troop_id is None:
        # Allow an agent to reference a copy by troop via any uid that maps to a
        # legal (troop, target) pair.
        for m in legal:
            if m.kind == MoveKind.PLACE and m.target == move.target:
                cand = next((t for t in p.rack if t.uid == m.tile_uid), None)
                if cand:
                    troop_id = cand.troop_id
                    move = Move(MoveKind.PLACE, tile_uid=m.tile_uid, target=move.target)
                    tile = cand
                    break
    if tile is None:
        raise IllegalMove(f"no such tile uid {move.tile_uid} in {color} rack")
    if not any(
        m.kind == MoveKind.PLACE
        and m.target == move.target
        and state_tile_troop(state, color, m.tile_uid) == troop_id
        for m in legal
    ):
        raise IllegalMove(move.describe(state) + " is not legal")
    return move


def state_tile_troop(state: GameState, color: str, uid: int) -> Optional[str]:
    t = next((t for t in state.players[color].rack if t.uid == uid), None)
    return t.troop_id if t else None


def _do_draw(state: GameState, color: str) -> None:
    n = draw_count(state, color, 2)
    p = state.players[color]
    for _ in range(n):
        p.rack.append(p.reserve.pop())
    state.log(
        f"{color} draws {n} tile(s) (rack {len(p.rack)}/8, reserve {len(p.reserve)})"
    )


def _do_place_action(state: GameState, color: str, move: Move, decider) -> None:
    p = state.players[color]
    tile = p.take_from_rack(move.tile_uid)
    placements: list[tuple[str, Tile]] = []
    _place_tile(state, color, tile, move.target, decider, placements)
    if state.winner is not None:
        return

    # Apply special-base effects of all placed troops, in placement order, after
    # all troop effects have resolved (rulebook ordering).
    for node_id, _t in placements:
        node = state.board.nodes[node_id]
        if node.kind == SPECIAL_BASE and node.effect and not effects_suppressed(node):
            handler = BASE_EFFECTS.get(node.effect)
            if handler:
                handler(_make_ctx(state, color, node_id, decider, placements))
                if state.winner is not None:
                    return

    _score_regions(state, color)
    _check_medal_win(state, color)


def _place_tile(
    state: GameState,
    color: str,
    tile: Tile,
    target: str,
    decider,
    placements: list[tuple[str, Tile]],
) -> None:
    node = state.board.nodes[target]
    state.push(target, tile)
    placements.append((target, tile))
    state.log(f"{color} places {tile.name} (force {TROOPS[tile.troop_id].force_label}) on {target}")

    # Capturing an enemy HQ ends the game immediately.
    if node.kind == HQ and node.owner != color:
        state.winner = color
        state.end_reason = "hq_captured"
        state.log(f"{color} captures the enemy HQ at {target} and wins!")
        return

    effect_key = TROOPS[tile.troop_id].effect
    if effect_key and not effects_suppressed(node):
        handler = TROOP_EFFECTS.get(effect_key)
        if handler:
            handler(_make_ctx(state, color, target, decider, placements))


def _make_ctx(state, color, node_id, decider, placements) -> EffectContext:
    def place_extra(t: Tile, tgt: str) -> None:
        _place_tile(state, color, t, tgt, decider, placements)

    return EffectContext(
        state=state,
        color=color,
        node_id=node_id,
        decider=decider,
        placements=placements,
        place_tile=place_extra,
    )


def _score_regions(state: GameState, color: str) -> None:
    for rid in newly_controlled_regions(state, color):
        region = next(r for r in state.board.regions if r.id == rid)
        state.claimed_regions.add(rid)
        state.players[color].medals += region.medals
        state.log(
            f"{color} controls region {rid}: +{region.medals} medals "
            f"(total {state.players[color].medals}/{state.board.medal_objective})"
        )


def _check_medal_win(state: GameState, color: str) -> None:
    if state.winner is None and state.players[color].medals >= state.board.medal_objective:
        state.winner = color
        state.end_reason = "medal_objective"
        state.log(f"{color} reaches the medal objective and wins!")


def check_stuck(state: GameState) -> bool:
    """Whether the current player has no legal move."""
    return len(legal_moves(state)) == 0


def _maybe_end_stuck(state: GameState) -> None:
    if legal_moves(state):
        return
    stuck = state.current
    other = state.opponent(stuck)
    ms = state.players[stuck].medals
    mo = state.players[other].medals
    if ms > mo:
        state.winner = stuck
    else:
        # More medals for the other player, OR a tie (the player who ends the
        # game loses ties) -> the other player wins.
        state.winner = other
    state.end_reason = "stuck"
    state.log(
        f"{stuck} cannot move. Medals: {stuck}={ms}, {other}={mo}. Winner: {state.winner}"
    )
