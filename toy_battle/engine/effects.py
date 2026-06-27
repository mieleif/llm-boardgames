"""Troop and special-base effect handlers.

Each handler receives an :class:`EffectContext`. Handlers that require a choice
build a :class:`~toy_battle.engine.moves.Decision` and resolve it through
``ctx.decider`` (options are pre-validated by the engine). The recursive
placement used by Cap'taine is injected as ``ctx.place_tile`` to avoid importing
``rules`` (which would be circular).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .moves import Decision, draw_count, placeable_targets
from .state import RACK_MAX, GameState, Tile
from .troops import TROOPS

# Special bases whose presence suppresses effects (e.g. station Metal-X).
EFFECT_SUPPRESSING_BASES = {"metalx_no_effect"}

Decider = Callable[[GameState, Decision], Optional[dict]]


@dataclass
class EffectContext:
    state: GameState
    color: str
    node_id: str  # base the triggering troop was placed on
    decider: Decider
    placements: list  # (node_id, tile) placed during this action, in order
    place_tile: Callable[[Tile, str], None]  # inject recursive placement


def effects_suppressed(node) -> bool:
    return node.effect in EFFECT_SUPPRESSING_BASES


def draw_to_rack(state: GameState, color: str, requested: int) -> int:
    n = draw_count(state, color, requested)
    p = state.players[color]
    for _ in range(n):
        p.rack.append(p.reserve.pop())
    return n


# --- troop effects --------------------------------------------------------

def eff_draw2(ctx: EffectContext) -> None:
    n = draw_to_rack(ctx.state, ctx.color, 2)
    ctx.state.log(f"  Skully: {ctx.color} draws {n} tile(s)")


def eff_draw1(ctx: EffectContext) -> None:
    n = draw_to_rack(ctx.state, ctx.color, 1)
    ctx.state.log(f"  Star: {ctx.color} draws {n} tile(s)")


def eff_random_discard_enemy_rack(ctx: EffectContext) -> None:
    state = ctx.state
    enemy = state.opponent(ctx.color)
    ep = state.players[enemy]
    if not ep.rack:
        return
    idx = state.rng.randrange(len(ep.rack))
    tile = ep.rack.pop(idx)
    ep.discard.append(tile)
    state.log(f"  XB-42: discards {enemy}'s {tile.name} from their rack")


def eff_discard_adjacent_enemy(ctx: EffectContext) -> None:
    state = ctx.state
    enemy = state.opponent(ctx.color)
    options: list[dict] = []
    for nb in state.board.neighbors(ctx.node_id):
        if state.occupies(enemy, nb):
            top = state.top_tile(nb)
            options.append(
                {"node": nb, "troop": top.troop_id, "label": f"discard {top.name} on {nb}"}
            )
    if not options:
        return
    choice = ctx.decider(
        state,
        Decision(
            "mastok_target",
            "Mastok: discard 1 visible adjacent enemy troop (optional).",
            options,
            optional=True,
        ),
    )
    if not choice:
        return
    tile = state.pop_top(choice["node"])
    state.players[enemy].discard.append(tile)
    state.log(f"  Mastok: discards {enemy}'s {tile.name} from {choice['node']}")


def eff_extra_place(ctx: EffectContext) -> None:
    """Cap'taine: place 1 extra troop (and apply its effect)."""
    state, color = ctx.state, ctx.color
    p = state.players[color]
    options: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for tile in p.rack:
        for target in placeable_targets(state, color, tile.troop_id):
            key = (tile.troop_id, target)
            if key in seen:
                continue
            seen.add(key)
            options.append(
                {
                    "tile_uid": tile.uid,
                    "troop": tile.troop_id,
                    "target": target,
                    "label": f"place {TROOPS[tile.troop_id].name} on {target}",
                }
            )
    if not options:
        return
    choice = ctx.decider(
        state,
        Decision(
            "captaine_extra",
            "Cap'taine: place 1 extra troop from your rack (optional).",
            options,
            optional=True,
        ),
    )
    if not choice:
        return
    tile = p.take_from_rack(choice["tile_uid"])
    state.log(f"  Cap'taine: places extra {tile.name} on {choice['target']}")
    ctx.place_tile(tile, choice["target"])


def eff_noop(ctx: EffectContext) -> None:  # ignore_connection handled at placement
    return


# --- special-base effects -------------------------------------------------

def base_plaine_return(ctx: EffectContext) -> None:
    """Plaine des chateaux: return 1 of your *other* on-board troops to your rack."""
    state, color = ctx.state, ctx.color
    if len(state.players[color].rack) >= RACK_MAX:
        return  # cannot exceed 8 tiles on the rack
    options: list[dict] = []
    for nid in state.on_board_nodes(color):
        if nid == ctx.node_id:
            continue
        top = state.top_tile(nid)
        options.append({"node": nid, "troop": top.troop_id, "label": f"return {top.name} from {nid}"})
    if not options:
        return
    choice = ctx.decider(
        state,
        Decision(
            "plaine_return",
            "Plaine des chateaux: return 1 of your other troops to your rack (optional).",
            options,
            optional=True,
        ),
    )
    if not choice:
        return
    tile = state.pop_top(choice["node"])
    state.players[color].rack.append(tile)
    state.log(f"  Plaine: {color} returns {tile.name} from {choice['node']} to rack")


TROOP_EFFECTS: dict[str, Callable[[EffectContext], None]] = {
    "draw2": eff_draw2,
    "draw1": eff_draw1,
    "extra_place": eff_extra_place,
    "discard_adjacent_enemy": eff_discard_adjacent_enemy,
    "random_discard_enemy_rack": eff_random_discard_enemy_rack,
    "ignore_connection": eff_noop,
}

BASE_EFFECTS: dict[str, Callable[[EffectContext], None]] = {
    "plaine_return": base_plaine_return,
}


def default_decider(state: GameState, decision: Decision) -> Optional[dict]:
    """Fallback when no agent decider is supplied: skip optional effects,
    pick the first option for mandatory ones."""
    if decision.optional:
        return None
    return decision.options[0] if decision.options else None
