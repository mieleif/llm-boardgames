"""Build a per-player observation (hidden info enforced) and render it as text.

The same observation dict feeds:
- the tool-use LLM mode (as JSON),
- the vision LLM mode (as a short caption next to the rendered PNG),
- the human CLI.
"""

from __future__ import annotations

from ..engine.board import HQ, SPECIAL_BASE
from ..engine.moves import Move, MoveKind, legal_moves
from ..engine.state import GameState
from ..engine.troops import TROOPS


def _tile_view(tile) -> dict | None:
    if tile is None:
        return None
    t = TROOPS[tile.troop_id]
    return {"troop": tile.troop_id, "name": t.name, "force": t.force_label, "color": tile.color}


def legal_moves_view(state: GameState) -> list[dict]:
    """The current player's legal moves, indexed, as plain dicts."""
    out = []
    for i, m in enumerate(legal_moves(state)):
        entry = {"index": i, "kind": m.kind.value, "description": m.describe(state)}
        if m.kind == MoveKind.PLACE:
            tile = next(
                (t for t in state.players[state.current].rack if t.uid == m.tile_uid), None
            )
            entry["troop"] = tile.troop_id if tile else None
            entry["tile_uid"] = m.tile_uid
            entry["target"] = m.target
        out.append(entry)
    return out


def build_observation(state: GameState, color: str) -> dict:
    """A complete, hidden-info-respecting view of the game for ``color``.

    Note: ``legal_moves`` reflect the player *to move*. They are only meaningful
    when ``color == state.current``.
    """
    board = state.board
    me = state.players[color]
    opp_color = state.opponent(color)
    opp = state.players[opp_color]

    bases = []
    for nid, node in board.nodes.items():
        top = state.top_tile(nid)
        stack = state.piles.get(nid, [])
        bases.append(
            {
                "id": nid,
                "kind": node.kind,
                "effect": node.effect,
                "hq_owner": node.owner if node.kind == HQ else None,
                "allowed_forces": list(node.allowed_forces) if node.allowed_forces else None,
                "occupied_by": top.color if top else None,
                "top": _tile_view(top),
                "stack_height": len(stack),
                "neighbors": sorted(board.neighbors(nid)),
            }
        )

    regions = []
    for r in board.regions:
        controlled_by = None
        for c in state.players:
            if all(state.occupies(c, nid) for nid in r.border):
                controlled_by = c
                break
        regions.append(
            {
                "id": r.id,
                "border": list(r.border),
                "medals": r.medals,
                "claimed": r.id in state.claimed_regions,
                "controlled_by": controlled_by,
            }
        )

    obs = {
        "game": "toy_battle",
        "board": {
            "name": board.name,
            "display_name": board.display_name,
            "medal_objective": board.medal_objective,
        },
        "you": color,
        "to_move": state.current,
        "turn_count": state.turn_count,
        "winner": state.winner,
        "end_reason": state.end_reason,
        "your_rack": [_tile_view(t) | {"uid": t.uid} for t in me.rack],
        "your_reserve_count": len(me.reserve),
        "your_discard": [_tile_view(t) for t in me.discard],
        "your_medals": me.medals,
        "opponent": {
            "color": opp_color,
            "rack_count": len(opp.rack),
            "reserve_count": len(opp.reserve),
            "discard": [_tile_view(t) for t in opp.discard],
            "medals": opp.medals,
            "last_action": state.last_moves.get(opp_color),
        },
        "your_last_action": state.last_moves.get(color),
        "bases": bases,
        "regions": regions,
        "your_hq": board.hq_of(color),
        "enemy_hqs": board.enemy_hqs(color),
    }
    if color == state.current and state.winner is None:
        obs["legal_moves"] = legal_moves_view(state)
    return obs


def _short(tile_view: dict | None) -> str:
    if tile_view is None:
        return "."
    code = TROOPS[tile_view["troop"]].code
    c = tile_view["color"][0].upper()  # R / B
    return f"{c}:{code}{tile_view['force']}"


def observation_to_text(obs: dict) -> str:
    """Compact human/LLM-readable rendering of an observation."""
    lines: list[str] = []
    b = obs["board"]
    lines.append(f"=== Toy Battle — {b['display_name']} (objective: {b['medal_objective']} medals) ===")
    lines.append(
        f"You are {obs['you'].upper()} | to move: {obs['to_move'].upper()} | turn {obs['turn_count']}"
    )
    if obs.get("winner"):
        lines.append(f"GAME OVER — winner: {obs['winner']} ({obs['end_reason']})")
    foe_last = obs["opponent"].get("last_action")
    lines.append(f"Opponent's last action: {foe_last if foe_last else '(none yet)'}")
    lines.append("")
    lines.append(
        f"Your medals: {obs['your_medals']}/{b['medal_objective']}  | "
        f"Opponent medals: {obs['opponent']['medals']}"
    )
    rack = ", ".join(f"{t['name']}(f{t['force']})" for t in obs["your_rack"]) or "(empty)"
    lines.append(f"Your rack ({len(obs['your_rack'])}/8): {rack}")
    lines.append(
        f"Your reserve: {obs['your_reserve_count']} | "
        f"Opp rack: {obs['opponent']['rack_count']} | Opp reserve: {obs['opponent']['reserve_count']}"
    )
    lines.append("")
    lines.append("Bases (id: occupant top tile | effect):")
    for base in obs["bases"]:
        marker = ""
        if base["kind"] == HQ:
            marker = f" [HQ {base['hq_owner']}]"
        elif base["kind"] == SPECIAL_BASE:
            marker = f" [special:{base['effect']}]"
        occ = _short(base["top"])
        stack = f" x{base['stack_height']}" if base["stack_height"] > 1 else ""
        lines.append(f"  {base['id']:7s}{marker}: {occ}{stack}  -> {', '.join(base['neighbors'])}")
    lines.append("")
    lines.append("Regions (control all border bases to win the medals):")
    for r in obs["regions"]:
        status = "CLAIMED" if r["claimed"] else f"controlled_by={r['controlled_by']}"
        lines.append(f"  {r['id']:9s}: {r['medals']} medals, border={r['border']} [{status}]")
    if "legal_moves" in obs:
        lines.append("")
        lines.append("Legal moves:")
        for m in obs["legal_moves"]:
            lines.append(f"  [{m['index']}] {m['description']}")
    return "\n".join(lines)
