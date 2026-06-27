"""End-to-end and rule-correctness tests for the engine."""

import random

import pytest

from toy_battle.engine import (
    Move,
    apply_move,
    legal_moves,
    load_board,
    new_game,
)
from toy_battle.engine.moves import MoveKind, placement_allowed
from toy_battle.engine.rules import IllegalMove
from toy_battle.engine.state import RACK_MAX, Tile


def random_decider(state, decision):
    if decision.optional and state.rng.random() < 0.5:
        return None
    return state.rng.choice(decision.options) if decision.options else None


def play_random_game(seed, max_steps=2000):
    board = load_board("plaine_des_chateaux")
    state = new_game(board, seed=seed)
    steps = 0
    while state.winner is None and steps < max_steps:
        lm = legal_moves(state)
        if not lm:
            break
        apply_move(state, state.rng.choice(lm), random_decider)
        steps += 1
    return state, steps


@pytest.mark.parametrize("seed", range(50))
def test_random_games_terminate_with_valid_winner(seed):
    state, steps = play_random_game(seed)
    assert state.winner in ("red", "blue")
    assert state.end_reason in ("hq_captured", "medal_objective", "stuck")
    assert steps < 2000


def test_setup_counts():
    board = load_board("plaine_des_chateaux")
    state = new_game(board, first_player="red", seed=1)
    # 24 tiles per colour minus 4 removed = 20; first player drew 3, second drew 4.
    red = state.players["red"]
    blue = state.players["blue"]
    assert len(red.rack) == 3
    assert len(red.rack) + len(red.reserve) == 20
    assert len(blue.rack) == 4
    assert len(blue.rack) + len(blue.reserve) == 20
    assert state.current == "red"


def test_draw_caps_at_rack_max():
    board = load_board("plaine_des_chateaux")
    state = new_game(board, first_player="red", seed=2)
    red = state.players["red"]
    # Fill rack to 7, then a DRAW (request 2) should only add 1.
    while len(red.rack) < 7:
        red.rack.append(red.reserve.pop())
    before = len(red.rack)
    apply_move(state, Move(MoveKind.DRAW))
    assert len(red.rack) == before + 1 == RACK_MAX


def _empty_state(first="red", seed=0):
    board = load_board("plaine_des_chateaux")
    state = new_game(board, first_player=first, seed=seed)
    # Clear racks/board so we can set up precise scenarios.
    for p in state.players.values():
        p.rack.clear()
    state.piles.clear()
    return state


def _give(state, color, troop_id, uid):
    t = Tile(troop_id=troop_id, color=color, uid=uid)
    state.players[color].rack.append(t)
    return t


def test_connection_rule_blocks_unconnected_placement():
    state = _empty_state("red")
    # Red HQ connects to b_tl and b_tr. br_c is far; not connected initially.
    assert not placement_allowed(state, "red", "roxy", "br_c")
    assert placement_allowed(state, "red", "roxy", "b_tl")


def test_strictly_greater_force_to_cover_enemy():
    state = _empty_state("red")
    # Put an enemy (blue) Mastok(force 3) on b_tl.
    state.push("b_tl", Tile("mastok", "blue", 900))
    # Red Cap'taine (force 2) cannot cover; Crochet(4) can (and ignores connection).
    assert not placement_allowed(state, "red", "captaine", "b_tl")
    assert placement_allowed(state, "red", "crochet", "b_tl", ignore_connection=True)
    # Equal force cannot cover.
    state.piles["b_tl"][-1] = Tile("crochet", "blue", 901)  # force 4
    assert not placement_allowed(state, "red", "crochet", "b_tl", ignore_connection=True)


def test_joker_cover_rules():
    state = _empty_state("red")
    # Enemy Roxy(7) on b_tl; red Kwak joker may always cover it.
    state.push("b_tl", Tile("roxy", "blue", 902))
    assert placement_allowed(state, "red", "kwak", "b_tl")
    # Enemy Kwak joker on b_tr; any red troop may cover it.
    state.push("b_tr", Tile("kwak", "blue", 903))
    assert placement_allowed(state, "red", "skully", "b_tr")


def test_capturing_enemy_hq_wins():
    state = _empty_state("red")
    # Occupy a base adjacent to blue HQ so the HQ is connected, then capture.
    # blue HQ connects to b_bl and b_br. Build a red chain is long; instead use
    # Crochet? Crochet still needs connection for HQ. So occupy b_br via setup.
    # Manually mark red occupying b_br and a chain back to red HQ is not needed for
    # this unit: we directly test the capture resolution by connecting through b_br.
    # Give red occupancy of b_br and the path to it.
    for nid in ["b_tr", "s_ur", "b_ur", "br_r", "b_lr", "b_br"]:
        state.push(nid, Tile("roxy", "red", 1000 + len(state.piles)))
    tile = _give(state, "red", "skully", 2000)
    assert placement_allowed(state, "red", "skully", "hq_blue")
    apply_move(state, Move(MoveKind.PLACE, tile_uid=2000, target="hq_blue"))
    assert state.winner == "red"
    assert state.end_reason == "hq_captured"


def test_region_control_awards_medals():
    state = _empty_state("red")
    # Region r_ul border = b_tl, b_ul, s_ul. Occupy two, then place the third
    # (connected) and expect +1 medal.
    state.push("b_tl", Tile("roxy", "red", 3000))
    state.push("b_ul", Tile("roxy", "red", 3001))
    tile = _give(state, "red", "roxy", 3002)
    assert placement_allowed(state, "red", "roxy", "s_ul")
    apply_move(state, Move(MoveKind.PLACE, tile_uid=3002, target="s_ul"))
    assert state.players["red"].medals == 1
    assert "r_ul" in state.claimed_regions


def test_illegal_move_raises():
    state = _empty_state("red")
    _give(state, "red", "roxy", 4000)
    with pytest.raises(IllegalMove):
        apply_move(state, Move(MoveKind.PLACE, tile_uid=4000, target="br_c"))
