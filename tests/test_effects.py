"""Targeted tests for individual troop and special-base effects."""

from toy_battle.engine import Move, apply_move, load_board, new_game
from toy_battle.engine.moves import MoveKind
from toy_battle.engine.state import Tile


def make_state(first="red", seed=0):
    board = load_board("plaine_des_chateaux")
    state = new_game(board, first_player=first, seed=seed)
    for p in state.players.values():
        p.rack.clear()
    state.piles.clear()
    return state


def give(state, color, troop_id, uid):
    t = Tile(troop_id=troop_id, color=color, uid=uid)
    state.players[color].rack.append(t)
    return t


def pick_first(state, decision):
    return decision.options[0] if decision.options else None


def test_skully_draws_two():
    state = make_state("red")
    red = state.players["red"]
    # Ensure a known reserve so we can count draws.
    red.reserve = [Tile("roxy", "red", 500 + i) for i in range(5)]
    give(state, "red", "skully", 600)  # force 1, connected to b_tl
    before = len(red.reserve)
    apply_move(state, Move(MoveKind.PLACE, tile_uid=600, target="b_tl"))
    # Skully drew 2 from reserve onto rack.
    assert len(red.reserve) == before - 2
    assert len(red.rack) == 2


def test_star_draws_one():
    state = make_state("red")
    red = state.players["red"]
    red.reserve = [Tile("roxy", "red", 700 + i) for i in range(3)]
    give(state, "red", "star", 800)  # force 6
    apply_move(state, Move(MoveKind.PLACE, tile_uid=800, target="b_tl"))
    assert len(red.rack) == 1
    assert len(red.reserve) == 2


def test_xb42_discards_random_enemy_rack_tile():
    state = make_state("red")
    blue = state.players["blue"]
    blue.rack = [Tile("roxy", "blue", 900), Tile("skully", "blue", 901)]
    give(state, "red", "xb42", 1000)  # force 5
    apply_move(state, Move(MoveKind.PLACE, tile_uid=1000, target="b_tl"))
    assert len(blue.rack) == 1
    assert len(blue.discard) == 1


def test_mastok_discards_adjacent_enemy():
    state = make_state("red")
    # Put a blue troop on b_ul (adjacent to b_tl), then place red Mastok on b_tl.
    state.push("b_ul", Tile("skully", "blue", 1100))
    give(state, "red", "mastok", 1200)  # force 3
    apply_move(
        state, Move(MoveKind.PLACE, tile_uid=1200, target="b_tl"), decider=pick_first
    )
    # b_ul enemy tile should be discarded.
    assert state.top_tile("b_ul") is None
    assert any(t.uid == 1100 for t in state.players["blue"].discard)


def test_captaine_places_extra_troop():
    state = make_state("red")
    give(state, "red", "captaine", 1300)  # force 2
    extra = give(state, "red", "roxy", 1301)

    def decider(st, dec):
        if dec.kind == "captaine_extra":
            # choose to place the Roxy on b_tr
            for o in dec.options:
                if o["tile_uid"] == 1301 and o["target"] == "b_tr":
                    return o
        return dec.options[0] if dec.options and not dec.optional else None

    apply_move(state, Move(MoveKind.PLACE, tile_uid=1300, target="b_tl"), decider=decider)
    assert state.top_tile("b_tl").troop_id == "captaine"
    assert state.top_tile("b_tr").troop_id == "roxy"
    assert extra not in state.players["red"].rack


def test_plaine_base_returns_a_troop():
    state = make_state("red")
    # Red already occupies b_ul; placing on special base s_ul triggers the
    # plaine effect to return the b_ul troop to the rack.
    state.push("b_ul", Tile("roxy", "red", 1400))
    state.push("b_tl", Tile("roxy", "red", 1401))  # connects s_ul
    give(state, "red", "roxy", 1500)  # force 7

    def decider(st, dec):
        if dec.kind == "plaine_return":
            for o in dec.options:
                if o["node"] == "b_ul":
                    return o
        return None

    apply_move(state, Move(MoveKind.PLACE, tile_uid=1500, target="s_ul"), decider=decider)
    # b_ul troop returned to rack; s_ul now occupied by the placed Roxy.
    assert state.top_tile("b_ul") is None
    assert any(t.uid == 1400 for t in state.players["red"].rack)
    assert state.top_tile("s_ul").uid == 1500
