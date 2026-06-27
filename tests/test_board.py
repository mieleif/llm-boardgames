from toy_battle.engine import load_board, list_boards
from toy_battle.engine.board import HQ


def test_plaine_loads_and_is_symmetric_in_size():
    b = load_board("plaine_des_chateaux")
    assert b.name == "plaine_des_chateaux"
    assert b.medal_objective == 7
    # Two HQs, one per colour.
    assert set(b.hqs) == {"red", "blue"}
    # Adjacency is symmetric.
    for a, nbrs in b.adj.items():
        for n in nbrs:
            assert a in b.adj[n], f"edge {a}-{n} not symmetric"


def test_total_medals_exceed_objective():
    b = load_board("plaine_des_chateaux")
    total = sum(r.medals for r in b.regions)
    assert total >= b.medal_objective
    assert total == 14


def test_regions_reference_real_bases():
    b = load_board("plaine_des_chateaux")
    for r in b.regions:
        for nid in r.border:
            assert nid in b.nodes
            assert b.nodes[nid].is_base


def test_list_boards_includes_plaine():
    assert "plaine_des_chateaux" in list_boards()
