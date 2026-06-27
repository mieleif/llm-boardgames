"""Mutable game state and game setup.

Hidden-information model:
- A player's ``reserve`` is an ordered face-down draw pile: the owner does not
  know the order/contents either; only its count is public.
- A player's ``rack`` (the "support") is private to that player.
- Board piles are public at the top; full stacks are inspectable by both
  players (rules allow consulting piles) but their order can never change.
- The ``discard`` is public (face up).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .board import Board
from .troops import COPIES_PER_TROOP, TROOPS, troop

RACK_MAX = 8
TILES_REMOVED_AT_SETUP = 4
START_TILES_FIRST = 3
START_TILES_SECOND = 4


@dataclass
class Tile:
    troop_id: str
    color: str
    uid: int  # unique per tile instance within a game

    @property
    def force(self) -> int | None:
        return TROOPS[self.troop_id].force

    @property
    def is_joker(self) -> bool:
        return TROOPS[self.troop_id].is_joker

    @property
    def name(self) -> str:
        return TROOPS[self.troop_id].name

    def to_public(self) -> dict:
        t = TROOPS[self.troop_id]
        return {
            "uid": self.uid,
            "troop": self.troop_id,
            "name": t.name,
            "force": t.force_label,
            "color": self.color,
        }


@dataclass
class PlayerState:
    color: str
    rack: list[Tile] = field(default_factory=list)
    reserve: list[Tile] = field(default_factory=list)
    discard: list[Tile] = field(default_factory=list)
    medals: int = 0

    def rack_has_troop(self, troop_id: str) -> bool:
        return any(t.troop_id == troop_id for t in self.rack)

    def take_from_rack(self, uid: int) -> Tile:
        for i, t in enumerate(self.rack):
            if t.uid == uid:
                return self.rack.pop(i)
        raise KeyError(f"tile uid {uid} not in {self.color} rack")


@dataclass
class GameState:
    board: Board
    players: dict[str, PlayerState]
    piles: dict[str, list[Tile]]  # node_id -> stack (top = last element)
    current: str
    claimed_regions: set[str] = field(default_factory=set)
    rng: random.Random = field(default_factory=random.Random)
    winner: str | None = None
    end_reason: str | None = None
    turn_count: int = 0
    history: list[str] = field(default_factory=list)  # human-readable log lines

    # --- queries -----------------------------------------------------------
    def opponent(self, color: str) -> str:
        return next(c for c in self.players if c != color)

    def top_tile(self, node_id: str) -> Tile | None:
        pile = self.piles.get(node_id)
        return pile[-1] if pile else None

    def occupies(self, color: str, node_id: str) -> bool:
        """Whether ``color`` currently occupies (tops the pile of) a base."""
        top = self.top_tile(node_id)
        return top is not None and top.color == color

    def on_board_nodes(self, color: str) -> list[str]:
        return [nid for nid in self.piles if self.occupies(color, nid)]

    # --- mutation helpers --------------------------------------------------
    def push(self, node_id: str, tile: Tile) -> None:
        self.piles.setdefault(node_id, []).append(tile)

    def pop_top(self, node_id: str) -> Tile:
        return self.piles[node_id].pop()

    def log(self, line: str) -> None:
        self.history.append(line)

    # --- cloning -----------------------------------------------------------
    def clone(self) -> "GameState":
        """Deep copy of the volatile state (board is shared/immutable)."""
        players = {
            c: PlayerState(
                color=p.color,
                rack=list(p.rack),
                reserve=list(p.reserve),
                discard=list(p.discard),
                medals=p.medals,
            )
            for c, p in self.players.items()
        }
        piles = {nid: list(stack) for nid, stack in self.piles.items()}
        new_rng = random.Random()
        new_rng.setstate(self.rng.getstate())
        return GameState(
            board=self.board,
            players=players,
            piles=piles,
            current=self.current,
            claimed_regions=set(self.claimed_regions),
            rng=new_rng,
            winner=self.winner,
            end_reason=self.end_reason,
            turn_count=self.turn_count,
            history=list(self.history),
        )


def _build_deck(color: str, start_uid: int) -> tuple[list[Tile], int]:
    tiles: list[Tile] = []
    uid = start_uid
    for troop_id in TROOPS:
        for _ in range(COPIES_PER_TROOP):
            tiles.append(Tile(troop_id=troop_id, color=color, uid=uid))
            uid += 1
    return tiles, uid


def new_game(
    board: Board,
    *,
    first_player: str | None = None,
    seed: int | None = None,
) -> GameState:
    """Set up a fresh game on ``board``.

    Each colour gets 24 tiles (3 copies x 8 troops), 4 removed unseen, leaving a
    20-tile reserve. The first player draws 3 onto their rack, the second draws 4.
    """
    rng = random.Random(seed)
    colors = board.colors
    if first_player is None:
        first_player = rng.choice(list(colors))
    if first_player not in colors:
        raise ValueError(f"unknown first player colour: {first_player}")

    players: dict[str, PlayerState] = {}
    uid = 0
    for color in colors:
        deck, uid = _build_deck(color, uid)
        rng.shuffle(deck)
        # Remove 4 unseen tiles for the whole game.
        deck = deck[TILES_REMOVED_AT_SETUP:]
        players[color] = PlayerState(color=color, reserve=deck)

    state = GameState(
        board=board,
        players=players,
        piles={},
        current=first_player,
        rng=rng,
    )

    # Initial draws: first player 3, opponent 4.
    second = state.opponent(first_player)
    _initial_draw(players[first_player], START_TILES_FIRST)
    _initial_draw(players[second], START_TILES_SECOND)

    state.log(
        f"Setup on '{board.display_name}'. {first_player} starts (3 tiles), "
        f"{second} holds 4 tiles. Medal objective: {board.medal_objective}."
    )
    return state


def _initial_draw(player: PlayerState, n: int) -> None:
    for _ in range(n):
        if not player.reserve:
            break
        player.rack.append(player.reserve.pop())
