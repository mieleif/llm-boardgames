"""Core rules engine for Toy Battle."""

from .troops import TROOPS, Troop, troop
from .board import Board, Node, Region, load_board, list_boards
from .state import GameState, PlayerState, Tile, new_game
from .moves import Move, MoveKind, Decision, legal_moves
from .rules import apply_move, check_stuck

__all__ = [
    "TROOPS",
    "Troop",
    "troop",
    "Board",
    "Node",
    "Region",
    "load_board",
    "list_boards",
    "GameState",
    "PlayerState",
    "Tile",
    "new_game",
    "Move",
    "MoveKind",
    "Decision",
    "legal_moves",
    "apply_move",
    "check_stuck",
]
