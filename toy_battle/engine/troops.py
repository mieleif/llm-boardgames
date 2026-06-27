"""Troop catalog.

Each troop type has a *force* (1..7, or ``None`` for the Kwak joker) and an
optional *effect* (a key into the effect registry in ``effects.py``). Every
troop type exists in 3 copies per colour (24 tiles per colour).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Troop:
    id: str
    name: str
    force: int | None  # None == joker (Kwak)
    is_joker: bool
    effect: str | None  # effect key, or None for no effect
    code: str  # short label for compact rendering / text views

    @property
    def force_label(self) -> str:
        return "*" if self.is_joker else str(self.force)


# Ordered by force. Effects are implemented in effects.py.
TROOPS: dict[str, Troop] = {
    "kwak": Troop("kwak", "Kwak", None, True, None, "Kw"),
    "skully": Troop("skully", "Skully", 1, False, "draw2", "Sk"),
    "captaine": Troop("captaine", "Cap'taine", 2, False, "extra_place", "Cp"),
    "mastok": Troop("mastok", "Mastok", 3, False, "discard_adjacent_enemy", "Ma"),
    "crochet": Troop("crochet", "Crochet", 4, False, "ignore_connection", "Cr"),
    "xb42": Troop("xb42", "XB-42", 5, False, "random_discard_enemy_rack", "Xb"),
    "star": Troop("star", "Star", 6, False, "draw1", "St"),
    "roxy": Troop("roxy", "Roxy", 7, False, None, "Ro"),
}

# Copies of each troop per colour.
COPIES_PER_TROOP = 3


def troop(troop_id: str) -> Troop:
    return TROOPS[troop_id]


def can_cover(placing_id: str, occupying_id: str) -> bool:
    """Can a tile of ``placing_id`` be placed on top of an enemy tile of
    ``occupying_id``? Joker rules: a Kwak may cover anything, and anything may
    cover a Kwak. Otherwise force must be strictly greater."""
    placing = TROOPS[placing_id]
    occupying = TROOPS[occupying_id]
    if placing.is_joker or occupying.is_joker:
        return True
    return placing.force > occupying.force
