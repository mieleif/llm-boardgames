# Toy Battle — distilled mechanics

This is a paraphrased reference of the rules **as implemented by the engine**. It
covers game mechanics only (which are not copyrightable); no artwork or rulebook
text is reproduced. Toy Battle is a 1-vs-1 game by Paolo Mori & Alessandro
Zucchini, published by Repos Production (2025).

## Objective
Win immediately by either:
1. placing one of your troops on the **enemy HQ**, or
2. collecting medals up to the board's **medal objective** (7 on La plaine des
   châteaux).

If a player has no legal move, the game ends and the player with more medals
wins. On a tie, the player who got stuck loses.

## Components (per colour)
- 24 troop tiles: 3 copies each of 8 troop types. At setup, **4 are removed
  unseen**, leaving a 20-tile reserve.
- A rack ("support"), max **8 tiles**.

## Troops (force, effect)
| Troop | Force | Effect |
|-------|-------|--------|
| Kwak | joker | none; wild for covering / being covered |
| Skully | 1 | draw 2 tiles |
| Cap'taine | 2 | place 1 extra troop this turn (and apply its effect) |
| Mastok | 3 | discard a visible enemy troop adjacent to Mastok |
| Crochet | 4 | ignore the connection rule when placing |
| XB-42 | 5 | randomly discard one tile from the enemy rack |
| Star | 6 | draw 1 tile |
| Roxy | 7 | none |

## A turn — exactly one action
- **Draw**: take up to 2 tiles from the reserve onto the rack (1 if only 1 slot
  or 1 reserve tile remains; impossible if the rack is full or the reserve is
  empty).
- **Place**: put one rack tile on a base (or the enemy HQ), then resolve its
  troop effect, then the special-base effect (if any).

## Placement legality
A tile may be placed on:
- an empty base,
- a base you already occupy (stacks on top),
- a base topped by an enemy troop of **strictly lower force** (jokers: a Kwak
  covers anything; anything covers a Kwak),
- the **enemy HQ** (never your own).

Only the **top tile** of a stack occupies a base. Stacks can be inspected; their
order never changes.

## Connection to your HQ
The base you place on must connect back to your HQ through a continuous chain of
bases **you currently occupy**. An empty base or an enemy-topped base breaks the
chain. Crochet ignores this for bases; capturing an enemy HQ always requires
connection.

## Regions & medals
A region is bounded by a set of bases. The instant you occupy **all** of a
region's border bases, you take its medals (kept for the rest of the game, even
if you later lose the bases). Medals pay out once; multiple regions can be
captured in one turn.

## Special bases
Modeled generically (effect id on the node). Implemented for the starter map:
- **plaine_return** (La plaine des châteaux): return one of your other on-board
  troops to your rack (cannot exceed 8 on the rack).

Other terrains' effects (cité des nuages, jungle volcanique, cimetière maudit,
champ de bataille, piscine des tropiques value-restriction, station Métal-X
effect suppression, mer des Caraïbes asymmetric HQs) are reserved for future
boards and slot into the same registry.
