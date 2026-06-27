"""Board (terrain) model and JSON loader.

A board is a graph: nodes (HQs, bases, special bases) connected by undirected
edges (path segments). Regions are closed zones identified by the set of bases
that border them; controlling all border bases captures the region's medals.

The format is data-driven so additional terrains (and other games) can be added
as JSON without code changes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

BOARDS_DIR = Path(__file__).resolve().parent.parent / "boards"

# Node kinds
HQ = "hq"
BASE = "base"
SPECIAL_BASE = "special_base"


@dataclass(frozen=True)
class Node:
    id: str
    kind: str  # HQ | BASE | SPECIAL_BASE
    x: float
    y: float
    owner: str | None = None  # colour, for HQ nodes
    effect: str | None = None  # base effect key, for special bases
    allowed_forces: tuple[int, ...] | None = None  # value restriction, if any

    @property
    def is_base(self) -> bool:
        """Whether this node counts as a 'base' (special bases included).
        The HQ is explicitly *not* a base."""
        return self.kind in (BASE, SPECIAL_BASE)


@dataclass(frozen=True)
class Region:
    id: str
    border: tuple[str, ...]  # node ids of the bases surrounding the region
    medals: int


@dataclass
class Board:
    name: str
    display_name: str
    medal_objective: int
    nodes: dict[str, Node]
    adj: dict[str, set[str]]
    regions: list[Region]
    hqs: dict[str, str] = field(default_factory=dict)  # colour -> node id
    colors: tuple[str, ...] = ("red", "blue")

    def neighbors(self, node_id: str) -> set[str]:
        return self.adj.get(node_id, set())

    def adjacent(self, a: str, b: str) -> bool:
        return b in self.adj.get(a, set())

    def hq_of(self, color: str) -> str:
        return self.hqs[color]

    def enemy_hqs(self, color: str) -> list[str]:
        return [nid for c, nid in self.hqs.items() if c != color]

    @property
    def base_nodes(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.is_base]


def _board_from_dict(data: dict) -> Board:
    nodes: dict[str, Node] = {}
    hqs: dict[str, str] = {}
    for nd in data["nodes"]:
        forces = nd.get("allowed_forces")
        node = Node(
            id=nd["id"],
            kind=nd["kind"],
            x=float(nd["x"]),
            y=float(nd["y"]),
            owner=nd.get("owner"),
            effect=nd.get("effect"),
            allowed_forces=tuple(forces) if forces else None,
        )
        nodes[node.id] = node
        if node.kind == HQ:
            if node.owner is None:
                raise ValueError(f"HQ node {node.id} must declare an owner colour")
            hqs[node.owner] = node.id

    adj: dict[str, set[str]] = {nid: set() for nid in nodes}
    for a, b in data["edges"]:
        if a not in nodes or b not in nodes:
            raise ValueError(f"edge references unknown node: {a}-{b}")
        adj[a].add(b)
        adj[b].add(a)

    regions = [
        Region(id=r["id"], border=tuple(r["border"]), medals=int(r["medals"]))
        for r in data["regions"]
    ]
    for r in regions:
        for nid in r.border:
            if nid not in nodes:
                raise ValueError(f"region {r.id} references unknown node {nid}")

    colors = tuple(data.get("colors", ("red", "blue")))
    return Board(
        name=data["name"],
        display_name=data.get("display_name", data["name"]),
        medal_objective=int(data["medal_objective"]),
        nodes=nodes,
        adj=adj,
        regions=regions,
        hqs=hqs,
        colors=colors,
    )


def load_board(name: str) -> Board:
    """Load a board by name (filename without .json) or by absolute path."""
    if os.path.sep in name or name.endswith(".json"):
        path = Path(name)
    else:
        path = BOARDS_DIR / f"{name}.json"
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return _board_from_dict(data)


def list_boards() -> list[str]:
    return sorted(p.stem for p in BOARDS_DIR.glob("*.json"))
