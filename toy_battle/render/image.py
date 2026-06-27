"""Render a board state to a PNG.

Fully programmatic (no copyrighted artwork): bases are rounded rectangles, the
top tile shows the troop's short code + force, HQs are diamonds, special bases
get a dashed accent, edges are drawn as path segments, and each region shows its
remaining medal count near its centroid. Legibility is deliberately high so the
vision/OCR benchmark measures model reading ability rather than image quality.
"""

from __future__ import annotations

from pathlib import Path

from ..engine.board import HQ, SPECIAL_BASE
from ..engine.state import GameState
from ..engine.troops import TROOPS

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - pillow is a hard dep for rendering
    Image = None

W, H = 900, 1100
MARGIN = 70
NODE_R = 46

COLORS = {
    "red": (208, 64, 56),
    "blue": (50, 96, 196),
    "bg": (232, 240, 220),
    "grid": (150, 170, 140),
    "base": (225, 222, 214),
    "base_edge": (120, 120, 120),
    "special": (245, 226, 150),
    "text": (30, 30, 30),
    "white": (255, 255, 255),
    "medal": (240, 190, 40),
    "path": (200, 188, 140),
}


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _xy(node) -> tuple[int, int]:
    x = MARGIN + node.x / 100.0 * (W - 2 * MARGIN)
    y = MARGIN + node.y / 100.0 * (H - 2 * MARGIN)
    return int(x), int(y)


def _text_center(draw, xy, text, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((xy[0] - tw / 2, xy[1] - th / 2 - bbox[1]), text, font=font, fill=fill)


def render_state(state: GameState, path: str | Path) -> str:
    """Render ``state`` to ``path`` (PNG). Returns the path as a string."""
    if Image is None:
        raise RuntimeError("Pillow is required for image rendering (pip install pillow)")
    board = state.board
    img = Image.new("RGB", (W, H), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    f_title = _font(26)
    f_label = _font(22)
    f_tile = _font(26)
    f_small = _font(18)

    # Title / objective.
    _text_center(
        draw,
        (W // 2, 28),
        f"{board.display_name}  (objective {board.medal_objective} medals)",
        f_title,
        COLORS["text"],
    )

    # Edges.
    drawn = set()
    for a, nbrs in board.adj.items():
        for b in nbrs:
            key = tuple(sorted((a, b)))
            if key in drawn:
                continue
            drawn.add(key)
            draw.line([_xy(board.nodes[a]), _xy(board.nodes[b])], fill=COLORS["path"], width=10)

    # Region medal counts at centroids.
    for r in board.regions:
        if r.id in state.claimed_regions:
            continue
        pts = [_xy(board.nodes[nid]) for nid in r.border]
        cx = sum(p[0] for p in pts) // len(pts)
        cy = sum(p[1] for p in pts) // len(pts)
        rad = 16
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=COLORS["medal"], outline=COLORS["text"], width=2)
        _text_center(draw, (cx, cy), str(r.medals), f_small, COLORS["text"])

    # Nodes.
    for nid, node in board.nodes.items():
        x, y = _xy(node)
        top = state.top_tile(nid)
        if node.kind == HQ:
            # Diamond in the owner's colour.
            col = COLORS[node.owner]
            pts = [(x, y - NODE_R), (x + NODE_R, y), (x, y + NODE_R), (x - NODE_R, y)]
            draw.polygon(pts, fill=col, outline=COLORS["text"])
            label = f"HQ {node.owner[0].upper()}"
            _text_center(draw, (x, y - 4), label, f_label, COLORS["white"])
            if top is not None:
                _text_center(draw, (x, y + 20), _tile_label(top), f_small, COLORS["white"])
            _text_center(draw, (x, y + NODE_R + 14), nid, f_small, COLORS["text"])
            continue

        fill = COLORS["special"] if node.kind == SPECIAL_BASE else COLORS["base"]
        box = [x - NODE_R, y - NODE_R, x + NODE_R, y + NODE_R]
        draw.rounded_rectangle(box, radius=12, fill=fill, outline=COLORS["base_edge"], width=3)
        if node.kind == SPECIAL_BASE:
            draw.rounded_rectangle(
                [box[0] + 5, box[1] + 5, box[2] - 5, box[3] - 5],
                radius=10,
                outline=COLORS["medal"],
                width=3,
            )
        # Top tile.
        if top is not None:
            tcol = COLORS[top.color]
            draw.rounded_rectangle(
                [x - NODE_R + 8, y - 22, x + NODE_R - 8, y + 22],
                radius=8,
                fill=tcol,
            )
            _text_center(draw, (x, y), _tile_label(top), f_tile, COLORS["white"])
            stack = len(state.piles.get(nid, []))
            if stack > 1:
                _text_center(draw, (x + NODE_R - 16, y - NODE_R + 12), f"x{stack}", f_small, COLORS["text"])
        # Node id label.
        _text_center(draw, (x, y + NODE_R - 12), nid, f_small, COLORS["base_edge"])

    # Score footer.
    red_m = state.players["red"].medals
    blue_m = state.players["blue"].medals
    _text_center(
        draw,
        (W // 2, H - 42),
        f"RED medals: {red_m}    |    BLUE medals: {blue_m}    |    to move: {state.current.upper()}",
        f_label,
        COLORS["text"],
    )
    # Last action line (the "foe action" describer), if any.
    if state.last_action_by and state.last_action_summary:
        last = f"Last: {state.last_action_by.upper()} {state.last_action_summary}"
        if len(last) > 90:
            last = last[:87] + "..."
        _text_center(draw, (W // 2, H - 16), last, f_small, COLORS["base_edge"])

    path = str(path)
    img.save(path)
    return path


def _tile_label(tile) -> str:
    t = TROOPS[tile.troop_id]
    return f"{t.code}{t.force_label}"
