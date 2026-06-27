"""Command-line entry point: play a game or run a benchmark.

Examples
--------
Play human vs heuristic bot::

    python -m toy_battle.cli play --red human --blue heuristic

Watch two bots::

    python -m toy_battle.cli play --red heuristic --blue random --verbose

LLM vs bot (needs the provider SDK + API key in the environment)::

    python -m toy_battle.cli play --red anthropic:claude-opus-4-8:tool --blue heuristic

Benchmark a set of agents round-robin::

    python -m toy_battle.cli benchmark --agents random heuristic --games 20 --out results/
"""

from __future__ import annotations

import argparse
import sys

from .agents import make_agent
from .engine import list_boards, load_board
from .match.benchmark import format_summary, run_benchmark
from .match.runner import play_game


def _make_factory(spec: str):
    return lambda color: make_agent(spec, color=color)


def cmd_play(args: argparse.Namespace) -> int:
    board = load_board(args.board)
    agents = {
        "red": make_agent(args.red, color="red"),
        "blue": make_agent(args.blue, color="blue"),
    }
    result = play_game(
        board,
        agents,
        first_player=args.first,
        seed=args.seed,
        log_path=args.log,
        verbose=args.verbose,
    )
    print("\n=== Result ===")
    print(f"Board:       {result.board}")
    print(f"Winner:      {result.winner} ({result.end_reason})")
    print(f"Turns:       {result.turns}")
    print(f"Medals:      {result.medals}")
    print(f"Agents:      {result.agents}")
    if result.error:
        print(f"Error:       {result.error}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    specs = {spec: _make_factory(spec) for spec in args.agents}
    summary = run_benchmark(
        args.board,
        specs,
        games_per_pair=args.games,
        base_seed=args.seed,
        out_dir=args.out,
    )
    print(format_summary(summary))
    if args.out:
        print(f"\nWrote results to {args.out}/ (summary.json, summary.csv, per-game logs)")
    return 0


def cmd_boards(_args: argparse.Namespace) -> int:
    for name in list_boards():
        b = load_board(name)
        print(f"{name:24s} objective={b.medal_objective} bases={len(b.base_nodes)} regions={len(b.regions)}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    from .engine import new_game
    from .render.image import render_state

    board = load_board(args.board)
    state = new_game(board, seed=args.seed)
    path = render_state(state, args.out)
    print(f"Rendered initial board to {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="toy-battle", description="Toy Battle engine + LLM benchmark")
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("play", help="play a single game")
    pp.add_argument("--board", default="plaine_des_chateaux")
    pp.add_argument("--red", default="human", help="agent spec for red")
    pp.add_argument("--blue", default="heuristic", help="agent spec for blue")
    pp.add_argument("--first", default=None, choices=[None, "red", "blue"])
    pp.add_argument("--seed", type=int, default=None)
    pp.add_argument("--log", default=None, help="write a JSONL turn log to this path")
    pp.add_argument("--verbose", action="store_true")
    pp.set_defaults(func=cmd_play)

    pb = sub.add_parser("benchmark", help="round-robin benchmark of agents")
    pb.add_argument("--board", default="plaine_des_chateaux")
    pb.add_argument("--agents", nargs="+", required=True, help="agent specs")
    pb.add_argument("--games", type=int, default=10, help="games per pair")
    pb.add_argument("--seed", type=int, default=0)
    pb.add_argument("--out", default=None, help="output directory for results")
    pb.set_defaults(func=cmd_benchmark)

    pl = sub.add_parser("boards", help="list available boards")
    pl.set_defaults(func=cmd_boards)

    pr = sub.add_parser("render", help="render a board to PNG")
    pr.add_argument("--board", default="plaine_des_chateaux")
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--out", default="board.png")
    pr.set_defaults(func=cmd_render)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
