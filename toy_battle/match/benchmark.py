"""Benchmark harness: run many games across agent pairs and aggregate metrics.

Metrics surfaced per agent: win-rate, average medals, average game length,
illegal-move attempt rate (a proxy for board-reading / rule-understanding
errors), and token/cost/latency totals (populated by LLM agents). Results are
written as CSV + JSON and printed as a summary table. A simple Elo is computed
across all agents that played.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ..agents.base import Agent, AgentStats
from ..engine.board import Board, load_board
from .runner import GameResult, play_game

AgentFactory = Callable[[str], Agent]  # color -> Agent


@dataclass
class AgentTotals:
    name: str
    games: int = 0
    wins: int = 0
    losses: int = 0
    medals: int = 0
    turns: int = 0
    stats: AgentStats = field(default_factory=AgentStats)

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def avg_medals(self) -> float:
        return self.medals / self.games if self.games else 0.0

    @property
    def illegal_rate(self) -> float:
        return self.stats.illegal_attempts / self.stats.moves if self.stats.moves else 0.0


def benchmark_pair(
    board: Board,
    factory_a: AgentFactory,
    factory_b: AgentFactory,
    *,
    games: int = 10,
    base_seed: int = 0,
    alternate_first: bool = True,
    log_dir: Optional[str | Path] = None,
) -> list[GameResult]:
    """Play ``games`` between two agent factories, alternating who starts and
    swapping colours so neither side keeps a colour/first-move advantage."""
    results: list[GameResult] = []
    log_dir = Path(log_dir) if log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
    for i in range(games):
        # Swap which factory is red/blue every other game.
        if i % 2 == 0:
            red_factory, blue_factory = factory_a, factory_b
        else:
            red_factory, blue_factory = factory_b, factory_a
        agents = {"red": red_factory("red"), "blue": blue_factory("blue")}
        first = "red" if (not alternate_first or i % 2 == 0) else "blue"
        log_path = log_dir / f"game_{i:03d}.jsonl" if log_dir else None
        res = play_game(
            board,
            agents,
            first_player=first,
            seed=base_seed + i,
            log_path=log_path,
        )
        # Attach the agent objects' stats for aggregation.
        res.turn_log = []  # keep results light in memory
        res._agent_objs = agents  # type: ignore[attr-defined]
        results.append(res)
    return results


def aggregate(results: list[GameResult]) -> dict[str, AgentTotals]:
    totals: dict[str, AgentTotals] = {}
    for res in results:
        agent_objs = getattr(res, "_agent_objs", {})
        for color, name in res.agents.items():
            t = totals.setdefault(name, AgentTotals(name=name))
            t.games += 1
            t.turns += res.turns
            t.medals += res.medals.get(color, 0)
            if res.winner == color:
                t.wins += 1
            elif res.winner is not None:
                t.losses += 1
            if color in agent_objs:
                t.stats.merge(agent_objs[color].stats)
    return totals


def compute_elo(results: list[GameResult], k: float = 32.0, base: float = 1000.0) -> dict[str, float]:
    elo: dict[str, float] = defaultdict(lambda: base)
    for res in results:
        names = list(res.agents.values())
        colors = list(res.agents.keys())
        if len(names) != 2 or names[0] == names[1]:
            continue
        a_color, b_color = colors
        a, b = res.agents[a_color], res.agents[b_color]
        ra, rb = elo[a], elo[b]
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        if res.winner == a_color:
            sa = 1.0
        elif res.winner == b_color:
            sa = 0.0
        else:
            sa = 0.5
        elo[a] = ra + k * (sa - ea)
        elo[b] = rb + k * ((1 - sa) - (1 - ea))
    return dict(elo)


def run_benchmark(
    board_name: str,
    agent_specs: dict[str, AgentFactory],
    *,
    games_per_pair: int = 10,
    base_seed: int = 0,
    out_dir: Optional[str | Path] = None,
) -> dict:
    """Round-robin every pair of agents in ``agent_specs`` ({name: factory})."""
    board = load_board(board_name)
    out_dir = Path(out_dir) if out_dir else None
    names = list(agent_specs)
    all_results: list[GameResult] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            log_dir = out_dir / f"{na}_vs_{nb}" if out_dir else None
            all_results += benchmark_pair(
                board,
                agent_specs[na],
                agent_specs[nb],
                games=games_per_pair,
                base_seed=base_seed,
                log_dir=log_dir,
            )
    totals = aggregate(all_results)
    elo = compute_elo(all_results)
    summary = {
        "board": board_name,
        "games": len(all_results),
        "agents": {
            name: {
                "games": t.games,
                "wins": t.wins,
                "losses": t.losses,
                "win_rate": round(t.win_rate, 3),
                "avg_medals": round(t.avg_medals, 2),
                "illegal_rate": round(t.illegal_rate, 4),
                "moves": t.stats.moves,
                "prompt_tokens": t.stats.prompt_tokens,
                "completion_tokens": t.stats.completion_tokens,
                "cost_usd": round(t.stats.cost_usd, 4),
                "avg_latency_s": round(t.stats.latency_s / t.stats.moves, 3) if t.stats.moves else 0.0,
                "elo": round(elo.get(name, 1000.0), 1),
            }
            for name, t in totals.items()
        },
    }
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        _write_csv(out_dir / "summary.csv", summary)
    return summary


def _write_csv(path: Path, summary: dict) -> None:
    rows = summary["agents"]
    if not rows:
        return
    fields = ["agent"] + list(next(iter(rows.values())).keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for name, data in rows.items():
            w.writerow({"agent": name, **data})


def format_summary(summary: dict) -> str:
    lines = [f"Benchmark on {summary['board']} — {summary['games']} games"]
    header = f"{'agent':16s} {'games':>5s} {'win%':>6s} {'medals':>7s} {'illegal%':>9s} {'elo':>6s}"
    lines.append(header)
    lines.append("-" * len(header))
    for name, d in sorted(summary["agents"].items(), key=lambda kv: -kv[1]["elo"]):
        lines.append(
            f"{name:16s} {d['games']:5d} {d['win_rate'] * 100:5.1f}% "
            f"{d['avg_medals']:7.2f} {d['illegal_rate'] * 100:8.2f}% {d['elo']:6.0f}"
        )
    return "\n".join(lines)
