"""Match running and benchmarking."""

from .runner import GameResult, play_game
from .benchmark import benchmark_pair, run_benchmark

__all__ = ["GameResult", "play_game", "benchmark_pair", "run_benchmark"]
