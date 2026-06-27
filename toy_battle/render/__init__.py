"""Rendering: structured/text observations (tool-use, humans) and PNG (vision)."""

from .text import build_observation, observation_to_text, legal_moves_view

__all__ = ["build_observation", "observation_to_text", "legal_moves_view"]
