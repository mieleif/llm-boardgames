"""Agents: the player abstraction and concrete implementations."""

from .base import Agent, AgentDecision, choose_legal_or_raise
from .random_agent import RandomAgent
from .heuristic_agent import HeuristicAgent
from .human_cli import HumanCliAgent

# Optional LLM agents (require provider SDKs). Imported lazily by the registry.
AGENT_REGISTRY = {
    "random": RandomAgent,
    "heuristic": HeuristicAgent,
    "human": HumanCliAgent,
}


def make_agent(spec: str, color: str, **kwargs) -> Agent:
    """Create an agent from a spec string.

    Examples: ``random``, ``heuristic``, ``human``,
    ``anthropic:claude-opus-4-8:tool``, ``openai:gpt-5:vision``.
    """
    head = spec.split(":", 1)[0]
    if head in AGENT_REGISTRY:
        return AGENT_REGISTRY[head](color=color, **kwargs)
    if head in ("anthropic", "openai", "gemini"):
        from .llm import make_llm_agent

        return make_llm_agent(spec, color=color, **kwargs)
    raise ValueError(f"unknown agent spec: {spec!r}")


__all__ = [
    "Agent",
    "AgentDecision",
    "choose_legal_or_raise",
    "RandomAgent",
    "HeuristicAgent",
    "HumanCliAgent",
    "AGENT_REGISTRY",
    "make_agent",
]
