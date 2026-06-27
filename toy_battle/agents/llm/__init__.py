"""LLM agents: a provider-agnostic base plus thin per-provider wrappers.

Spec strings: ``provider:model[:mode]`` where mode is ``tool`` (default) or
``vision``. Examples::

    anthropic:claude-opus-4-8:tool
    openai:gpt-5:vision
    gemini:gemini-2.5-pro:tool
"""

from __future__ import annotations

from ..base import Agent


def make_llm_agent(spec: str, color: str, **kwargs) -> Agent:
    parts = spec.split(":")
    provider = parts[0]
    model = parts[1] if len(parts) > 1 else None
    mode = parts[2] if len(parts) > 2 else "tool"
    if provider == "anthropic":
        from .anthropic_agent import AnthropicAgent

        return AnthropicAgent(color=color, model=model, mode=mode, **kwargs)
    if provider == "openai":
        from .openai_agent import OpenAIAgent

        return OpenAIAgent(color=color, model=model, mode=mode, **kwargs)
    if provider == "gemini":
        from .gemini_agent import GeminiAgent

        return GeminiAgent(color=color, model=model, mode=mode, **kwargs)
    raise ValueError(f"unknown LLM provider: {provider!r}")


__all__ = ["make_llm_agent"]
