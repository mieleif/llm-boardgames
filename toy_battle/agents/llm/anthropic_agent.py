"""Anthropic (Claude) provider wrapper."""

from __future__ import annotations

import base64
from typing import Optional

from .base_llm import ContentPart, LLMAgent, LLMResponse, Tool


class AnthropicAgent(LLMAgent):
    provider = "anthropic"

    def default_model(self) -> str:
        return "claude-opus-4-8"

    def api_key_env(self) -> str:
        return "ANTHROPIC_API_KEY"

    def _client(self):
        import anthropic

        return anthropic.Anthropic(api_key=self.api_key)

    def _complete(
        self, system: str, content: list[ContentPart], tools: list[Tool], force_tool: Optional[str]
    ) -> LLMResponse:
        client = self._client()
        blocks = []
        for part in content:
            if part.kind == "text":
                blocks.append({"type": "text", "text": part.text})
            elif part.kind == "image":
                with open(part.image_path, "rb") as fh:
                    data = base64.standard_b64encode(fh.read()).decode()
                blocks.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": data},
                    }
                )
        tool_defs = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in tools
        ]
        kwargs = dict(
            model=self.model,
            max_tokens=1024,
            system=system,
            temperature=self.temperature,
            tools=tool_defs,
            messages=[{"role": "user", "content": blocks}],
        )
        if force_tool:
            kwargs["tool_choice"] = {"type": "tool", "name": force_tool}
        msg = client.messages.create(**kwargs)
        tool_name, args, text = None, {}, ""
        for block in msg.content:
            if block.type == "tool_use":
                tool_name, args = block.name, dict(block.input)
            elif block.type == "text":
                text += block.text
        usage = getattr(msg, "usage", None)
        return LLMResponse(
            tool_name=tool_name,
            arguments=args,
            text=text,
            prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
            completion_tokens=getattr(usage, "output_tokens", 0) or 0,
            raw=msg,
        )
