"""OpenAI (ChatGPT) provider wrapper."""

from __future__ import annotations

import base64
import json
from typing import Optional

from .base_llm import ContentPart, LLMAgent, LLMResponse, Tool


class OpenAIAgent(LLMAgent):
    provider = "openai"

    def default_model(self) -> str:
        return "gpt-4o"

    def api_key_env(self) -> str:
        return "OPENAI_API_KEY"

    def _client(self):
        import openai

        return openai.OpenAI(api_key=self.api_key)

    def _complete(
        self, system: str, content: list[ContentPart], tools: list[Tool], force_tool: Optional[str]
    ) -> LLMResponse:
        client = self._client()
        user_parts = []
        for part in content:
            if part.kind == "text":
                user_parts.append({"type": "text", "text": part.text})
            elif part.kind == "image":
                with open(part.image_path, "rb") as fh:
                    data = base64.standard_b64encode(fh.read()).decode()
                user_parts.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}
                )
        tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]
        kwargs = dict(
            model=self.model,
            temperature=self.temperature,
            tools=tool_defs,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_parts},
            ],
        )
        if force_tool:
            kwargs["tool_choice"] = {"type": "function", "function": {"name": force_tool}}
        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message
        tool_name, args, text = None, {}, choice.content or ""
        if choice.tool_calls:
            call = choice.tool_calls[0]
            tool_name = call.function.name
            try:
                args = json.loads(call.function.arguments)
            except (TypeError, ValueError):
                args = {}
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            tool_name=tool_name,
            arguments=args,
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            raw=resp,
        )
