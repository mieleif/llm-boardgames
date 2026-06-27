"""Google Gemini provider wrapper (google-genai SDK)."""

from __future__ import annotations

from typing import Optional

from .base_llm import ContentPart, LLMAgent, LLMResponse, Tool


class GeminiAgent(LLMAgent):
    provider = "gemini"

    def default_model(self) -> str:
        return "gemini-2.5-pro"

    def api_key_env(self) -> str:
        return "GEMINI_API_KEY"

    def _client(self):
        from google import genai

        return genai.Client(api_key=self.api_key)

    def _complete(
        self, system: str, content: list[ContentPart], tools: list[Tool], force_tool: Optional[str]
    ) -> LLMResponse:
        from google import genai
        from google.genai import types

        client = self._client()
        parts = []
        for part in content:
            if part.kind == "text":
                parts.append(types.Part.from_text(text=part.text))
            elif part.kind == "image":
                with open(part.image_path, "rb") as fh:
                    parts.append(types.Part.from_bytes(data=fh.read(), mime_type="image/png"))

        function_decls = [
            types.FunctionDeclaration(
                name=t.name, description=t.description, parameters=t.parameters
            )
            for t in tools
        ]
        tool_obj = types.Tool(function_declarations=function_decls)
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            tools=[tool_obj],
        )
        if force_tool:
            config.tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY", allowed_function_names=[force_tool]
                )
            )
        resp = client.models.generate_content(
            model=self.model,
            contents=[types.Content(role="user", parts=parts)],
            config=config,
        )
        tool_name, args, text = None, {}, ""
        for cand in resp.candidates or []:
            for p in cand.content.parts or []:
                if getattr(p, "function_call", None):
                    tool_name = p.function_call.name
                    args = dict(p.function_call.args or {})
                elif getattr(p, "text", None):
                    text += p.text
        usage = getattr(resp, "usage_metadata", None)
        return LLMResponse(
            tool_name=tool_name,
            arguments=args,
            text=text,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            raw=resp,
        )
