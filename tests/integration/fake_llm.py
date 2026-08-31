from __future__ import annotations

from collections.abc import AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types


def call(name: str, **args) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(name=name, args=args))]))


def text(msg: str) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=msg)]))


class ScriptedLlm(BaseLlm):
    """LLM falso: devuelve respuestas programadas en orden y registra las instrucciones recibidas."""

    model: str = "scripted"
    script: list[LlmResponse] = []
    calls: int = 0
    seen_instructions: list[str] = []

    @classmethod
    def supported_models(cls) -> list[str]:
        return ["scripted"]

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False
                                     ) -> AsyncGenerator[LlmResponse, None]:
        self.calls += 1
        si = llm_request.config.system_instruction if llm_request.config else None
        self.seen_instructions.append(si if isinstance(si, str) else str(si))
        idx = min(self.calls - 1, len(self.script) - 1)
        yield self.script[idx]
