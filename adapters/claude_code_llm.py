"""ClaudeCodeLlm: un BaseLlm de ADK que usa Claude Code headless (`claude -p`) como motor.

Corre dentro del plan de Claude del usuario (sin API key ni crédito): cada llamada invoca el
CLI con las herramientas del CLI desactivadas (`--tools ""`) y un contrato de salida validado
por `--json-schema`. Los tool-calls del agente ADK viajan como JSON estructurado y acá se
convierten a `FunctionCall` nativos; el resto del framework (subcortex incluido) no nota la
diferencia. Caveats: consume cuota del plan, agrega segundos de overhead por llamada (spawn del
proceso) y la fidelidad del tool-calling depende del contrato JSON, no del canal nativo.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any, ClassVar

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

log = logging.getLogger("subcortex")

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["tool", "text"]},
        "tool": {"type": "string"},
        "args": {"type": "object"},
        "text": {"type": "string"},
    },
    "required": ["kind"],
    "additionalProperties": False,
}

CONTRACT = """## Cómo responder (obligatorio)
Respondés SIEMPRE con un único JSON que cumple el schema dado.
- Para llamar a una herramienta: {"kind": "tool", "tool": "<nombre>", "args": {...}} con TODOS los
  argumentos requeridos por su schema.
- Para tu respuesta final en texto: {"kind": "text", "text": "..."} — SOLO cuando la tarea ya está
  terminada y ejecutaste las herramientas necesarias. NUNCA respondas kind=text en tu primer turno
  ni sin haber ejecutado al menos una herramienta: trabajás usando herramientas, no opinando.
Una sola herramienta por turno. No inventes herramientas ni argumentos fuera de sus schemas."""

# Variante conversacional (require_action=False): el texto es una respuesta al usuario, no el
# cierre de una tarea. Con el contrato estricto, un agente de chat encadena tools sin hablar nunca.
CONTRACT_RELAXED = """## Cómo responder (obligatorio)
Respondés SIEMPRE con un único JSON que cumple el schema dado.
- Para llamar a una herramienta: {"kind": "tool", "tool": "<nombre>", "args": {...}} con TODOS los
  argumentos requeridos por su schema.
- Para hablarle al usuario (saludar, explicar, resumir lo hecho): {"kind": "text", "text": "..."}.
  Después de ejecutar herramientas, contale qué hiciste y qué viste antes de seguir.
Una sola herramienta por turno. No inventes herramientas ni argumentos fuera de sus schemas."""


def render_tools(llm_request: LlmRequest) -> str:
    decls = []
    for tool in (llm_request.config.tools or []) if llm_request.config else []:
        for fd in getattr(tool, "function_declarations", None) or []:
            schema = fd.parameters.model_dump(exclude_none=True, mode="json") if fd.parameters else {}
            decls.append(f"### {fd.name}\n{fd.description or ''}\nSchema de args: "
                         f"{json.dumps(schema, ensure_ascii=False, sort_keys=True)}")
    return "## Herramientas disponibles\n\n" + "\n\n".join(decls) if decls else ""


def render_transcript(llm_request: LlmRequest) -> str:
    """El historial ADK como transcript de texto: user/model, tool-calls y sus resultados."""
    lines = []
    for content in llm_request.contents or []:
        role = content.role or "user"
        for part in content.parts or []:
            if part.text:
                lines.append(f"[{role}] {part.text}")
            elif part.function_call is not None:
                fc = part.function_call
                lines.append(f"[model→tool] {fc.name}({json.dumps(dict(fc.args or {}), ensure_ascii=False, sort_keys=True)})")
            elif part.function_response is not None:
                fr = part.function_response
                lines.append(f"[tool:{fr.name}] {json.dumps(dict(fr.response or {}), ensure_ascii=False, sort_keys=True)}")
    return "## Conversación hasta ahora\n" + "\n".join(lines) if lines else ""


def build_prompt(llm_request: LlmRequest, contract: str = CONTRACT) -> str:
    system = ""
    if llm_request.config and llm_request.config.system_instruction:
        si = llm_request.config.system_instruction
        system = si if isinstance(si, str) else str(si)
    blocks = [b for b in (f"## Instrucciones del agente\n{system}" if system else "",
                          render_tools(llm_request), render_transcript(llm_request), contract) if b]
    return "\n\n".join(blocks)


def parse_result(payload: str) -> LlmResponse:
    data = json.loads(payload)
    if data.get("kind") == "tool" and data.get("tool"):
        part = types.Part(function_call=types.FunctionCall(name=data["tool"], args=data.get("args") or {}))
    else:
        part = types.Part(text=data.get("text") or "")
    return LlmResponse(content=types.Content(role="model", parts=[part]))


class ClaudeCodeLlm(BaseLlm):
    """Motor de inferencia = Claude Code headless con la suscripción del usuario."""

    model: str = "claude-code"
    cli_model: str = "sonnet"          # sonnet | opus | fable… lo que el plan habilite
    effort: str | None = None          # low | medium | high | xhigh | max
    timeout_s: float = 300.0
    require_action: bool = True        # True: A/B (texto solo al terminar) · False: chat conversacional
    binary: ClassVar[str] = "claude"

    @classmethod
    def supported_models(cls) -> list[str]:
        return ["claude-code", r"claude-code:.*"]

    def _cmd(self) -> list[str]:
        cmd = [self.binary, "-p", "--output-format", "json", "--model", self.cli_model,
               "--no-session-persistence", "--max-turns", "8", "--tools", "",
               "--json-schema", json.dumps(OUTPUT_SCHEMA)]
        if self.effort:
            cmd += ["--effort", self.effort]
        return cmd

    async def _invoke(self, prompt: str) -> dict[str, Any]:
        proc = await asyncio.create_subprocess_exec(
            *self._cmd(), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(prompt.encode()), timeout=self.timeout_s)
        except TimeoutError:
            proc.kill()
            raise RuntimeError(f"claude -p superó el timeout de {self.timeout_s}s") from None
        if proc.returncode != 0 and not out:
            raise RuntimeError(f"claude -p falló (rc={proc.returncode}): {err.decode()[:300]}")
        return json.loads(out.decode())

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False
                                     ) -> AsyncGenerator[LlmResponse, None]:
        contract = CONTRACT if self.require_action else CONTRACT_RELAXED
        envelope = await self._invoke(build_prompt(llm_request, contract))
        if envelope.get("is_error"):
            raise RuntimeError(
                f"claude -p error (subtype={envelope.get('subtype')}, "
                f"api_error={envelope.get('api_error_status')}, "
                f"terminal={envelope.get('terminal_reason')}): "
                f"{str(envelope.get('result'))[:300]}")
        usage = envelope.get("usage") or {}
        try:
            response = parse_result(envelope.get("result") or "")
        except json.JSONDecodeError:
            # Sin JSON válido pese al schema: lo tratamos como texto final, no como crash.
            log.warning("claude-code: salida no-JSON, se devuelve como texto")
            response = LlmResponse(content=types.Content(
                role="model", parts=[types.Part(text=str(envelope.get("result") or ""))]))
        in_tok = (usage.get("input_tokens") or 0) + (usage.get("cache_read_input_tokens") or 0) \
            + (usage.get("cache_creation_input_tokens") or 0)
        out_tok = usage.get("output_tokens") or 0
        response.usage_metadata = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=in_tok, candidates_token_count=out_tok,
            total_token_count=in_tok + out_tok)
        yield response
