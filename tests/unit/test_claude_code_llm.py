import json

import pytest
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from adapters.claude_code_llm import ClaudeCodeLlm, build_prompt, parse_result


def req() -> LlmRequest:
    r = LlmRequest()
    r.config = types.GenerateContentConfig(
        system_instruction="Sos operador.",
        tools=[types.Tool(function_declarations=[types.FunctionDeclaration(
            name="restart", description="Reinicia el servicio.",
            parameters=types.Schema(type="OBJECT", properties={"service": types.Schema(type="STRING")},
                                    required=["service"]))])])
    r.contents = [
        types.Content(role="user", parts=[types.Part(text="Incidente: latencia alta.")]),
        types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(
            name="inspect_service", args={"service": "api"}))]),
        types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
            name="inspect_service", response={"finding": "memory_high"}))]),
    ]
    return r


def test_prompt_carries_system_tools_transcript_and_contract():
    p = build_prompt(req())
    assert "Sos operador." in p and "### restart" in p and "Schema de args" in p
    assert "[user] Incidente" in p and "[model→tool] inspect_service" in p
    assert '[tool:inspect_service] {"finding": "memory_high"}' in p
    assert '"kind": "tool"' in p and "Una sola herramienta por turno" in p
    assert "NUNCA respondas kind=text en tu primer turno" in p


def test_parse_tool_and_text():
    r = parse_result('{"kind":"tool","tool":"restart","args":{"service":"api"}}')
    fc = r.get_function_calls()[0]
    assert fc.name == "restart" and fc.args == {"service": "api"}
    r = parse_result('{"kind":"text","text":"listo"}')
    assert r.content.parts[0].text == "listo" and not r.get_function_calls()


@pytest.mark.asyncio
async def test_generate_maps_envelope_usage_and_errors(monkeypatch):
    llm = ClaudeCodeLlm(cli_model="sonnet")
    envelope = {"is_error": False, "result": '{"kind":"tool","tool":"restart","args":{"service":"api"}}',
                "usage": {"input_tokens": 10, "cache_read_input_tokens": 90, "output_tokens": 7}}
    captured = {}

    async def fake_invoke(prompt):
        captured["prompt"] = prompt
        return envelope

    monkeypatch.setattr(llm, "_invoke", fake_invoke)
    out = [r async for r in llm.generate_content_async(req())]
    assert out[0].get_function_calls()[0].name == "restart"
    assert out[0].usage_metadata.total_token_count == 107
    assert "### restart" in captured["prompt"]
    # error del CLI → excepción (el runner la reintenta como transitoria)
    monkeypatch.setattr(llm, "_invoke", lambda p: _err())
    with pytest.raises(RuntimeError):
        [r async for r in llm.generate_content_async(req())]
    # salida no-JSON → texto, no crash
    async def fake_bad(prompt):
        return {"is_error": False, "result": "no soy json", "usage": {}}
    monkeypatch.setattr(llm, "_invoke", fake_bad)
    out = [r async for r in llm.generate_content_async(req())]
    assert out[0].content.parts[0].text == "no soy json"


async def _err():
    return {"is_error": True, "result": "Not logged in", "usage": {}}


def test_cmd_flags():
    llm = ClaudeCodeLlm(cli_model="opus", effort="low")
    cmd = llm._cmd()
    assert cmd[:2] == ["claude", "-p"] and "--model" in cmd and "opus" in cmd
    assert "--tools" in cmd and "" in cmd and "--json-schema" in cmd and "--effort" in cmd
    schema = json.loads(cmd[cmd.index("--json-schema") + 1])
    assert schema["properties"]["kind"]["enum"] == ["tool", "text"]
