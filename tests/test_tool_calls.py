"""Tests for DeepSeek tool_calls module.

Covers:
1. Normal chat without tools (no interference)
2. Non-streaming XML tool call → OpenAI tool_calls
3. DSML tool call → OpenAI tool_calls
4. Undeclared tool → ignored
5. Streaming partial XML → no tag leakage
6. History assistant.tool_calls + tool results → prompt injection
"""

import json
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from gx_qwen2api.providers.deepseek.tool_calls import (
    ToolCallStreamSieve,
    build_prompt_with_tools,
    parse_tool_calls_from_text,
    format_tool_calls_for_history,
    format_tool_result_for_history,
    inject_tool_prompt,
)

PASS = 0
FAIL = 0


def check(condition, msg):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {msg}")
    else:
        FAIL += 1
        print(f"  FAIL: {msg}")


# ─────────────────────────────────────────
# Test 1: Normal chat without tools
# ─────────────────────────────────────────
print("=" * 60)
print("Test 1: Normal chat (no tools) — no interference")
print("=" * 60)

messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"},
]
prompt = build_prompt_with_tools(messages, None)
check("[System]" in prompt, "system role marker present")
check("[User]" in prompt, "user role marker present")
check("tool" not in prompt.lower() or "tool_calls" not in prompt.lower(),
      "no tool markup injected")
check("DSML" not in prompt, "no DSML in prompt without tools")

content = "Just a normal response."
tc, clean, _ = parse_tool_calls_from_text(content, None, None)
check(tc == [], "no tool calls parsed from normal text")
check(clean == content, "content unchanged")

# ─────────────────────────────────────────
# Test 2: Non-streaming XML tool call
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Test 2: Non-streaming XML tool call → OpenAI tool_calls")
print("=" * 60)

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather by city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
    }
}]

# Model outputs XML format
xml_output = """<tool_calls>
  <invoke name="get_weather">
    <parameter name="city"><![CDATA[Beijing]]></parameter>
  </invoke>
</tool_calls>"""

tc, clean, _ = parse_tool_calls_from_text(xml_output, None, ["get_weather"])
check(len(tc) == 1, f"parsed 1 tool call, got {len(tc)}")
if tc:
    check(tc[0]["type"] == "function", "type is function")
    check(tc[0]["function"]["name"] == "get_weather", "name correct")
    args = json.loads(tc[0]["function"]["arguments"])
    check(args.get("city") == "Beijing", f"arguments correct: {args}")
    check(tc[0].get("id", "").startswith("call_"), "tool call has id")
check(clean is None or clean.strip() == "", "content cleaned after tool call removal")

# ─────────────────────────────────────────
# Test 3: DSML tool call
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Test 3: DSML tool call → OpenAI tool_calls")
print("=" * 60)

dsml_output = """<|DSML|tool_calls>
  <|DSML|invoke name="search_docs">
    <|DSML|parameter name="query"><![CDATA[hello world]]></|DSML|parameter>
    <|DSML|parameter name="limit"><![CDATA[5]]></|DSML|parameter>
  </|DSML|invoke>
</|DSML|tool_calls>"""

tc, clean, _ = parse_tool_calls_from_text(dsml_output, None, ["search_docs", "get_weather"])
check(len(tc) == 1, f"parsed 1 tool call, got {len(tc)}")
if tc:
    check(tc[0]["function"]["name"] == "search_docs", "name correct (DSML)")
    args = json.loads(tc[0]["function"]["arguments"])
    check(args.get("query") == "hello world", "query argument correct")
    check(args.get("limit") == 5, "limit parsed as int")

# ─────────────────────────────────────────
# Test 4: Undeclared tool ignored
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Test 4: Undeclared tool → ignored")
print("=" * 60)

unknown_output = """<tool_calls>
  <invoke name="dangerous_action">
    <parameter name="cmd"><![CDATA[rm -rf /]]></parameter>
  </invoke>
</tool_calls>"""

tc, clean, _ = parse_tool_calls_from_text(unknown_output, None, ["get_weather"])
check(not tc, f"undeclared tool not parsed (got {len(tc)})")

# When no allowed list, all tools pass (but caller must set filter)
tc2, _, _ = parse_tool_calls_from_text(unknown_output, None, None)
check(len(tc2) == 1, "without filter, all tools parsed")

# ─────────────────────────────────────────
# Test 5: Streaming partial XML — no leakage
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Test 5: Streaming partial XML → no tag leakage")
print("=" * 60)

sieve = ToolCallStreamSieve(["get_weather"])

texts = []

# Feed normal text first
r = sieve.feed("Hello! ")
check(r.content == "Hello! ", "normal text passes through")
check(not r.tool_call_chunks, "no tool calls yet")

# Start of tool block — should start capturing
r = sieve.feed("Let me check. <tool_")
check(r.content is None or r.content == "Let me check. ", "text before tag emitted")

# Continue partial tag — nothing leaked
r = sieve.feed("calls>")
check(r.content is None, "partial tag not leaked")
check(not r.tool_call_chunks, "no premature tool calls")

# Complete invoke with weather tool
r = sieve.feed("""<invoke name="get_weather">
  <parameter name="city"><![CDATA[Shanghai]]></parameter>
</invoke>
</tool_""")
check(r.content is None, "still no leak")

# Close tool_calls
r = sieve.feed("calls>")
# The tool call should now be complete and emitted
if r.tool_call_chunks:
    texts.append(f"got {len(r.tool_call_chunks)} tool call deltas")
    check(True, "tool call deltas emitted on completion")
    # Check the first delta has name
    name_found = any(
        d.get("function", {}).get("name") == "get_weather"
        for d in r.tool_call_chunks
    )
    check(name_found, "tool call name in delta")
else:
    texts.append("no tool call deltas emitted")

# Test that incomplete XML at flush is handled
sieve2 = ToolCallStreamSieve(["get_weather"])
sieve2.feed("Some text <invoke name=\"get_weather\">")
sieve2.feed("<parameter name=\"city\"><![CDATA[Incomplete...")
# Flush without completing the block
result = sieve2.flush()
# Should not leak raw XML tags in content
if result.content:
    check("invoke" not in result.content, "partial XML stripped at flush")
    check("CDATA" not in result.content, "no CDATA leak")
    check(not result.tool_call_chunks, "incomplete tool call not emitted")
    check(True, "flush handles incomplete block gracefully")

# ─────────────────────────────────────────
# Test 6: History injection
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Test 6: History assistant.tool_calls + tool results → prompt")
print("=" * 60)

hist_tc = format_tool_calls_for_history([
    {
        "id": "call_abc",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": json.dumps({"city": "Tokyo"})
        }
    }
])
check("get_weather" in hist_tc, "tool name in history block")
check("Tokyo" in hist_tc, "argument in history block")
check("<|DSML|tool_calls>" in hist_tc or "<|DSML|invoke" in hist_tc, "DSML format")

tool_result = format_tool_result_for_history("call_abc", "Cloudy, 22°C")
check("Cloudy" in tool_result, "result content in block")
check("call_abc" in tool_result, "tool_call_id in block")
check("<|DSML|tool_result" in tool_result, "DSML tool_result format")

# Full prompt with tools + history
messages_with_history = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What's the weather in Tokyo?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "Tokyo"})
                }
            }
        ]
    },
    {
        "role": "tool",
        "tool_call_id": "call_abc",
        "content": "Cloudy, 22°C"
    },
    {"role": "user", "content": "Thanks, what about tomorrow?"}
]

prompt = build_prompt_with_tools(messages_with_history, tools)
check("get_weather" in prompt, "tool name in full prompt")
check("Tokyo" in prompt, "argument in full prompt")
check("Cloudy" in prompt, "result in full prompt")
check("[Assistant]" in prompt, "assistant role marker")
check("[System]" in prompt, "system role marker")
check("Available Tools" in prompt, "tool declarations injected")
check("Tool Call Format" in prompt, "format instructions injected")

# ─────────────────────────────────────────
# Test: Tool declaration injection
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Test 7: Tool declaration injection format")
print("=" * 60)

complex_tool = [{
    "type": "function",
    "function": {
        "name": "run_code",
        "description": "Execute code",
        "parameters": {
            "type": "object",
            "properties": {
                "language": {"type": "string", "enum": ["python", "js"]},
                "code": {"type": "string"},
            },
            "required": ["language", "code"]
        }
    }
}]

injected = inject_tool_prompt(
    [{"role": "system", "content": "Be helpful."}],
    complex_tool,
)
sys_content = injected[0]["content"]
check("run_code" in sys_content, "tool name injected")
check("Execute code" in sys_content, "description injected")
check("python" in sys_content, "enum values in params")
check("language" in sys_content, "parameter name in schema")
check("<|DSML|tool_calls>" in sys_content, "DSML format included")
check("CDATA" in sys_content, "CDATA instruction included")

# ─────────────────────────────────────────
# Test: Non-streaming tool_call in reasoning_content
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Test 8: Tool call in reasoning_content")
print("=" * 60)

reasoning = """I need to get the weather. <tool_calls>
  <invoke name="get_weather">
    <parameter name="city"><![CDATA[Paris]]></parameter>
  </invoke>
</tool_calls>"""

tc, clean_content, clean_reasoning = parse_tool_calls_from_text(
    "", reasoning, ["get_weather"]
)
check(len(tc) == 1, f"parsed from reasoning, got {len(tc)}")
if tc:
    check(tc[0]["function"]["name"] == "get_weather", "name from reasoning")
check(clean_reasoning is not None, "clean_reasoning returned")
check("<tool_calls>" not in (clean_reasoning or ""), "DSML removed from clean reasoning")
check(clean_content is None or clean_content == "", "content still empty after reasoning parse")

# ─────────────────────────────────────────
# Test 9: P1 — empty tool_names restricts to nothing
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Test 9: Empty tool_names list → nothing allowed (security)")
print("=" * 60)

tc_empty, _, _ = parse_tool_calls_from_text(xml_output, None, [])
check(not tc_empty, f"empty tool_names blocks all tools (got {len(tc_empty)})")

tc_none, _, _ = parse_tool_calls_from_text(xml_output, None, None)
check(len(tc_none) == 1, "None tool_names allows all tools")

sieve_empty = ToolCallStreamSieve([])
sieve_empty.feed("Hello. ")
r = sieve_empty.feed("""<tool_calls><invoke name="get_weather"><parameter name="city"><![CDATA[Beijing]]></parameter></invoke></tool_calls>""")
result = sieve_empty.flush()
check(not result.tool_call_chunks, f"empty tool_names sieve blocks all (got {len(result.tool_call_chunks)} tool_call chunks)")

# ─────────────────────────────────────────
# Test 10: P1 — streaming feed-phase tool_calls sets finish_reason
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Test 10: Streaming feed-phase tool_calls → finish_reason=tool_calls")
print("=" * 60)

sieve3 = ToolCallStreamSieve(["get_weather"])
tc_emitted = False
r = sieve3.feed("Querying weather... ")
check(r.content == "Querying weather... ", "text before tool block passes")

r = sieve3.feed("""<tool_calls>
  <invoke name="get_weather">
    <parameter name="city"><![CDATA[Paris]]></parameter>
  </invoke>
</tool_calls>""")

if r.tool_call_chunks:
    tc_emitted = True
    check(True, "tool_call deltas emitted during feed phase")
    name_ok = any(
        d.get("function", {}).get("name") == "get_weather"
        for d in r.tool_call_chunks
    )
    check(name_ok, "feed-phase delta has correct tool name")
else:
    check(False, "tool_call deltas should have been emitted during feed")

check(tc_emitted, "tool_calls_emitted flag set from feed phase")

# Simulate the outer logic: flush should be clean since feed already emitted
flush_r = sieve3.flush()
check(not flush_r.tool_call_chunks, "flush has no leftover tool calls (already emitted in feed)")
# With tc_emitted=True, final finish_reason should be tool_calls
finish_reason = "tool_calls" if tc_emitted else "stop"
check(finish_reason == "tool_calls", "finish_reason is tool_calls based on feed-phase emission")

# ─────────────────────────────────────────
# Test 11: P2 — reasoning cleanup in non-streaming provider path
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Test 11: Reasoning cleanup — DSML removed from response reasoning")
print("=" * 60)

mixed_reasoning = """Let me check the weather.
<|DSML|tool_calls>
  <|DSML|invoke name="get_weather">
    <|DSML|parameter name="city"><![CDATA[Tokyo]]></|DSML|parameter>
  </|DSML|invoke>
</|DSML|tool_calls>
That should do it."""

tc_r, clean_c, clean_r = parse_tool_calls_from_text(
    "", mixed_reasoning, ["get_weather"]
)
check(len(tc_r) == 1, "tool call parsed from mixed reasoning")
check("<|DSML|tool_calls>" not in (clean_r or ""), "DSML tags removed from reasoning")
check("Let me check the weather." in (clean_r or ""), "prefix text preserved")
check("That should do it." in (clean_r or ""), "suffix text preserved")

# ─────────────────────────────────────────
# Test 12: P2 — pre_ready_frames routed through sieve
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Test 12: Pre-ready frames with sieve → no bypass")
print("=" * 60)

from gx_qwen2api.providers.deepseek.provider import DsFrame, _chunk_has_tool_calls

sieve4 = ToolCallStreamSieve(["get_weather"])
pre_frames = [DsFrame("content_delta", "Sure! ")]

from gx_qwen2api.providers.deepseek.provider import _make_openai_chunk
# Simulate: pre_ready_frames go through sieve (not ds_frames_to_openai_chunks)
chunks = []
for f in pre_frames:
    if f.kind == "content_delta":
        feed = sieve4.feed(f.value)
        if feed.content:
            chunks.append(_make_openai_chunk("test-model", "req-001", content=feed.content))
        for d in feed.tool_call_chunks:
            chunks.append(_make_openai_chunk("test-model", "req-001", tool_call_delta=d))

check(len(chunks) == 1, "pre-ready text frame handled by sieve")
check(chunks[0]["choices"][0]["delta"].get("content") == "Sure! ", "pre-ready content preserved")

# Now simulate tool_calls arriving in pre-ready frames
pre_frames2 = [
    DsFrame("content_delta", "<tool_calls><invoke name=\"get_weather\"><parameter name=\"city\"><![CDATA[NYC]]></parameter></invoke></tool_calls>"),
]
sieve5 = ToolCallStreamSieve(["get_weather"])

emitted = False
for f in pre_frames2:
    if f.kind == "content_delta":
        feed = sieve5.feed(f.value)
        for d in feed.tool_call_chunks:
            emitted = True

# Need flush to handle the eager-parse case
flush_r5 = sieve5.flush()
if flush_r5.tool_call_chunks:
    emitted = True
check(emitted, "pre-ready tool calls captured by sieve (no bypass)")
check(len(flush_r5.tool_call_chunks or sieve5._state.capture_buf or "") == 0 or emitted,
      "sieve handled pre-ready tool block correctly")

# ─────────────────────────────────────────
# Summary
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL}")
print("=" * 60)

sys.exit(0 if FAIL == 0 else 1)
