"""Tool call support for DeepSeek provider.

Converts OpenAI tools/tool_calls to/from DSML/XML prompt format.
Handles non-streaming parsing, streaming sieve, and prompt injection.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger("gx_qwen2api.deepseek.tool_calls")


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

_DSML_PREFIX = "<|DSML|"
_DSML_CLOSE_PREFIX = "</|DSML|"

_CDATA_STRING_PARAMS = frozenset({
    "content", "file_content", "text", "prompt", "query",
    "command", "cmd", "script", "code", "old_string", "new_string",
    "pattern", "path", "file_path",
})


# ──────────────────────────────────────────────────────────────────────
# Prompt construction
# ──────────────────────────────────────────────────────────────────────

def _render_parameter_xml(name: str, value: Any) -> str:
    """Render a parameter value into DSML/XML parameter tags.

    Strings are wrapped in CDATA. Dicts/lists produce nested XML.
    """
    if isinstance(value, str):
        safe = _cdata_escape(value)
        return f'<|DSML|parameter name="{_xml_escape_attr(name)}"><![CDATA[{safe}]]></|DSML|parameter>'
    if isinstance(value, (int, float, bool)):
        return f'<|DSML|parameter name="{_xml_escape_attr(name)}"><![CDATA[{json.dumps(value)}]]></|DSML|parameter>'
    if isinstance(value, (dict, list)):
        inner = _render_value_xml(value)
        return f'<|DSML|parameter name="{_xml_escape_attr(name)}">{inner}</|DSML|parameter>'
    return f'<|DSML|parameter name="{_xml_escape_attr(name)}"><![CDATA[{_cdata_escape(str(value))}]]></|DSML|parameter>'


def _render_value_xml(value: Any, tag_hint: str = "item") -> str:
    """Render a JSON value as nested XML elements (no CDATA for structures)."""
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                parts.append(f"<{_xml_escape_attr(k)}>{_render_value_xml(v, 'item')}</{_xml_escape_attr(k)}>")
            elif isinstance(v, str):
                safe = _cdata_escape(v)
                parts.append(f"<{_xml_escape_attr(k)}><![CDATA[{safe}]]></{_xml_escape_attr(k)}>")
            elif isinstance(v, bool):
                parts.append(f"<{_xml_escape_attr(k)}><![CDATA[{'true' if v else 'false'}]]></{_xml_escape_attr(k)}>")
            elif v is None:
                parts.append(f"<{_xml_escape_attr(k)}><![CDATA[null]]></{_xml_escape_attr(k)}>")
            else:
                parts.append(f"<{_xml_escape_attr(k)}><![CDATA[{_cdata_escape(str(v))}]]></{_xml_escape_attr(k)}>")
        return "".join(parts)
    if isinstance(value, list):
        tag = _xml_escape_attr(tag_hint)
        parts = [f"<{tag}>{_render_value_xml(item, 'item')}</{tag}>" for item in value]
        return "".join(parts)
    if isinstance(value, str):
        return f"<![CDATA[{_cdata_escape(value)}]]>"
    if isinstance(value, bool):
        return f"<![CDATA[{'true' if value else 'false'}]]>"
    if value is None:
        return "<![CDATA[null]]>"
    return f"<![CDATA[{_cdata_escape(str(value))}]]>"


def _cdata_escape(text: str) -> str:
    """Escape text for CDATA by splitting ]]&gt; sequences."""
    return text.replace("]]>", "]]]]><![CDATA[>")


def _xml_escape_attr(text: str) -> str:
    """Minimal XML attribute escaping."""
    text = text.replace("&", "&amp;")
    text = text.replace('"', "&quot;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


def _json_schema_to_text(schema: dict[str, Any]) -> str:
    """Convert a JSON Schema object to a readable text representation."""
    return json.dumps(schema, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────────────
# Tool declaration injection
# ──────────────────────────────────────────────────────────────────────

def build_tool_instructions(tool_names: list[str]) -> str:
    """Build the DSML tool call format instructions for the prompt."""
    if not tool_names:
        return ""

    lines = [
        "# Tool Call Format",
        "",
        "When you need to call a tool, output ONLY a tool call block using the following format:",
        "",
        "<|DSML|tool_calls>",
        '  <|DSML|invoke name="tool_name">',
        '    <|DSML|parameter name="param1"><![CDATA[string value]]></|DSML|parameter>',
        '    <|DSML|parameter name="param2"><![CDATA[{"key": "value"}]]></|DSML|parameter>',
        "  </|DSML|invoke>",
        "</|DSML|tool_calls>",
        "",
        "Rules:",
        "- Wrap ALL string values in <![CDATA[...]]> to avoid XML escaping issues.",
        "- For nested objects/arrays, use nested XML elements inside parameters.",
        "- Do NOT output any text before or after the tool call block.",
        "- Do NOT wrap the tool call block in markdown code fences (```).",
        "- Only call tools that are listed above. If no tool is needed, respond normally.",
        "- You may call multiple tools in parallel by including multiple <invoke> blocks.",
    ]
    return "\n".join(lines)


def inject_tool_prompt(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Inject tool declarations and format instructions into the message list.

    Returns a new list of messages with tool information injected into
    the system message (or prepended as a new system message).
    """
    if not tools:
        return messages

    tool_names: list[str] = []
    tool_descs: list[str] = []

    for tool in tools:
        fn = tool.get("function", {})
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        if not name:
            continue
        tool_names.append(name)
        parts = [f"Tool: {name}"]
        if desc:
            parts.append(f"Description: {desc}")
        if params:
            parts.append(f"Parameters: {_json_schema_to_text(params)}")
        tool_descs.append("\n".join(parts))

    if not tool_names:
        return messages

    tool_block = "# Available Tools\n\n" + "\n\n".join(tool_descs)
    instructions = build_tool_instructions(tool_names)
    full_tool_text = tool_block + "\n\n" + instructions

    result = list(messages)

    system_idx = next((i for i, m in enumerate(result) if m.get("role") == "system"), None)
    if system_idx is not None:
        orig_content = result[system_idx].get("content", "")
        if isinstance(orig_content, str):
            result[system_idx] = {
                **result[system_idx],
                "content": orig_content + "\n\n" + full_tool_text,
            }
        else:
            result[system_idx] = {
                **result[system_idx],
                "content": full_tool_text,
            }
    else:
        result.insert(0, {"role": "system", "content": full_tool_text})

    return result


# ──────────────────────────────────────────────────────────────────────
# History conversion: assistant tool_calls → DSML block
# ──────────────────────────────────────────────────────────────────────

def format_tool_calls_for_history(tool_calls: list[dict[str, Any]]) -> str:
    """Convert assistant message tool_calls to a DSML/XML block for prompt history."""
    if not tool_calls:
        return ""

    parts = [_DSML_PREFIX + "tool_calls>"]
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        args_str = fn.get("arguments", "{}")
        if isinstance(args_str, str):
            try:
                args = json.loads(args_str)
            except Exception:
                args = {}
        else:
            args = args_str if isinstance(args_str, dict) else {}

        parts.append(f'  {_DSML_PREFIX}invoke name="{_xml_escape_attr(name)}">')
        if isinstance(args, dict):
            for k, v in args.items():
                parts.append("    " + _render_parameter_xml(k, v))
        parts.append(f"  {_DSML_CLOSE_PREFIX}invoke>")
    parts.append(_DSML_CLOSE_PREFIX + "tool_calls>")
    return "\n".join(parts)


def format_tool_result_for_history(tool_call_id: str, content: Any) -> str:
    """Convert a tool/function result to a DSML result block for prompt history.

    Format: <|DSML|tool_result tool_call_id="...">
              <![CDATA[result_content]]>
            </|DSML|tool_result>
    """
    content_str = ""
    if isinstance(content, str):
        content_str = content
    elif content is not None:
        try:
            content_str = json.dumps(content, ensure_ascii=False)
        except Exception:
            content_str = str(content)

    safe = _cdata_escape(content_str)
    return (
        f'{_DSML_PREFIX}tool_result tool_call_id="{_xml_escape_attr(tool_call_id)}">'
        f"<![CDATA[{safe}]]>"
        f"{_DSML_CLOSE_PREFIX}tool_result>"
    )


def build_prompt_with_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> str:
    """Build the full prompt text including tool declarations and history.

    Handles:
    - System, user, assistant messages as [Role]\ncontent
    - Assistant messages with tool_calls: converts to DSML invoke block
    - Tool messages: converts to DSML tool_result block
    - Tool declarations injected into system message
    """
    enhanced = list(messages) if messages else []
    if tools:
        enhanced = inject_tool_prompt(enhanced, tools)

    parts: list[str] = []
    for msg in enhanced:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            parts.append(f"[System]\n{content}")
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                tc_block = format_tool_calls_for_history(tool_calls)
                if content:
                    parts.append(f"[Assistant]\n{content}\n\n{tc_block}")
                else:
                    parts.append(f"[Assistant]\n{tc_block}")
            else:
                parts.append(f"[Assistant]\n{content}")
            continue

        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            tc_block = format_tool_result_for_history(tool_call_id, content)
            parts.append(tc_block)
            continue

        if role == "user":
            parts.append(f"[User]\n{content}")
            continue

        parts.append(f"[{role}]\n{content}")

    return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────────────
# Non-streaming tool call parser
# ──────────────────────────────────────────────────────────────────────

_INVOKE_PATTERN = re.compile(
    r'<\s*(?:\|?\s*DSML\s*\|?\s*)?invoke\s+name\s*=\s*"([^"]*)"\s*>'
    r'(.*?)'
    r'<\s*/\s*(?:\|?\s*DSML\s*\|?\s*)?invoke\s*>',
    re.DOTALL | re.IGNORECASE,
)

_PARAM_PATTERN = re.compile(
    r'<\s*'
    r'(?:\|?\s*DSML\s*\|?\s*)?'
    r'parameter\s+name\s*=\s*"([^"]*)"\s*>'
    r'(.*?)'
    r'<\s*/\s*'
    r'(?:\|?\s*DSML\s*\|?\s*)?'
    r'parameter\s*>',
    re.DOTALL | re.IGNORECASE,
)

_CDATA_INNER_PATTERN = re.compile(
    r'^\s*<!\[CDATA\[(.*)\]\]>\s*$',
    re.DOTALL,
)


def _extract_tool_calls_block(text: str) -> str | None:
    """Extract the tool_calls block from text (supports DSML and plain XML)."""
    pat = re.compile(
        r'<\s*(?:\|?\s*DSML\s*\|?\s*)?tool_calls\s*>'
        r'(.*?)'
        r'<\s*/\s*(?:\|?\s*DSML\s*\|?\s*)?tool_calls\s*>',
        re.DOTALL | re.IGNORECASE,
    )
    m = pat.search(text)
    if m:
        return m.group(1)
    return None


def _parse_tool_calls_from_block(block_text: str, allowed_names: set[str] | None) -> list[dict[str, Any]]:
    """Parse invoke blocks from a tool_calls block body."""
    tool_calls: list[dict[str, Any]] = []
    for invoke_m in _INVOKE_PATTERN.finditer(block_text):
        name = invoke_m.group(1)
        body = invoke_m.group(2)

        if allowed_names is not None and name not in allowed_names:
            continue

        args = _parse_invoke_body(body)
        tool_calls.append({
            "id": _make_tool_call_id(),
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })
    return tool_calls


def _parse_invoke_body(body: str) -> dict[str, Any]:
    """Parse parameter blocks from an invoke body into a dict."""
    result: dict[str, Any] = {}
    for pm in _PARAM_PATTERN.finditer(body):
        pname = pm.group(1)
        raw = pm.group(2).strip()

        value = _parse_param_value(pname, raw)
        result[pname] = value
    return result


def _parse_param_value(name: str, raw: str) -> Any:
    """Parse a parameter value from its raw XML body.

    Tries CDATA extraction first, then JSON parsing for structured data.
    """
    cdata_m = _CDATA_INNER_PATTERN.match(raw)
    if cdata_m:
        inner = cdata_m.group(1)
        if name in _CDATA_STRING_PARAMS:
            return inner
        return _try_parse_json_literal(inner)

    inner = _xml_unescape(raw)
    if not inner.strip():
        return inner

    if "<" in inner and ">" in inner:
        return _try_parse_nested_xml(inner)

    return _try_parse_json_literal(inner)


def _try_parse_json_literal(text: str) -> Any:
    """Try to parse text as a JSON literal. Falls back to string."""
    text = text.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


def _try_parse_nested_xml(text: str) -> Any:
    """Try to parse structured nested XML into a dict or list."""
    text = text.strip()
    if text.startswith("<item>") or text.startswith("<item "):
        items: list[Any] = []
        item_pat = re.compile(r'<item\b[^>]*>(.*?)</item\s*>', re.DOTALL | re.IGNORECASE)
        for m in item_pat.finditer(text):
            items.append(_parse_param_value("item", m.group(1)))
        return items
    result: dict[str, Any] = {}
    el_pat = re.compile(r'<([a-zA-Z_][a-zA-Z0-9_.:-]*)\b[^>]*>(.*?)</\1\s*>', re.DOTALL)
    for m in el_pat.finditer(text):
        key = m.group(1)
        val = _parse_param_value(key, m.group(2))
        if key in result:
            existing = result[key]
            if isinstance(existing, list):
                existing.append(val)
            else:
                result[key] = [existing, val]
        else:
            result[key] = val
    return result if result else text


def _xml_unescape(text: str) -> str:
    """Unescape common XML entities."""
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&apos;", "'")
    text = text.replace("&#39;", "'")
    text = text.replace("&amp;", "&")
    return text


_tc_counter = 0


def _make_tool_call_id() -> str:
    global _tc_counter
    _tc_counter += 1
    return f"call_{_tc_counter:08x}"


def parse_tool_calls_from_text(
    content: str,
    reasoning_content: str | None = None,
    allowed_tool_names: list[str] | None = None,
) -> tuple[list[dict[str, Any]], str, str | None]:
    """Parse tool calls from model output text.

    Searches both content and reasoning_content for DSML/XML tool_calls blocks.
    Returns (tool_calls, clean_content, clean_reasoning).
    Clean content has the tool call block removed.
    """
    allowed: set[str] | None = set(allowed_tool_names) if allowed_tool_names is not None else None
    tool_calls: list[dict[str, Any]] = []
    clean_content = content
    clean_reasoning = reasoning_content

    sources = [
        ("content", content),
        ("reasoning", reasoning_content or ""),
    ]

    for src_name, text in sources:
        if not text:
            continue
        block = _extract_tool_calls_block(text)
        if block:
            tc = _parse_tool_calls_from_block(block, allowed)
            tool_calls.extend(tc)

            # Remove tool_calls block from the source text
            removal_re = re.compile(
                r'\s*<\s*(?:\|?\s*DSML\s*\|?\s*)?tool_calls\s*>.*?<\s*/\s*(?:\|?\s*DSML\s*\|?\s*)?tool_calls\s*>\s*',
                re.DOTALL | re.IGNORECASE,
            )
            cleaned = removal_re.sub("", text).strip()
            if src_name == "content":
                clean_content = cleaned or None
            elif src_name == "reasoning":
                clean_reasoning = cleaned or None

    return tool_calls, clean_content, clean_reasoning


# ──────────────────────────────────────────────────────────────────────
# Streaming sieve
# ──────────────────────────────────────────────────────────────────────

_TOOL_TAG_START_RE = re.compile(
    r'<\s*'
    r'(?:'
    r'\|?\s*DSML\s*\|?\s*(?:tool_calls|invoke|parameter)'
    r'|'
    r'(?:tool_calls|invoke)\b'
    r')'
    r'(?:\s|>|/|$|$)',
    re.IGNORECASE,
)

_TOOL_CALLS_CLOSE_FULL_RE = re.compile(r'<\s*/\s*(?:\|?\s*DSML\s*\|?\s*)?tool_calls\s*>', re.IGNORECASE)
_INVOKE_CLOSE_FULL_RE = re.compile(r'<\s*/\s*(?:\|?\s*DSML\s*\|?\s*)?invoke\s*>', re.IGNORECASE)
_PARTIAL_TAG_RE = re.compile(r'<\s*/?\s*\|?\s*D?S?M?L?\s*\|?[a-z_]*$', re.IGNORECASE)


class _StreamSieveState:
    """State for the streaming tool call sieve."""

    __slots__ = (
        "capturing", "capture_buf", "pending_buf",
        "text_mode", "code_fence_depth",
        "tool_call_delta_index", "emitted_text",
    )

    def __init__(self) -> None:
        self.capturing = False
        self.capture_buf: str = ""
        self.pending_buf: str = ""
        self.text_mode = True
        self.code_fence_depth = 0
        self.tool_call_delta_index = 0
        self.emitted_text = ""


class ProcessResult:
    """Result of processing a chunk through the sieve."""

    __slots__ = ("content", "tool_call_chunks")

    def __init__(self, content: str | None = None, tool_call_chunks: list[dict[str, Any]] | None = None) -> None:
        self.content = content
        self.tool_call_chunks = tool_call_chunks or []


class ToolCallStreamSieve:
    """Streaming sieve that buffers DSML/XML tool call blocks and emits OpenAI format.

    Usage:
        sieve = ToolCallStreamSieve(tool_names=["my_tool"])
        for chunk in content_chunks:
            result = sieve.feed(chunk)
            yield result.content          # normal text
            yield result.tool_call_chunks  # structured tool call deltas
        final = sieve.flush()
        # handle final content or unreleased buffer
    """

    def __init__(self, tool_names: list[str] | None = None) -> None:
        self._state = _StreamSieveState()
        self._allowed: set[str] | None = set(tool_names) if tool_names is not None else None

    def feed(self, chunk: str) -> ProcessResult:
        """Feed a text chunk to the sieve. Returns ProcessResult."""
        self._state.pending_buf += chunk

        if self._state.capturing:
            return self._handle_capturing()

        text_result = self._handle_text()

        if self._state.capturing:
            cap_result = self._handle_capturing()
            return _merge_process_results(text_result, cap_result)

        return text_result

    def flush(self) -> ProcessResult:
        """Flush remaining buffered content."""
        st = self._state
        result = ProcessResult()

        if st.capturing:
            st.capture_buf += st.pending_buf
            st.pending_buf = ""

            block = _extract_tool_calls_block(st.capture_buf)
            if block:
                tool_calls = _parse_tool_calls_from_block(block, self._allowed)
                if tool_calls:
                    result.tool_call_chunks = _make_stream_tool_call_deltas(
                        tool_calls, st.tool_call_delta_index
                    )
            else:
                result.content = _strip_partial_tool_xml(st.capture_buf)

            st.capturing = False
            st.capture_buf = ""
        else:
            if st.pending_buf:
                result.content = st.pending_buf
                st.pending_buf = ""

        return result

    def _handle_text(self) -> ProcessResult:
        st = self._state
        combined = st.pending_buf
        st.pending_buf = ""

        match = _TOOL_TAG_START_RE.search(combined)
        if match:
            before = combined[:match.start()]
            rest = combined[match.start():]
            st.capturing = True
            st.capture_buf = rest
            return ProcessResult(content=before if before else None)

        # No full tag start found, but check for partial tag at end
        split_point = _find_partial_tag_start(combined)
        if split_point >= 0:
            safe = combined[:split_point]
            rest = combined[split_point:]
            st.pending_buf = rest
            return ProcessResult(content=safe if safe else None)

        return ProcessResult(content=combined)

    def _handle_capturing(self) -> ProcessResult:
        st = self._state
        st.capture_buf += st.pending_buf
        st.pending_buf = ""

        block = _extract_tool_calls_block(st.capture_buf)
        if not block:
            return ProcessResult()

        tool_calls = _parse_tool_calls_from_block(block, self._allowed)

        st.capturing = False
        st.capture_buf = ""

        if tool_calls:
            deltas = _make_stream_tool_call_deltas(tool_calls, st.tool_call_delta_index)
            return ProcessResult(tool_call_chunks=deltas)
        return ProcessResult()

    @property
    def is_capturing(self) -> bool:
        return self._state.capturing


def _make_stream_tool_call_deltas(
    tool_calls: list[dict[str, Any]],
    start_index: int,
) -> list[dict[str, Any]]:
    """Convert parsed tool calls to OpenAI streaming delta format."""
    deltas: list[dict[str, Any]] = []
    for i, tc in enumerate(tool_calls):
        idx = start_index + i
        fn = tc["function"]
        fn_name = fn.get("name", "")
        fn_args = fn.get("arguments", "")

        deltas.append({
            "index": idx,
            "id": tc.get("id", ""),
            "type": "function",
            "function": {"name": fn_name, "arguments": ""},
        })
        deltas.append({
            "index": idx,
            "function": {"arguments": fn_args},
        })
    return deltas


def _merge_process_results(first: ProcessResult, second: ProcessResult) -> ProcessResult:
    """Merge two ProcessResults, concatenating content and extending tool_call_chunks."""
    content = first.content or ""
    content += second.content or ""
    tc = list(first.tool_call_chunks)
    tc.extend(second.tool_call_chunks)
    return ProcessResult(
        content=content if content else None,
        tool_call_chunks=tc if tc else None,
    )


def _strip_partial_tool_xml(text: str) -> str:
    """Best-effort stripping of incomplete tool tags from buffered text.

    If we have a partial tool_calls or invoke tag without a complete close,
    we try to remove the incomplete portion to avoid leaking XML to the client.
    """
    idx_open = text.find("<")
    if idx_open < 0:
        return text

    safe = text[:idx_open].rstrip()
    return safe if safe else ""


def _has_complete_tool_calls(text: str) -> bool:
    """Check if text contains a complete tool_calls block."""
    block = _extract_tool_calls_block(text)
    return block is not None


# ──────────────────────────────────────────────────────────────────────
# Utility: check if text might contain a tool call
# ──────────────────────────────────────────────────────────────────────

def might_contain_tool_call(text: str) -> bool:
    """Quick check if text might contain DSML/XML tool call markup."""
    return bool(_TOOL_TAG_START_RE.search(text))


def _find_partial_tag_start(text: str) -> int:
    """Find the position where a partial tool tag might start at the end of text.

    Returns the index of the first `<` that might be the start of a tool tag,
    or -1 if no partial tag is detected.
    """
    last_lt = text.rfind("<")
    if last_lt < 0:
        return -1

    suffix = text[last_lt:]
    partial = _PARTIAL_TAG_RE.search(suffix)
    if partial:
        return last_lt
    return -1
