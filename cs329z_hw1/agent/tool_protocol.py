"""Problem 6 (tool_protocol): Design the tool-call protocol (4 points)"""

from __future__ import annotations

import json

from cs329z_hw1.types import ParsedResponse, ToolCall

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"

_DECODER = json.JSONDecoder()


def render_tools(schemas: list[dict]) -> str:
    """
    Text for the system prompt that describes the tools in registry.schemas() and how to call them.

    Ran once to build the system prompt, not ran on every model call.

    Only describes the tools and how to call them so it'll work under any instructions.
    """

    # LIVE_INSTRUCTION will be appended as "Carry out the user's request by calling one of the tools described below."
    prompt_text = ""

    # Tools
    prompt_text += "Tools:\n"
    tools_json = json.dumps(schemas, indent=2)
    prompt_text += f"{tools_json}\n\n"

    # Format instructions on how to call the tools
    prompt_text += """
    How to call a tool:
    
    To call a tool, you must end your reply with exactly one call in this format:
    
    <tool_call>
      {
        "name": "tool_name",
        "args": {"arg_name": "arg_value"}
      }
    </tool_call>
    
    It is a JSON object between the opening <tool_call> and closing </tool_call> tags.
    The JSON object must have a string "name" and an object "args".
    The keys of "args" are the tool's parameter names. Each value must have the JSON type the
    tool's parameters require: numbers without quotes (3.5, not "3.5"), true or false, null,
    and strings in double quotes.
    Tools with no parameters just take an empty JSON object for "args".
    
    If you do not need to call a tool, just reply with the text you want to say.
    """

    # Example
    prompt_text += """
    Example:
    
    Adding the two numbers.
    <tool_call>
    {"name": "add", "args": {"a": 17, "b": 25}}
    </tool_call>
    """
    return prompt_text


def format_tool_call(text: str, call: ToolCall) -> str:
    """
    Writes a reply containing text and call in your format.
    Parsing it must give back the same name and args, with their JSON types.

    Example:

    Adding the two numbers.
    <tool_call>
    {"name": "add", "args": {"a": 17, "b": 25}}
    </tool_call>
    """
    call_json = json.dumps({"name": call.name, "args": call.args})
    formatted_text = f"""
{text.strip()}\n\n
{TOOL_CALL_START}
{call_json}
{TOOL_CALL_END}
    """
    return formatted_text


def parse_response(text: str) -> ParsedResponse:
    """
    Returns the reply's text (the reply with the call removed and surrounding whitespace stripped)
    and its tool_call (a ToolCall with a name and an args dict, or None). A reply may contain at most
    one call. For a reply with several calls, or a call you cannot parse, set error and
    return no call. Never raise.
    """

    # Find tool call start index
    text = text.strip()
    tool_call_start_index = text.find(TOOL_CALL_START)

    # If no tool call, just return the text
    if tool_call_start_index == -1:
        return ParsedResponse(text=text)

    # Get text before tool call block
    text_before_tool_call = text[:tool_call_start_index].strip()
    text_after_start_tag = text[tool_call_start_index + len(TOOL_CALL_START) :].strip()
    try:
        # raw_decode automatically finds the end of the JSON obj
        tool_call_json, json_end_index = _DECODER.raw_decode(text_after_start_tag)
    except (ValueError, RecursionError):
        return ParsedResponse(
            text=text_before_tool_call,
            error="the JSON after <tool_call> is incomplete or invalid.",
        )

    text_after_json = text_after_start_tag[json_end_index:].strip()

    # Check that there is a closing </tool_call> tag
    if not text_after_json.startswith(TOOL_CALL_END):
        return ParsedResponse(
            text=text_before_tool_call,
            error="the call is missing its closing </tool_call>.",
        )

    text_after_end_tag = text_after_json[len(TOOL_CALL_END) :].strip()

    # Check that there are no more tool calls
    # (theoretically, there is still an edge case where model's reply references the "<tool_call>"
    # string without intention of calling a tool)
    if TOOL_CALL_START in text_after_end_tag:
        return ParsedResponse(
            text=text_before_tool_call,
            error="the reply contains more than one tool call; make one call per reply.",
        )

    # Paranoia type checks that we have valid JSON, since raw_decode can parse out any JSON valid types such as strings, numbers, etc.
    if not isinstance(tool_call_json, dict) or not isinstance(
        tool_call_json.get("name"), str
    ):
        return ParsedResponse(
            text=text_before_tool_call,
            error='the call must be a JSON object with a string "name".',
        )
    args = tool_call_json.get("args", {})
    if not isinstance(args, dict):
        return ParsedResponse(
            text=text_before_tool_call,
            error='"args" must be a JSON object.',
        )

    return ParsedResponse(
        text=f"{text_before_tool_call}\n{text_after_end_tag}".strip(),
        tool_call=ToolCall(name=tool_call_json["name"], args=args),
    )
