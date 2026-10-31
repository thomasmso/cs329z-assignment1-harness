"""Call one Cardinal tool directly, for stepping through it in a debugger.

    uv run python scripts/debug_cardinal_tool.py search_emails '{"query": "collateral letter of credit", "k": 3}'

The tool function is called as-is (not through the registry), so an exception stops in the
debugger instead of turning into a "tool_error". email_qa and daily_digest call the live
grading model through the cached ``LM``.
"""

import json
import sys

from cs329z_hw1.agent.cardinal_tools import cardinal_specs
from cs329z_hw1.llm import LM
from cs329z_hw1.types import AgentConfig

name = sys.argv[1] if len(sys.argv) > 1 else "search_emails"
args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {"query": "collateral letter of credit"}

spec = cardinal_specs(AgentConfig(), LM(tag="debug_cardinal_tool"))[name]
print(f"{name}({args})\n")
print(spec.fn(**args))
