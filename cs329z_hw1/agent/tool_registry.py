"""Problem 5 (tool_registry): Build the toolbox (3 points)"""

from __future__ import annotations

from jsonschema import Draft202012Validator

from cs329z_hw1.types import ToolResult, ToolSpec


class ToolRegistry:
    """
    Registry containing the tools an agent can call, via lookup by exact name.
    """

    tools: dict[str, ToolSpec]  # name -> spec, in registration order

    def __init__(self, tools: list[ToolSpec]) -> None:
        self.tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self.tools.values()
        ]

    def validate(self, name: str, args: dict) -> ToolResult | None:
        """
        Look up the tool and check args against its parameters schema. Return the "unknown_tool"
        or "invalid_args" result, or None if the call can run. It never calls the tool and never
        raises. Use jsonschema.Draft202012Validator for the check.
        """
        if name not in self.tools:
            return ToolResult("unknown_tool", f"There is no tool named {name!r}.")

        validator = Draft202012Validator(self.tools[name].parameters)
        errors = [error.message for error in validator.iter_errors(args)]
        if len(errors) > 0:
            problems = "\n".join(f"- {error}" for error in errors)
            return ToolResult(
                "invalid_args", f"Invalid arguments for {name!r}:\n{problems}"
            )

        return None  # No validation errors, so the call can run.

    def execute(self, name: str, args: dict) -> ToolResult:
        """
        Validate, then run the tool. It always returns a ToolResult and never raises.
        """
        validation_result = self.validate(name, args)
        if validation_result is not None:
            return validation_result

        # No validation errors, so the call can run.
        try:
            value = self.tools[name].fn(**args)
            return ToolResult("ok", str(value))
        except Exception as exc:  # always return a result even if execption
            return ToolResult("tool_error", f"{name} failed: {exc}")
