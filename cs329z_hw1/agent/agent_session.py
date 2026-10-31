"""Problem 7 (agent_loop): Implement the tool-calling loop (8 points)"""

from __future__ import annotations

from typing import Optional

from cs329z_hw1.agent.cardinal_tools import cardinal_specs
from cs329z_hw1.agent.tool_protocol import parse_response, render_tools
from cs329z_hw1.agent.tool_registry import INVALID_ARGS, UNKNOWN_TOOL, ToolRegistry
from cs329z_hw1.llm import OutputTruncated
from cs329z_hw1.tokens import count_message_tokens, count_tokens
from cs329z_hw1.types import AgentConfig, AgentResult, LMCallable, ToolSpec

MAX_MALFORMED_REPLIES_IN_A_ROW = 3
MALFORMED_STATUSES = (UNKNOWN_TOOL, INVALID_ARGS)  # the model's fault, not the tool's

SYSTEM_PROMPT = """You answer the user's request by calling tools when necessary.

Use a tool when you need to take an action to answer the user's request.

After a tool call, you will see its result - use it to decide the next step.

When you are finished, reply to the user with no tool call.
"""


class AgentSession:
    """An agent session represents one conversation with the agent."""

    # Parent Objects
    lm: LMCallable  # the model for the agent's own turns
    tool_registry: ToolRegistry
    config: AgentConfig
    mode: str  # "auto" or "confirm"

    # Controlled Objects
    transcript: list[dict]  # append-only event log
    messages: list[dict]  # conversation lives through sessions as part of the context

    def __init__(
        self, lm: LMCallable, tools: Optional[list[ToolSpec]], config: AgentConfig
    ) -> None:
        self.lm = lm
        self.config = config
        self.mode = config.mode
        self.transcript = []
        self.tool_registry = ToolRegistry(self._select_tools(tools))

        self.messages = []

        # Initial system prompt
        self._add_to_messages(
            "system", f"{SYSTEM_PROMPT}\n\n{render_tools(self.tool_registry.schemas())}"
        )

    def send(self, user_message: str) -> AgentResult:
        """
        Runs the loop for one user message and returns the reply `text`, a `status`,
        and the transcript. Each send continues the same conversation.
        """

        # Add the user message to the transcript
        self._add_to_transcript("user", content=user_message)

        # Add user message to messages
        self._add_to_messages("user", user_message)

        malformed_replies_in_a_row, turns, message_token_count = 0, 0, 0
        while True:
            # if this send has made ``config.max_turns`` model calls, stop with status "max_turns"
            if turns >= self.config.max_turns:
                return AgentResult(
                    text=f"SESSION STOPPED: The model used all {self.config.max_turns} turns.",
                    status="max_turns",
                    transcript=list(self.transcript),
                )
            # Check the two budgets before each model call, so a final answer on the call that crosses
            # the token budget is still done.
            if message_token_count >= self.config.token_budget:
                return AgentResult(
                    text=f"SESSION STOPPED: The model used its budget of {self.config.token_budget} tokens without finishing.",
                    status="token_budget",
                    transcript=list(self.transcript),
                )
            turns += 1
            message_token_count += count_message_tokens(self.messages)

            # Call the model
            try:
                reply = self.lm(self.messages)
            # Handle the model raising OutputTruncated
            except OutputTruncated as exc:
                message_token_count += count_tokens(exc.text)
                self._add_to_transcript("assistant", content=exc.text)
                # NOTE: may be empty if hidden reasoning but I wanted to add it to give model more context
                self._add_to_messages("assistant", exc.text)
                error = "Your above previous reply was truncated due to limit length. Reply again, but more briefly."
            else:
                message_token_count += count_tokens(reply)
                # Add raw model response to transcript and messages
                self._add_to_transcript("assistant", content=reply)
                self._add_to_messages("assistant", reply)
                parsed = parse_response(reply)
                error = None
                if parsed.error is not None:
                    error = f"Your tool call is invalid due to error: {parsed.error}. Reply again."
                # handle hiccup case when testing live where model replied with empty string and we thought that was answer... should be considered malformed
                elif parsed.tool_call is None and parsed.text == "":
                    error = "Your reply was empty. Reply again."

            if error is None:
                # No tool call means the model is done
                if parsed.tool_call is None:
                    # Must make shallow copy of transcript to pass mutability test
                    return AgentResult(
                        text=parsed.text,
                        status="done",
                        transcript=list(self.transcript),
                    )

                # Else we need to run the tool and log to transcript and messages
                name, args = parsed.tool_call.name, parsed.tool_call.args
                self._add_to_transcript("tool_call", name=name, args=args)
                result = self.tool_registry.execute(name, args)
                self._add_to_transcript(
                    "tool_result",
                    name=name,
                    status=result.status,
                    content=result.content,
                )
                self._add_to_messages(
                    "user",
                    f"Result of tool call for {name} has status {result.status} and content: {result.content}",
                )
                if result.status not in MALFORMED_STATUSES:
                    malformed_replies_in_a_row = 0
                    continue  # the loop to call the model again
                # else: # handle malformed statuses outside of if-else statement...
            else:
                # Unreadable reply: log the error and show it to the model
                self._add_to_transcript("error", content=error)
                self._add_to_messages("user", error)
                # handle malformed statuses below...

            malformed_replies_in_a_row += 1
            if malformed_replies_in_a_row == MAX_MALFORMED_REPLIES_IN_A_ROW:
                return AgentResult(
                    text=f"SESSION STOPPED: The model made {MAX_MALFORMED_REPLIES_IN_A_ROW} malformed replies in a row.",
                    status="malformed",
                    transcript=list(self.transcript),
                )

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def _select_tools(self, tools: Optional[list[ToolSpec]]) -> list[ToolSpec]:
        """
        Return a list of ToolSpecs depending on whether explicit list of tools were provided.

        If None, return the whole Cardinal toolset.

        Explicit lists get ask_user and remember/recall depending on config.
        """
        cardinal = cardinal_specs(self.config, self.config.aux_lm or self.lm)
        if tools is None:
            return list(cardinal.values())

        selected = list(tools)
        if self.config.user is not None:
            selected.append(cardinal["ask_user"])
        if self.config.memory_dir is not None:
            selected.extend([cardinal["remember"], cardinal["recall"]])

        return selected

    def _add_to_transcript(self, event_type: str, **fields) -> None:
        self.transcript.append({"type": event_type, **fields})

    def _add_to_messages(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
