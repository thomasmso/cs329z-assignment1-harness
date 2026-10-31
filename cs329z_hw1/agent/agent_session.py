"""Problem 7 (agent_loop): Implement the tool-calling loop (8 points)"""

from __future__ import annotations

import itertools
import json
import os
from typing import Optional

from cs329z_hw1.agent.cardinal_tools import cardinal_specs
from cs329z_hw1.agent.tool_protocol import parse_response, render_tools
from cs329z_hw1.agent.tool_registry import INVALID_ARGS, UNKNOWN_TOOL, ToolRegistry
from cs329z_hw1.llm import OutputTruncated
from cs329z_hw1.tokens import CHARS_PER_TOKEN, count_message_tokens, count_tokens
from cs329z_hw1.types import AgentConfig, AgentResult, LMCallable, ToolSpec

_session_numbers = itertools.count(1)

MAX_MALFORMED_REPLIES_IN_A_ROW = 3
MALFORMED_STATUSES = (UNKNOWN_TOOL, INVALID_ARGS)  # the model's fault, not the tool's
COMPACTION_TARGET_FRACTION = 0.5  # compact down to this share of config.context_budget
TOOL_RESULT_BUDGET_FRACTION = (
    0.25  # max share of config.context_budget for one tool result
)
SUMMARY_MAX_TOKENS = 100  # about 400 characters

SUMMARY_PROMPT = """
Below is the earlier part of a conversation between an employee of Cardinal Energy and their
assistant (you are the assistant), and the summary of what came before it.

Write one updated summary of everything so far.

Make sure to include:
- Numbers, names, and dates (keep exactly as written).
- Any user asks, constraints or preferences.
- Facts found.
- Any doc_id, email_id, file, and where they were found.
- Any decisions made.
- Anything left unfinished.

Leave out raw tool output that won't matter later. Write at most {max_words} words.

Summary so far:
{summary}

Earlier conversation:
{conversation}
"""

# Failing a few tests after adding more context via the tools in Problem 10...
# I gave the agent an identity and what tools it has at its disposal.
#
# #5 came from d20 - the model found "... described in the Benefits Summary..." and stopped instead of reading benefits-summary-2026.md
# #6 came from d17 - "gym membership" matches a doc saying there's no gym. But "fitness" term would've showed up the "wellness" allowance.
# #7 came from d18 - the model replied "I'll look up Cardinal Energy's reporting instructions." with no tool call, so that was taken as the final answer.
# #8 came from t03 and t04 - the model replied "I can't access the archive in this chat" without trying the terminal.
SYSTEM_PROMPT = """
You are an assistant for employees of Cardinal Energy. You have access to the company
email archive, the company documents (policies, guides, specs, announcements), and a terminal.

1. Questions are about Cardinal Energy and its archive and documents; never ask which company or archive.
2. Look the answer up with your tools before replying; never answer from memory or guess.
3. If one tool doesn't have the answer, try another (emails, documents, or the terminal) before giving up.
4. To use a tool, call it in the same reply. When you are finished, reply with no tool call.
5. If a result points to another document or section, look it up before answering.
6. If a search finds nothing, search again with other words (synonyms, the likely policy name).
7. Do not reply with what you are going to do next (e.g. "I'll look up Cardinal Energy's reporting instructions"); do it with a tool call instead.
8. Never say you can't access the archive or documents: emails.jsonl and docs/ are in your terminal's workspace, and the search tools cover both.
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
    user_message_index: int  # where the current send's user message is in messages

    # summary of the compacted-away messages, kept at messages[1] if exists
    summary: str | None

    def __init__(
        self, lm: LMCallable, tools: Optional[list[ToolSpec]], config: AgentConfig
    ) -> None:
        self.session_number = next(
            _session_numbers
        )  # labels printed events (CS329Z_PRINT_EVENTS)
        self.lm = lm
        self.config = config
        self.mode = config.mode
        self.transcript = []
        self.tool_registry = ToolRegistry(self._select_tools(tools))

        self.messages = []
        self.summary = None

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

        # Add user message to messages (e.g. new "send" to existing convo)
        self.user_message_index = len(
            self.messages
        )  # compaction keeps this user'ss latest message and all after it
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
            # Check if messages exceed context budget, if so, compact
            if count_message_tokens(self.messages) > self.config.context_budget:
                self._compact()

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
                    error = "Your reply was empty. Reply again with a tool call, or with your final answer."
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

                # Cut tool results that are too big for the context (solve `test_one_result_larger_than_the_budget` test case)
                content = result.content
                max_chars = (
                    int(self.config.context_budget * TOOL_RESULT_BUDGET_FRACTION)
                    * CHARS_PER_TOKEN
                )
                if len(content) > max_chars:
                    content = f"{content[:max_chars]}\n...[Result truncated: {len(content) - max_chars} characters left out.]"
                self._add_to_transcript(
                    "tool_result",
                    name=name,
                    status=result.status,
                    content=content,
                )
                self._add_to_messages(
                    "user",
                    f"Result of tool call for {name} has status {result.status} and content: {content}",
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

    def _compact(self) -> None:
        """
        Compact `self.messages` such that `count_message_tokens(self.messages) <= config.context_budget`.

        Keeps the system prompt and the current user message. Summaries are written by
        `config.aux_lm or self.lm`. The transcript is never edited.
        """

        print(
            f"Compacting messages from {count_message_tokens(self.messages)} tokens to under {self.config.context_budget} tokens..."
        )

        # Q: How far to shrink when over budget?
        # Since we don't want to compact every turn, target a goal of 50% compaction

        # Q: What happens to older messages?
        # We will drop them (sliding window) first, then see how many tests fail.
        # Then we will try rolling summary.
        #  Summary will just be a new message in self.messages from the aux_lm.

        # We want to keep obviously the system prompt, but in additionto that, we want to keep:
        #    - The current user message and everything after it
        #    - The latest summary if there is one
        #
        # This means we are going to drop first: the oldest messages after the system prompt,
        # up to the current user message.
        tokens_before = count_message_tokens(self.messages)
        target = int(self.config.context_budget * COMPACTION_TARGET_FRACTION)
        had_summary = self.summary is not None
        first = 2 if had_summary else 1  # don't drop the summary message itself
        dropped_messages = []
        while (
            self.user_message_index > first
            and count_message_tokens(self.messages) > target
        ):
            dropped_messages.append(self.messages.pop(first))
            self.user_message_index -= 1

        # Roll the dropped messages into the summary
        if dropped_messages:
            self.summary = self._summarize(dropped_messages)
            summary_message = {
                "role": "user",
                "content": f"Summary of the earlier conversation (older messages were removed to save space):\n{self.summary}",
            }
            if had_summary:
                self.messages[1] = summary_message
            else:
                self.messages.insert(1, summary_message)
                self.user_message_index += 1

        self._add_to_transcript(
            "compaction",
            dropped=len(dropped_messages),
            tokens_before=tokens_before,
            tokens_after=count_message_tokens(self.messages),
            summary=self.summary,
        )

        print(f"Compacted messages to {count_message_tokens(self.messages)} tokens.")

    # NOTE: summarizing passed `test_live_constraint_survives_compaction` test
    def _summarize(self, dropped_messages: list[dict]) -> str:
        """Ask `config.aux_lm` or `self.lm` to summarize `dropped_messages` to `self.summary`."""
        conversation = "\n\n".join(
            f"{m['role']}: {m['content']}" for m in dropped_messages
        )
        prompt = SUMMARY_PROMPT.format(
            summary=self.summary or "(none yet)",
            conversation=conversation,
            max_words=SUMMARY_MAX_TOKENS
            * 3
            // 4,  # about 0.75 words per token (avg English word is 5chars)
        )
        lm_to_use = self.config.aux_lm or self.lm
        reply = lm_to_use([{"role": "user", "content": prompt}])
        return reply.strip()[
            : SUMMARY_MAX_TOKENS * CHARS_PER_TOKEN
        ]  # ensure reply is within token budget

    def _add_to_transcript(self, event_type: str, **fields) -> None:
        self.transcript.append({"type": event_type, **fields})
        if os.environ.get("CS329Z_PRINT_EVENTS"):
            # sessions of a live test can run in parallel threads, so tag each line with its session
            print(
                f"[session {self.session_number}] {event_type}: {json.dumps(fields)[:1000]}"
            )

    def _add_to_messages(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
