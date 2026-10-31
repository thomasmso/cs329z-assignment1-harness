"""Problem 8 (cardinal): Assemble the toolset (2 points)"""

from __future__ import annotations

import functools
import json

from cs329z_hw1 import data
from cs329z_hw1.cardinal import cardinal_spec
from cs329z_hw1.email_utils import format_email
from cs329z_hw1.pipelines import digest
from cs329z_hw1.pipelines.bm25 import BM25Index
from cs329z_hw1.pipelines.email_qa import answer_question
from cs329z_hw1.types import AgentConfig, Email, LMCallable, ToolSpec


@functools.cache
def _email_index() -> BM25Index:
    """
    Creating a session must not read the email archive: build the email index the first
    time a tool needs it.

    NOTE: `data.load_emails()` is already cached by `data` itself.
    """
    return BM25Index([data.email_text(email) for email in data.load_emails()])


@functools.lru_cache(maxsize=1)
def _emails_by_id() -> dict[str, Email]:
    return {email["id"]: email for email in data.load_emails()}


def cardinal_specs(config: AgentConfig, lm: LMCallable) -> dict[str, ToolSpec]:
    """
    Builds the ten cardinal tool specs for a given session and its config / provided LM.

    Note that `lm` is either aux_lm or agent's lm - we don't need to know which one here.
    """

    def search_emails(query: str, k: int = 5) -> str:

        #
        # NOTE: it's not entirely obvious but data.load_emails() returns the same list that
        # is fed into the BM25 index, so it returns the email POSITION, not the email IDs.
        #
        # In the essence of time, I will not implement good coding practice to make this more obvious.
        #
        emails = data.load_emails()
        results = []
        for email_pos_id, _ in _email_index().search(query, k):
            email = emails[email_pos_id]
            results.append(
                {
                    "id": email["id"],
                    "date": email["date"],
                    "from": email["from"],
                    "subject": email["subject"],
                    # NOTE: deliberately left content out since it might blow up context window
                }
            )
        return json.dumps(results, indent=2)

    def read_email(email_id: str) -> str:
        email = _emails_by_id().get(email_id)
        if email is None:
            # NOTE: Rather propagate a legit error case than return an error string which might
            # look like everything is fine...
            raise ValueError(
                f"Failed to find email with id {email_id!r} - are you sure it's valid?"
            )
        return f"Id: {email_id}\n{format_email(email)}"

    def email_qa(question: str) -> str:
        emails = data.load_emails()

        def search(query: str, k: int) -> list[Email]:
            results = []
            for email_pos_id, _ in _email_index().search(query, k):
                results.append(emails[email_pos_id])
            return results

        return json.dumps(answer_question(question, search, lm))

    def daily_digest(date: str) -> str:
        return digest.daily_digest(data.emails_on(date), lm)

    # Problem 9 (search_docs)
    def search_docs(query: str, k: int = 5) -> str:
        raise NotImplementedError("search_docs is not implemented yet")

    def read_doc(doc_id: str, start_line: int = 1) -> str:
        raise NotImplementedError("read_doc is not implemented yet")

    # Problem 10 (terminal)
    def run_terminal(cmd: str) -> str:
        raise NotImplementedError("run_terminal is not implemented yet")

    # Problem 12 (memory)
    def remember(fact: str) -> str:
        raise NotImplementedError("remember is not implemented yet")

    def recall(query: str) -> str:
        raise NotImplementedError("recall is not implemented yet")

    # Problem 13 (interactive)
    def ask_user(question: str) -> str:
        raise NotImplementedError("ask_user is not implemented yet")

    return {
        spec.name: spec
        for spec in [
            cardinal_spec(
                "search_emails",
                "Keyword search over the company email archive. Returns the k best-matching "
                "emails with their id, date, sender and subject.",
                search_emails,
            ),
            cardinal_spec(
                "read_email",
                "Read one email in full (headers and body) by its id.",
                read_email,
            ),
            cardinal_spec(
                "email_qa",
                "Answer a factual question from the email archive. Searches the archive itself "
                "and returns an answer with the ids of the emails that support it.",
                email_qa,
            ),
            cardinal_spec(
                "daily_digest",
                "Write a short digest of the emails that need attention on one day "
                "(date as YYYY-MM-DD).",
                daily_digest,
            ),
            cardinal_spec("search_docs", "TODO", search_docs),
            cardinal_spec("read_doc", "TODO", read_doc),
            cardinal_spec("run_terminal", "TODO", run_terminal),
            cardinal_spec("remember", "TODO", remember),
            cardinal_spec("recall", "TODO", recall),
            cardinal_spec("ask_user", "TODO", ask_user),
        ]
    }
