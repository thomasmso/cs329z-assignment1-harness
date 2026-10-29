"""Problem 1 (priority): Mark email priority (3 points)"""

from __future__ import annotations

import re
from datetime import datetime

from cs329z_hw1.data import REPO_ROOT
from cs329z_hw1.llm import OutputTruncated
from cs329z_hw1.types import Email, LMCallable

GOLD_LABELS = ("urgent", "normal", "ignore")

RUBRIC = (REPO_ROOT / "data" / "priority_rubric.md").read_text(encoding="utf-8")


# NOTE: Before I added the section after the "---" line, we were hitting 56/60 test cases.
#
# The emails that failed:
#     - em-04298 - "...fax the docs asap??", we labeled as "normal" but should be "urgent"
#     - em-06661 - "...Stuart is sitting down with them tomorrow...", we labeled as "urgent" but should be "normal"
#     - em-21318 - "...seating chart is coming Thursday...", we labeled as "urgent" but should be "normal"
#
# After the change, score went up to 59/60.
SYSTEM_PROMPT = (
    RUBRIC
    + """

---

Applying the rubric:
- A request "clearly implies" same-day action when it says ASAP, right away,
  or as soon as possible, or when the sender says they are blocked waiting on it.
- A deadline is a time by which THIS recipient must act. Dates of other events
  mentioned in the email (a meeting someone else attends, a document arriving)
  are not deadlines unless the recipient's action is due by then.

You will be given one email. Classify it for its recipient under the rubric
above. Reply with exactly two lines and nothing else:

LABEL: <urgent|normal|ignore>
REASON: <one sentence explaining the label>
"""
)

_STRICT_LABEL = re.compile(
    r"\blabel\s*[:=]\s*[*\"'`]*\s*(urgent|normal|ignore)\b", re.IGNORECASE
)
_JSON_LABEL = re.compile(
    r"\"category\"\s*:\s*\"(urgent|normal|ignore)\"", re.IGNORECASE
)
_ANY_LABEL = re.compile(r"\b(urgent|normal|ignore)\b", re.IGNORECASE)
_STRICT_REASON = re.compile(r"\breason\s*[:=]\s*[*\"'`]*\s*(.+)", re.IGNORECASE)
_JSON_REASON = re.compile(r"\"reason\"\s*:\s*\"([^\"]+)\"", re.IGNORECASE)


def format_email(email: Email) -> str:
    """The user message: headers the rubric depends on, then the email itself."""
    recipients = email.get("to") or []
    listed = ", ".join(recipients)
    to_line = (
        f"{listed} ({len(recipients)} recipient{'s' if len(recipients) != 1 else ''})"
        if recipients
        else "(none listed)"
    )

    body = email.get("body") or ""

    return (
        f"From: {email.get('from') or '(unknown)'}\n"
        f"To: {to_line}\n"
        f"Date: {_format_date(email.get('date') or '')}\n"
        f"Subject: {email.get('subject') or '(no subject)'}\n"
        f"\n{body.strip() or '(empty body)'}"
    )


def _format_date(date: str) -> str:
    """'2001-06-22T16:56:37Z' -> 'Friday 2001-06-22 16:56 UTC', so the model can count working days."""
    try:
        return datetime.fromisoformat(date.replace("Z", "+00:00")).strftime(
            "%A %Y-%m-%d %H:%M UTC"
        )
    except ValueError:
        return date or "(unknown)"


def parse_reply(text: str) -> dict:
    """Pull a label and a reason out of any reply. Never raises.

    Tries the requested ``LABEL:`` format, then a JSON ``"category"`` key, then
    falls back to the last label word in the text. No label word gives "unknown".
    """
    text = text or ""
    match = _STRICT_LABEL.search(text) or _JSON_LABEL.search(text)
    if match:
        category = match.group(1).lower()
    else:
        words = _ANY_LABEL.findall(text)
        category = words[-1].lower() if words else "unknown"
    return {"category": category, "reason": _parse_reason(text)}


def _parse_reason(text: str) -> str:
    match = _STRICT_REASON.search(text) or _JSON_REASON.search(text)
    if match:
        reason = match.group(1)
    else:
        reason = next((line for line in text.splitlines() if line.strip()), "")
    reason = reason.strip().strip("*\"'`").strip()
    return reason or "The model gave no reason."


def classify_priority(email: Email, lm: LMCallable) -> dict:
    """One model call; returns {"category", "reason"} and never raises."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_email(email)},
    ]
    try:
        reply = lm(messages)
    except OutputTruncated as exc:
        reply = exc.text or ""
    except Exception as exc:
        return {"category": "unknown", "reason": f"The model call failed: {exc}"}
    print(reply)
    return parse_reply(reply)
