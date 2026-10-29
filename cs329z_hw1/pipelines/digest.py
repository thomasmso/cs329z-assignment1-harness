"""Problem 2 (daily_digest): Daily digest generation (4 points)"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from cs329z_hw1.email_utils import format_email
from cs329z_hw1.llm import OutputTruncated
from cs329z_hw1.pipelines.priority import classify_priority
from cs329z_hw1.types import Email, LMCallable

MAX_WORKERS = 16
WORD_COUNT_LIMIT = 250  # the hard limit is 300 (len(digest.split())); aim below it

RANK = {"urgent": 0, "normal": 1}

SYSTEM_PROMPT = f"""You write a short morning email digest for a busy employee.

You are given the day's emails that need attention, already sorted: URGENT
items first, then NORMAL items. Mail that can be ignored has been removed.

Rules:
- Write at most {WORD_COUNT_LIMIT} words in total. This is a hard limit.
- Start with the urgent items. For EVERY urgent item, name the specific
  company, project, site, or asset it concerns (exactly as written in the
  email), what is needed, and the deadline.
- Then summarize the normal items briefly. Group similar ones and skip
  minor details; one short line per item or group.
- Mention only things that appear in the items below. Do not invent details.
- Plain text. Short headings and bullet points are fine.
"""


@dataclass(frozen=True)
class DigestItem:
    """One email kept for the digest, with its priority label."""

    email: Email
    category: str  # "urgent" or "normal"; unknown is already mapped to normal
    reason: str

    @property
    def subject(self) -> str:
        return self.email.get("subject") or "(no subject)"

    @property
    def sort_key(self) -> tuple[int, str]:
        """Urgent before normal, then oldest first."""
        return (RANK[self.category], self.email["date"])


def daily_digest(emails: list[Email], lm: LMCallable) -> str:
    """Classify, drop ignore, keep one item per thread, order by urgency, then
    one LLM call writes the digest. Returns a string for every input and reply."""
    if not emails:
        return "No emails arrived on this day."

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        labels = list(pool.map(lambda email: classify_priority(email, lm), emails))

    items = _select_digest_items(emails, labels)
    if not items:
        return "Nothing needs your attention today: every email was advertising, a newsletter, or a mass announcement."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _render_digests(items)},
    ]
    try:
        digest = lm(messages)
    except OutputTruncated as exc:
        digest = exc.text or ""
    except Exception as exc:
        return f"The digest could not be written: {exc}"
    return (
        digest.strip()
        or "The digest could not be written: the model returned an empty reply."
    )


def _select_digest_items(emails: list[Email], labels: list[dict]) -> list[DigestItem]:
    """One item per thread, ignore dropped, urgent first, then by time."""

    best_items: dict[str, DigestItem] = {}  # item_id -> best item in the thread

    for email, label in zip(emails, labels):
        category = "normal" if label["category"] == "unknown" else label["category"]

        # NOTE: I decided to ignore emails in code so model will NEVER get to see them
        if category == "ignore":
            continue
        item = DigestItem(email, category, label["reason"])

        # Logic to keep one item per thread, order by urgency
        if email.get("thread_id"):
            item_id = email["thread_id"]
        else:
            item_id = email["id"]  # no thread: the email stands on its own
        best_item = best_items.get(item_id)
        if best_item is None or _replace_item_if_needed(item, best_item):
            best_items[item_id] = item

    # sort ALL digest items by urgency and time
    return sorted(best_items.values(), key=lambda item: item.sort_key)


def _replace_item_if_needed(new: DigestItem, current: DigestItem) -> bool:
    """The most urgent email of a thread wins; on a tie, the later one."""
    if new.category != current.category:
        return RANK[new.category] < RANK[current.category]
    return new.email["date"] >= current.email["date"]


def _render_digests(items: list[DigestItem]) -> str:
    """Urgent items with the whole email since the problem emphasizes on its details,
    so the names the digest must mention are there; normal items as one line each,
    to keep a full day's prompt small."""
    blocks = []
    for n, item in enumerate(items, 1):
        # NOTE: I deliberately include entire email into conext for urgent emails,
        # since we need to extract more details from them.
        if item.category == "urgent":
            blocks.append(
                f"[{n}] URGENT\n{format_email(item.email)}\nWhy urgent: {item.reason}"
            )
        else:
            sender = item.email.get("from") or "(unknown)"
            blocks.append(
                f"[{n}] NORMAL from {sender}: {item.subject} -- {item.reason}"
            )
    return "\n\n".join(blocks)
