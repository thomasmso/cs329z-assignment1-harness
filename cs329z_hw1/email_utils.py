"""Helpers for presenting archive emails to a model."""

from __future__ import annotations

from datetime import datetime

from cs329z_hw1.types import Email


def format_email(email: Email) -> str:
    """An email as prompt text: From, To, Date, and Subject headers, then the body."""
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
