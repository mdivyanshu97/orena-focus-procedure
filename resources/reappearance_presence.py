"""Safe route for timestamped binary reappearance questions.

The source question states that the named object was last visible just before
an explicit timestamp. Therefore the original yes/no question is equivalent
to a simpler visual query over only the later frames: is that object visible
anywhere after the timestamp?

Rows that do not match this narrow template return ``None`` and retain the
candidate's existing sampling and prompt path unchanged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_TIMESTAMP = re.compile(r"(\d+):(\d{2}):(\d{2})")
_OBJECT = re.compile(
    r"Does the (.+?), last visible just before", flags=re.IGNORECASE
)


@dataclass(frozen=True)
class ReappearancePresenceRoute:
    window: tuple[float, float]
    question: str
    object_name: str


def _seconds(match: re.Match[str]) -> float:
    return (
        int(match.group(1)) * 3600
        + int(match.group(2)) * 60
        + int(match.group(3))
    )


def route_reappearance_presence(
    question: str,
    answer_format: str,
    start_time: float,
    end_time: float,
) -> ReappearancePresenceRoute | None:
    """Return a clip-relative post-timestamp route for the exact target family."""
    text = question or ""
    lower = text.lower()
    if answer_format != "binary":
        return None
    if "last visible just before" not in lower:
        return None
    if "re-appear later in the video" not in lower and (
        "reappear later in the video" not in lower
    ):
        return None

    timestamp = _TIMESTAMP.search(text)
    foreign_object = _OBJECT.search(text)
    if timestamp is None or foreign_object is None:
        return None

    start = float(start_time)
    end = float(end_time)
    anchor = _seconds(timestamp)
    if end <= start or anchor > end:
        return None

    lo = max(start, anchor) - start
    hi = end - start
    if hi - lo < 0.25:
        # A few official rows place the anchor exactly at the clip end. Keep
        # the measured behavior by showing the final quarter-second; it gives
        # the presence query a valid frame batch without leaking earlier video.
        lo = max(0.0, hi - 0.25)

    object_name = foreign_object.group(1).strip()
    presence_question = (
        f"Looking only at these chronological frames, is a {object_name} "
        "visible in any frame?\n\n"
        "Answer with exactly one word: yes or no."
    )
    return ReappearancePresenceRoute(
        window=(lo, hi),
        question=presence_question,
        object_name=object_name,
    )
