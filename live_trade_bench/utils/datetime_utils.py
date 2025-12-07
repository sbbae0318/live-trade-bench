from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def parse_utc_datetime(s: Optional[str]) -> datetime:
    """
    Parse a variety of datetime string formats into an aware UTC datetime.

    Accepted examples:
    - "2025-11-25"
    - "2025-11-25 00:00:00"
    - "2025-11-25 00:00:00 UTC"
    - "2025-11-25T00:00:00"
    - "2025-11-25T000000"
    - ISO strings without timezone (assumed UTC)
    """
    if not s:
        return datetime.now(timezone.utc)
    text = s.strip()
    # Strip common UTC suffixes
    if text.endswith(" UTC"):
        text = text[:-4].strip()
    if text.endswith("Z"):
        text = text[:-1]
    # Try explicit known formats (try date-only format first to avoid partial matches)
    candidates = [
        "%Y-%m-%d",  # Date only - try first to avoid partial match issues
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H%M%S",
    ]
    for fmt in candidates:
        try:
            dt = datetime.strptime(text, fmt)
            # If date-only format, set time to 00:00:00
            if fmt == "%Y-%m-%d":
                dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    # Fallback: fromisoformat after replacing 'T' with space for flexibility
    try:
        cleaned = text.replace("T", " ")
        dt = datetime.fromisoformat(cleaned)
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    except Exception:
        # Last resort: current time in UTC
        return datetime.now(timezone.utc)


