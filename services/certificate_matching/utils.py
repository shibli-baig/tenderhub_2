"""Utility helpers for certificate-tender matching."""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Sequence, Set


TOKEN_SPLIT_PATTERN = re.compile(r"[,/;&]|\s+")
NUMBER_PATTERN = re.compile(r"([+-]?\d+[\d,\.]*)(?:\s*(crore|cr|million|mn|billion|bn|lakh|lakhs|k|thousand))?", re.IGNORECASE)

UNIT_MULTIPLIERS = {
    "crore": 10_000_000,
    "cr": 10_000_000,
    "million": 1_000_000,
    "mn": 1_000_000,
    "billion": 1_000_000_000,
    "bn": 1_000_000_000,
    "lakh": 100_000,
    "lakhs": 100_000,
    "k": 1_000,
    "thousand": 1_000,
}


def tokenize(value: Optional[Iterable[str] | str]) -> Set[str]:
    """Convert comma/space separated strings or iterables into normalized tokens."""

    tokens: Set[str] = set()

    if not value:
        return tokens

    if isinstance(value, (list, tuple, set)):
        iterable: Sequence[str] = value  # type: ignore[assignment]
    else:
        iterable = TOKEN_SPLIT_PATTERN.split(str(value))

    for item in iterable:
        token = normalize_token(item)
        if token:
            tokens.add(token)

    return tokens


def normalize_token(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"[^a-z0-9+& ]", "", text.strip().lower())


def parse_inr_amount(value: Optional[str | float | int]) -> Optional[float]:
    """Parse Indian currency expressions into numeric rupees."""

    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value)
    match = NUMBER_PATTERN.search(text.replace(',', ''))
    if not match:
        return None

    number = float(match.group(1))
    unit = match.group(2).lower() if match.group(2) else None

    if unit and unit in UNIT_MULTIPLIERS:
        number *= UNIT_MULTIPLIERS[unit]

    if 'lakh' in text.lower() and not unit:
        number *= UNIT_MULTIPLIERS['lakh']

    return number


def parse_numeric_value(value: Optional[str | float | int]) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    numbers = re.findall(r"[+-]?\d+[\d,\.]*", value.replace(',', ''))
    if not numbers:
        return None
    try:
        return float(numbers[0])
    except ValueError:
        return None


def infer_years_from_text(text: str) -> Optional[int]:
    match = re.search(r"(\d+)\s*(?:years?|yrs?)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def within_lookback(date_value: Optional[datetime], years: int) -> bool:
    if not date_value:
        return False
    try:
        cutoff = datetime.utcnow() - timedelta(days=365 * years)
    except OverflowError:
        cutoff = datetime.utcnow()
    return date_value >= cutoff


def normalize_location(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return re.sub(r"[^a-z0-9 ]", "", value.lower()).strip()


def percent(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return max(0.0, min(1.0, value / max_value))


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def summarize_overlap(source: Set[str], target: Set[str]) -> List[str]:
    return sorted(source & target)


def stringify_list(items: Iterable[str]) -> str:
    return ", ".join(sorted(set(items)))

