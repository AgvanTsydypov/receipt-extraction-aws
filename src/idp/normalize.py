"""Normalization used before comparing predictions with ground truth."""

import re


def norm_amount(value) -> str | None:
    """Keep digits only, so '45,500', '45.500' and 'Rp 45500' all compare equal."""
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    if not digits:
        return None
    return str(int(digits))  # drops leading zeros


def norm_text(value) -> str | None:
    """Lowercase, drop punctuation and collapse whitespace."""
    if value is None:
        return None
    text = re.sub(r"[^\w\s]", " ", str(value).lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text or None
