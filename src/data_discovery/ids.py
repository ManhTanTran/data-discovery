"""Stable identifiers for corpora, documents, pages, and chunks."""

from __future__ import annotations

import hashlib


def make_id(*parts: object) -> str:
    """Tạo ID ngắn, ổn định từ các thành phần đầu vào."""
    raw = "::".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
