"""Tests for session ID generation."""

from __future__ import annotations

import re

from p2p.session.session_id import generate_session_id

SESSION_RE = re.compile(r"^session_\d{8}_\d{6}_.+$")


def test_default_format():
    sid = generate_session_id()
    assert SESSION_RE.match(sid)
    # Default tail is 8-char hex
    tail = sid.split("_", 3)[3]
    assert len(tail) == 8


def test_custom_suffix():
    sid = generate_session_id("cfg_0_seed_42")
    assert SESSION_RE.match(sid)
    assert sid.endswith("_cfg_0_seed_42")


def test_uniqueness():
    ids = {generate_session_id() for _ in range(10)}
    assert len(ids) == 10
