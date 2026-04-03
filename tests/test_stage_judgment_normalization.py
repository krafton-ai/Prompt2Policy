"""Tests for StageJudgment normalization in judge_agent.py.

Verifies that _parse_vlm_response and _run_code_judge return properly
constructed StageJudgment TypedDicts (not raw dicts), including the
optional evidence field.
"""

from __future__ import annotations

import json
from pathlib import Path

from p2p.agents.judge_agent import (
    _parse_vlm_response,
    _run_code_judge,
)


def _json(**kwargs: object) -> str:
    return json.dumps(kwargs)


# ---------------------------------------------------------------------------
# _parse_vlm_response
# ---------------------------------------------------------------------------


class TestParseVlmResponse:
    def test_returns_stage_judgment_fields(self):
        text = _json(
            intent_score=0.8,
            diagnosis="good",
            failure_tags=["a"],
        )
        result = _parse_vlm_response(text)

        assert result["intent_score"] == 0.8
        assert result["diagnosis"] == "good"
        assert result["failure_tags"] == ["a"]

    def test_preserves_evidence_field(self):
        text = _json(
            intent_score=0.5,
            diagnosis="ok",
            failure_tags=[],
            evidence=["e1", "e2"],
        )
        result = _parse_vlm_response(text)

        assert result["evidence"] == ["e1", "e2"]

    def test_preserves_empty_evidence_list(self):
        text = _json(
            intent_score=0.5,
            diagnosis="ok",
            failure_tags=[],
            evidence=[],
        )
        result = _parse_vlm_response(text)

        assert result["evidence"] == []

    def test_missing_evidence_not_in_result(self):
        text = _json(
            intent_score=0.5,
            diagnosis="ok",
            failure_tags=[],
        )
        result = _parse_vlm_response(text)

        assert "evidence" not in result

    def test_clamps_score_above_1(self):
        text = _json(
            intent_score=7.5,
            diagnosis="",
            failure_tags=[],
        )
        result = _parse_vlm_response(text)

        assert result["intent_score"] == 0.75  # 7.5 / 10.0

    def test_score_none_when_missing(self):
        text = _json(
            diagnosis="no score",
            failure_tags=[],
        )
        result = _parse_vlm_response(text)

        assert result["intent_score"] is None

    def test_falls_back_on_score_key(self):
        text = _json(
            score=0.6,
            diagnosis="",
            failure_tags=[],
        )
        result = _parse_vlm_response(text)

        assert result["intent_score"] == 0.6

    def test_invalid_json_returns_empty_result(self):
        result = _parse_vlm_response("not json at all")

        assert result["intent_score"] is None
        assert "JSON parse failed" in result["diagnosis"]

    def test_drops_extra_keys(self):
        text = _json(
            intent_score=0.5,
            diagnosis="",
            failure_tags=[],
            random_extra=123,
        )
        result = _parse_vlm_response(text)

        assert "random_extra" not in result

    def test_defaults_missing_optional_fields(self):
        text = _json(intent_score=0.5)
        result = _parse_vlm_response(text)

        assert result["diagnosis"] == ""
        assert result["failure_tags"] == []


# ---------------------------------------------------------------------------
# _run_code_judge
# ---------------------------------------------------------------------------

_JUDGE_TEMPLATE = """\
def judge_fn(trajectory, summary):
    return {result}
"""


class TestRunCodeJudge:
    def test_returns_stage_judgment(self, tmp_path: Path):
        traj_path = tmp_path / "trajectory.npz"
        code = _JUDGE_TEMPLATE.format(
            result=dict(
                intent_score=0.9,
                diagnosis="ok",
                failure_tags=[],
            )
        )
        result = _run_code_judge(traj_path, code, {})

        assert result["intent_score"] == 0.9
        assert result["diagnosis"] == "ok"

    def test_preserves_evidence(self, tmp_path: Path):
        traj_path = tmp_path / "trajectory.npz"
        code = _JUDGE_TEMPLATE.format(
            result=dict(
                intent_score=0.5,
                diagnosis="",
                failure_tags=[],
                evidence=["proof"],
            )
        )
        result = _run_code_judge(traj_path, code, {})

        assert result["evidence"] == ["proof"]

    def test_drops_extra_keys(self, tmp_path: Path):
        traj_path = tmp_path / "trajectory.npz"
        code = _JUDGE_TEMPLATE.format(
            result=dict(
                intent_score=0.5,
                diagnosis="",
                failure_tags=[],
                bonus=42,
            )
        )
        result = _run_code_judge(traj_path, code, {})

        assert "bonus" not in result
