"""Tests for intent elicitation module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from p2p.agents.intent_elicitor import build_elaborated_text, elaborate_intent

# ---------------------------------------------------------------------------
# build_elaborated_text
# ---------------------------------------------------------------------------


class TestBuildElaboratedText:
    def test_no_criteria_returns_original(self):
        assert build_elaborated_text("Walk forward", []) == "Walk forward"

    def test_no_criteria_with_empty_custom(self):
        assert build_elaborated_text("Walk forward", [], custom_criteria=[]) == "Walk forward"

    def test_selected_criteria(self):
        result = build_elaborated_text(
            "Walk forward",
            ["Alternate legs symmetrically", "Torso upright"],
        )
        assert result.startswith("Walk forward\n\nBehavioral criteria:\n")
        assert "- Alternate legs symmetrically" in result
        assert "- Torso upright" in result

    def test_custom_criteria_only(self):
        result = build_elaborated_text(
            "Walk forward",
            [],
            custom_criteria=["Move at least 5m"],
        )
        assert "- Move at least 5m" in result

    def test_selected_plus_custom(self):
        result = build_elaborated_text(
            "Walk forward",
            ["Alternate legs"],
            custom_criteria=["No arm flailing"],
        )
        lines = result.split("\n")
        criteria_lines = [line for line in lines if line.startswith("- ")]
        assert len(criteria_lines) == 2


# ---------------------------------------------------------------------------
# elaborate_intent (mocked LLM)
# ---------------------------------------------------------------------------


_MOCK_LLM_RESPONSE = (
    '[{"title": "Alternate legs symmetrically",'
    ' "description": "Alternate left/right legs in symmetric strides",'
    ' "category": "gait", "default_on": true},'
    ' {"title": "Torso upright",'
    ' "description": "Torso upright with slight forward lean",'
    ' "category": "posture", "default_on": true},'
    ' {"title": "Contralateral arm swing",'
    ' "description": "Arms swing opposite to corresponding leg",'
    ' "category": "dynamics", "default_on": false}]'
)

_MOCK_ENV = MagicMock(
    env_id="HalfCheetah-v5",
    name="HalfCheetah",
    description="A 2D cheetah",
    obs_dim=17,
    action_dim=6,
    state_ref="half_cheetah",
    engine="mujoco",
)


class TestElaborateIntent:
    @patch("p2p.agents.intent_elicitor.extract_response_text")
    @patch("p2p.agents.intent_elicitor.create_message")
    @patch("p2p.training.env_spec.get_env_spec", return_value=_MOCK_ENV)
    @patch("p2p.training.env_spec.extract_mujoco_body_info", return_value="body info")
    @patch("p2p.training.env_spec.extract_joint_semantics", return_value="joint sem")
    @patch("p2p.training.env_spec.extract_body_geometry", return_value="body geo")
    def test_basic_parsing(self, _geo, _sem, _body, _env, mock_create, mock_extract):
        mock_extract.return_value = _MOCK_LLM_RESPONSE

        client = MagicMock()
        criteria = elaborate_intent("Walk forward", "HalfCheetah-v5", client=client)

        assert len(criteria) == 3
        assert criteria[0]["title"] == "Alternate legs symmetrically"
        assert criteria[0]["description"] == "Alternate left/right legs in symmetric strides"
        assert criteria[0]["category"] == "gait"
        assert criteria[0]["default_on"] is True
        assert criteria[2]["default_on"] is False
        mock_create.assert_called_once()

    @patch("p2p.agents.intent_elicitor.extract_response_text")
    @patch("p2p.agents.intent_elicitor.create_message")
    @patch("p2p.training.env_spec.get_env_spec", return_value=_MOCK_ENV)
    @patch("p2p.training.env_spec.extract_mujoco_body_info", return_value="")
    @patch("p2p.training.env_spec.extract_joint_semantics", return_value="")
    @patch("p2p.training.env_spec.extract_body_geometry", return_value="")
    def test_markdown_code_block(self, _geo, _sem, _body, _env, mock_create, mock_extract):
        mock_extract.return_value = "```json\n" + _MOCK_LLM_RESPONSE + "\n```"

        client = MagicMock()
        criteria = elaborate_intent("Walk forward", "HalfCheetah-v5", client=client)

        assert len(criteria) == 3

    @patch("p2p.agents.intent_elicitor.extract_response_text")
    @patch("p2p.agents.intent_elicitor.create_message")
    @patch("p2p.training.env_spec.get_env_spec", return_value=_MOCK_ENV)
    @patch("p2p.training.env_spec.extract_mujoco_body_info", return_value="")
    @patch("p2p.training.env_spec.extract_joint_semantics", return_value="")
    @patch("p2p.training.env_spec.extract_body_geometry", return_value="")
    def test_invalid_entries_filtered(self, _geo, _sem, _body, _env, mock_create, mock_extract):
        mock_extract.return_value = (
            '[{"description": "Good one", "category": "gait", "default_on": true},'
            ' "not a dict", {"no_desc": true}]'
        )

        client = MagicMock()
        criteria = elaborate_intent("Walk forward", "HalfCheetah-v5", client=client)

        assert len(criteria) == 1
        assert criteria[0]["title"] == "Good one"
        assert criteria[0]["description"] == "Good one"

    @patch("p2p.agents.intent_elicitor.extract_response_text")
    @patch("p2p.agents.intent_elicitor.create_message")
    @patch("p2p.training.env_spec.get_env_spec", return_value=_MOCK_ENV)
    @patch("p2p.training.env_spec.extract_mujoco_body_info", return_value="")
    @patch("p2p.training.env_spec.extract_joint_semantics", return_value="")
    @patch("p2p.training.env_spec.extract_body_geometry", return_value="")
    def test_invalid_json_returns_empty(self, _geo, _sem, _body, _env, mock_create, mock_extract):
        mock_extract.return_value = "This is not JSON at all"

        client = MagicMock()
        criteria = elaborate_intent("Walk forward", "HalfCheetah-v5", client=client)

        assert criteria == []
