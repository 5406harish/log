"""
Tests for src/agent.py — tool-using agent loop (mocked to avoid live API calls).
Uses OpenAI-compatible mock format for Groq Cloud responses.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agent import LogAnalysisAgent, AGENT_SYSTEM_INSTRUCTION, TOOL_DECLARATIONS


# ---------------------------------------------------------------------------
# Mock LLM responses
# ---------------------------------------------------------------------------
MOCK_FINAL_RESPONSE = """\
## Root Cause Analysis
The database connection pool exhausted all retry attempts.

## Probable Cause
The PostgreSQL instance was unreachable due to max connection limits.

## Remediation Steps
1. Check database status with systemctl.
2. Verify network connectivity.
3. Increase connection pool limits.

## Confidence
HIGH
"""


class TestAgentInit:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
            LogAnalysisAgent(api_key=None)

    @patch("src.agent.OpenAI")
    def test_initialises_with_key(self, mock_client_cls):
        agent = LogAnalysisAgent(api_key="test-key")
        mock_client_cls.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.groq.com/openai/v1",
        )
        assert agent.api_key == "test-key"


class TestAgentInvestigate:
    @patch("src.agent.OpenAI")
    def test_direct_response_no_tool_calls(self, mock_client_cls):
        """Agent returns immediately when LLM gives a final text response."""
        mock_message = MagicMock()
        mock_message.content = MOCK_FINAL_RESPONSE
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 256

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_client_cls.return_value = mock_client_instance

        agent = LogAnalysisAgent(api_key="fake-key")
        result = agent.investigate("/tmp/test.log")

        assert result["confidence"] == "HIGH"
        assert "database" in result["root_cause"].lower()
        assert result["iterations"] == 1
        assert result["tool_calls_log"] == []

    @patch("src.agent.TOOL_REGISTRY", {
        "scan_all_anomalies": lambda **kw: {"status": "anomalies_found", "anomaly_count": 2},
    })
    @patch("src.agent.OpenAI")
    def test_agent_loop_with_tool_call(self, mock_client_cls):
        """Agent executes a tool call, sends result back, then gets final response."""
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_001"
        mock_tool_call.function.name = "scan_all_anomalies"
        mock_tool_call.function.arguments = json.dumps({"file_path": "/tmp/test.log"})

        mock_message_1 = MagicMock()
        mock_message_1.content = None
        mock_message_1.tool_calls = [mock_tool_call]
        mock_message_1.model_dump.return_value = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_001",
                "type": "function",
                "function": {
                    "name": "scan_all_anomalies",
                    "arguments": json.dumps({"file_path": "/tmp/test.log"}),
                },
            }],
        }

        mock_choice_1 = MagicMock()
        mock_choice_1.message = mock_message_1
        mock_response_1 = MagicMock()
        mock_response_1.choices = [mock_choice_1]

        mock_message_2 = MagicMock()
        mock_message_2.content = MOCK_FINAL_RESPONSE
        mock_message_2.tool_calls = None
        mock_choice_2 = MagicMock()
        mock_choice_2.message = mock_message_2
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 512
        mock_response_2 = MagicMock()
        mock_response_2.choices = [mock_choice_2]
        mock_response_2.usage = mock_usage

        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.side_effect = [
            mock_response_1,
            mock_response_2,
        ]
        mock_client_cls.return_value = mock_client_instance

        agent = LogAnalysisAgent(api_key="fake-key")
        result = agent.investigate("/tmp/test.log")

        assert result["iterations"] == 2
        assert len(result["tool_calls_log"]) == 1
        assert result["tool_calls_log"][0]["tool"] == "scan_all_anomalies"
        assert result["confidence"] == "HIGH"

    @patch("src.agent.OpenAI")
    def test_on_tool_call_callback_invoked(self, mock_client_cls):
        """Verify the on_tool_call callback is called during tool execution."""
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_002"
        mock_tool_call.function.name = "parse_log_file"
        mock_tool_call.function.arguments = json.dumps({"file_path": "/tmp/test.log", "context_window": 20.0})

        mock_message_1 = MagicMock()
        mock_message_1.content = None
        mock_message_1.tool_calls = [mock_tool_call]
        mock_message_1.model_dump.return_value = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_002",
                "type": "function",
                "function": {
                    "name": "parse_log_file",
                    "arguments": json.dumps({"file_path": "/tmp/test.log", "context_window": 20.0}),
                },
            }],
        }

        mock_choice_1 = MagicMock()
        mock_choice_1.message = mock_message_1
        mock_response_1 = MagicMock()
        mock_response_1.choices = [mock_choice_1]

        mock_message_2 = MagicMock()
        mock_message_2.content = MOCK_FINAL_RESPONSE
        mock_message_2.tool_calls = None
        mock_choice_2 = MagicMock()
        mock_choice_2.message = mock_message_2
        mock_response_2 = MagicMock()
        mock_response_2.choices = [mock_choice_2]
        mock_response_2.usage = MagicMock()
        mock_response_2.usage.prompt_tokens = 300

        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.side_effect = [
            mock_response_1, mock_response_2,
        ]
        mock_client_cls.return_value = mock_client_instance

        callback_log = []

        def my_callback(name, args, result):
            callback_log.append({"name": name, "args": args})

        agent = LogAnalysisAgent(api_key="fake-key")
        with patch("src.agent.TOOL_REGISTRY", {
            "parse_log_file": lambda **kw: {"status": "anomaly_found", "severity": 6},
        }):
            agent.investigate("/tmp/test.log", on_tool_call=my_callback)

        assert len(callback_log) == 1
        assert callback_log[0]["name"] == "parse_log_file"
        assert callback_log[0]["args"]["context_window"] == 20

    @patch("src.agent.OpenAI")
    def test_max_iterations_safety(self, mock_client_cls):
        """Agent returns a LOW-confidence fallback if max iterations exceeded."""
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_loop"
        mock_tool_call.function.name = "scan_all_anomalies"
        mock_tool_call.function.arguments = json.dumps({"file_path": "/tmp/test.log"})

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        mock_message.model_dump.return_value = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_loop",
                "type": "function",
                "function": {
                    "name": "scan_all_anomalies",
                    "arguments": json.dumps({"file_path": "/tmp/test.log"}),
                },
            }],
        }

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_client_cls.return_value = mock_client_instance

        with patch("src.agent.TOOL_REGISTRY", {
            "scan_all_anomalies": lambda **kw: {"status": "clean"},
        }):
            with patch("src.agent.MAX_AGENT_ITERATIONS", 3):
                agent = LogAnalysisAgent(api_key="fake-key")
                result = agent.investigate("/tmp/test.log")

        assert result["confidence"] == "LOW"
        assert result["iterations"] == 3


class TestAgentSystemPrompt:
    def test_system_instruction_contains_tool_descriptions(self):
        assert "parse_log_file" in AGENT_SYSTEM_INSTRUCTION
        assert "scan_all_anomalies" in AGENT_SYSTEM_INSTRUCTION
        assert "read_log_lines" in AGENT_SYSTEM_INSTRUCTION
        assert "search_log_pattern" in AGENT_SYSTEM_INSTRUCTION

    def test_tool_declarations_has_four_functions(self):
        assert len(TOOL_DECLARATIONS) == 4

    def test_tool_declaration_names(self):
        names = {td["function"]["name"] for td in TOOL_DECLARATIONS}
        assert names == {
            "parse_log_file",
            "scan_all_anomalies",
            "read_log_lines",
            "search_log_pattern",
        }
