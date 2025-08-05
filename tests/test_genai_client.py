import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock

# It's better to import the specific functions/classes we need to test or mock
from api.modules.m1.analyzer import analyze_symptoms, M1_SCHEMA, M1_PROMPT_TEMPLATE
from api.core.genai_client import genai_client
from config.config import config

@pytest.mark.asyncio
async def test_analyze_symptoms_with_claude_uses_tool_choice():
    """
    Verify that when the provider is 'claude', the genai_client constructs
    the correct payload using the 'tool_choice' mechanism for JSON output.
    """
    # --- Setup ---
    # Temporarily switch config to use Claude
    original_provider = config.GENAI_PROVIDER
    original_key = config.CLAUDE_API_KEY
    config.GENAI_PROVIDER = "claude"
    config.CLAUDE_API_KEY = "test-claude-key"
    # The genai_client is a singleton; its provider is set at import time.
    # We must also update the provider on the existing instance for the test.
    genai_client.provider = "claude"

    user_input = "My mother has been forgetting things lately."

    # This is the expected JSON *object* that our analyzer should receive from the client
    # The Claude API returns this object within a 'tool_use' block
    mock_api_response_input = {
        "analysis_process": "The user input mentions memory loss, which maps to M1-01.",
        "matched_warnings": [{
            "warning_id": 1,
            "warning_name": "記憶力減退影響日常生活",
            "match_confidence": 8.5
        }],
        "overall_confidence": 8.5,
        "risk_level": "moderate",
        "recommendations": ["Consult a specialist for a formal evaluation."]
    }

    # This is the full mock response from the Claude /v1/messages API endpoint
    mock_claude_api_response = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "json_output",
                "input": mock_api_response_input
            }
        ],
        "model": "claude-3-sonnet-20240229",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 50, "output_tokens": 150}
    }

    # --- Mocking ---
    # We mock the entire session to control the response of `session.post`
    mock_session = MagicMock()
    mock_post_response = MagicMock()
    # The response from `response.json()` should be our mock Claude response,
    # and since it's awaited, the json() method must be an AsyncMock.
    mock_post_response.json = AsyncMock(return_value=mock_claude_api_response)
    # The `async with` statement will return this mock response object
    mock_session.post.return_value.__aenter__.return_value = mock_post_response

    # --- Execution ---
    # We patch the get_session method on the class. This is more robust.
    # We use AsyncMock because get_session is an async method.
    with patch('api.core.genai_client.SimpleGenAIClient.get_session', new_callable=AsyncMock) as mock_get_session:
        mock_get_session.return_value = mock_session
        result = await analyze_symptoms(user_input)

    # --- Assertions ---
    # 1. Verify that the `post` method was called exactly once.
    mock_session.post.assert_called_once()

    # 2. Extract the arguments passed to the `post` call to inspect the payload.
    _, kwargs = mock_session.post.call_args
    sent_payload = kwargs.get("json", {})

    # 3. Check that the payload was constructed correctly for Claude's tool use.
    assert "tools" in sent_payload
    assert len(sent_payload["tools"]) == 1
    tool_definition = sent_payload["tools"][0]
    assert tool_definition["name"] == "json_output"
    assert tool_definition["input_schema"] == M1_SCHEMA

    assert "tool_choice" in sent_payload
    assert sent_payload["tool_choice"] == {"type": "tool", "name": "json_output"}

    expected_prompt = M1_PROMPT_TEMPLATE.format(user_input=user_input)
    assert sent_payload["messages"][0]["content"] == expected_prompt

    # 4. Check that the final result returned by `analyze_symptoms` is correctly parsed.
    # The `genai_client` should have extracted the `input` dict and `json.dumps` it.
    # The `analyze_symptoms` function then does `json.loads` on it.
    assert result["analysis_process"] == "The user input mentions memory loss, which maps to M1-01."
    assert result["risk_level"] == "moderate"
    assert "metadata" in result
    assert result["metadata"]["provider"] == "claude"
    assert result["metadata"]["tokens_used"] == 200 # 50 input + 150 output

    # --- Teardown ---
    # Restore original config to avoid side effects in other tests
    config.GENAI_PROVIDER = original_provider
    config.CLAUDE_API_KEY = original_key
    # Restore the provider on the client instance as well
    genai_client.provider = original_provider
    # It's also good practice to reset the client's session
    genai_client.session = None
