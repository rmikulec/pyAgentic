import pytest
from unittest.mock import AsyncMock, patch
from typing import List

from pyagentic._base._agent import Agent
from pyagentic._base._context import ContextItem, computed_context
from pyagentic._base._tool import tool
from pyagentic.updates import Status, EmitUpdate, AiUpdate, ToolUpdate


# Mock OpenAI response classes
class MockFunctionCall:
    def __init__(
        self, name: str, arguments: str, call_id: str = "test_call_id", type: str = "function_call"
    ):
        self.name = name
        self.arguments = arguments
        self.call_id = call_id
        self.type = type
        self.id = call_id


class MockResponseOutput:
    def __init__(self, type: str, content: str = None):
        self.type = type
        self.content = content

    def to_dict(self):
        return {"type": self.type, "content": self.content}


class MockOpenAIResponse:
    def __init__(
        self,
        output_text: str = None,
        function_calls: List[MockFunctionCall] = None,
        reasoning: List[str] = None,
    ):
        self.output_text = output_text or ""
        self.output = []

        # Add reasoning if provided
        if reasoning:
            for reason in reasoning:
                self.output.append(MockResponseOutput("reasoning", reason))

        # Add function calls if provided
        if function_calls:
            self.output.extend(function_calls)


@pytest.fixture
def test_agent():
    """Create a test agent with tools for testing"""

    class TestAgent(Agent):
        __system_message__ = "I am a test agent"

        counter: int = ContextItem(default=0)
        test_data: str = ContextItem(default="test")

        @computed_context
        def dynamic_value(self):
            return f"dynamic_{self.counter}"

        @tool("Add two numbers together")
        def add_numbers(self, a: int, b: int) -> str:
            result = a + b
            self.context.counter += 1
            return f"The sum of {a} and {b} is {result}"

        @tool("Get current counter value")
        def get_counter(self) -> str:
            return f"Current counter: {self.context.counter}"

        @tool("Update test data")
        def update_data(self, new_data: str) -> str:
            self.context.test_data = new_data
            return f"Updated data to: {new_data}"

        @tool("Tool that raises an error")
        def error_tool(self, should_error: bool = True) -> str:
            if should_error:
                raise ValueError("This is a test error")
            return "No error occurred"

    return TestAgent(model="gpt-4", api_key="test_key")


@pytest.fixture
def mock_emitter():
    """Create a mock emitter function"""
    return AsyncMock()


class TestAgentBasicRun:
    """Test basic agent run functionality"""

    @pytest.mark.asyncio
    async def test_simple_text_response(self, test_agent):
        """Test agent returning simple text without tool calls"""

        with patch.object(test_agent, "client") as mock_client:
            # Mock OpenAI response with just text
            mock_response = MockOpenAIResponse(output_text="Hello, I'm a test response!")
            mock_client.responses.create = AsyncMock(return_value=mock_response)

            result = await test_agent.run("Hello")

            assert result.final_output == "Hello, I'm a test response!"
            assert len(result.tool_responses) == 0
            assert mock_client.responses.create.call_count == 1

            # Verify the call parameters
            call_args = mock_client.responses.create.call_args
            assert call_args.kwargs["model"] == "gpt-4"
            assert "tools" in call_args.kwargs
            assert call_args.kwargs["max_tool_calls"] == 5

    @pytest.mark.asyncio
    async def test_single_tool_call(self, test_agent):
        """Test agent making a single tool call"""

        with patch.object(test_agent, "client") as mock_client:
            # First call returns tool call, second call returns final text
            tool_call = MockFunctionCall("add_numbers", '{"a": 5, "b": 3}')
            mock_tool_response = MockOpenAIResponse(function_calls=[tool_call])
            mock_final_response = MockOpenAIResponse(output_text="I calculated 5 + 3 = 8 for you!")

            mock_client.responses.create = AsyncMock(
                side_effect=[mock_tool_response, mock_final_response]
            )

            result = await test_agent.run("Add 5 and 3")

            assert result.final_output == "I calculated 5 + 3 = 8 for you!"
            assert len(result.tool_responses) == 1

            # Check tool response details
            tool_response = result.tool_responses[0]
            assert tool_response.a == 5
            assert tool_response.b == 3
            assert tool_response.output == "The sum of 5 and 3 is 8"
            assert tool_response.call_depth == 0

            # Verify context was updated
            assert test_agent.context.counter == 1

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_parallel(self, test_agent):
        """Test agent making multiple parallel tool calls"""

        with patch.object(test_agent, "client") as mock_client:
            # First call returns multiple tool calls
            tool_call1 = MockFunctionCall("add_numbers", '{"a": 1, "b": 2}', "call_1")
            tool_call2 = MockFunctionCall("get_counter", "{}", "call_2")
            mock_tool_response = MockOpenAIResponse(function_calls=[tool_call1, tool_call2])
            mock_final_response = MockOpenAIResponse(output_text="I did both calculations!")

            mock_client.responses.create = AsyncMock(
                side_effect=[mock_tool_response, mock_final_response]
            )

            result = await test_agent.run("Do some math")

            assert result.final_output == "I did both calculations!"
            assert len(result.tool_responses) == 2

            # Verify both tools were called
            tool_names = [tr.__class__.__name__ for tr in result.tool_responses]
            assert "ToolResponse[add_numbers]" in tool_names
            assert "ToolResponse[get_counter]" in tool_names

    @pytest.mark.asyncio
    async def test_max_call_depth_respected(self, test_agent):
        """Test that max_call_depth is respected"""

        # Set max_call_depth to 2
        test_agent.max_call_depth = 2

        with patch.object(test_agent, "client") as mock_client:
            # Always return tool calls (would create infinite loop without depth limit)
            tool_call1 = MockFunctionCall("add_numbers", '{"a": 1, "b": 2}', "call_1")
            tool_call2 = MockFunctionCall("get_counter", "{}", "call_2")
            mock_tool_response1 = MockOpenAIResponse(function_calls=[tool_call1])
            mock_tool_response2 = MockOpenAIResponse(function_calls=[tool_call2])
            mock_final_response = MockOpenAIResponse(output_text="Max depth reached!")

            mock_client.responses.create = AsyncMock(
                side_effect=[mock_tool_response1, mock_tool_response2, mock_final_response]
            )

            result = await test_agent.run("Keep calling tools")
            assert result.final_output == "Max depth reached!"
            assert len(result.tool_responses) == 2  # Called twice due to depth limit
            assert mock_client.responses.create.call_count == 3  # 2 tool rounds + 1 final


class TestAgentErrorHandling:
    """Test agent error handling scenarios"""

    @pytest.mark.asyncio
    async def test_openai_api_error(self, test_agent):
        """Test handling of OpenAI API errors"""

        with patch.object(test_agent, "client") as mock_client:
            # Mock API error
            mock_client.responses.create = AsyncMock(side_effect=Exception("API Error"))

            result = await test_agent.run("This will fail")

            assert "OpenAI failed to generate a response: API Error" in result
            # Context should still have the user message
            assert any("This will fail" in str(msg) for msg in test_agent.context._messages)

    @pytest.mark.asyncio
    async def test_tool_execution_error(self, test_agent):
        """Test handling of tool execution errors"""

        with patch.object(test_agent, "client") as mock_client:
            # Tool call that will cause an error
            tool_call = MockFunctionCall("error_tool", '{"should_error": true}')
            mock_tool_response = MockOpenAIResponse(function_calls=[tool_call])
            mock_final_response = MockOpenAIResponse(output_text="I handled the error gracefully")

            mock_client.responses.create = AsyncMock(
                side_effect=[mock_tool_response, mock_final_response]
            )

            result = await test_agent.run("Cause an error")

            assert result.final_output == "I handled the error gracefully"
            assert len(result.tool_responses) == 1

            # Check that error message is in tool response
            tool_response = result.tool_responses[0]
            assert "Tool `error_tool` failed:" in tool_response.output
            assert "This is a test error" in tool_response.output


class TestAgentEmitter:
    """Test agent emitter functionality"""

    @pytest.mark.asyncio
    async def test_emitter_called_for_status_updates(self, test_agent, mock_emitter):
        """Test that emitter is called with proper status updates"""

        test_agent.emitter = mock_emitter

        with patch.object(test_agent, "client") as mock_client:
            mock_response = MockOpenAIResponse(output_text="Test response")
            mock_client.responses.create = AsyncMock(return_value=mock_response)

            result = await test_agent.run("Test message")

            # Check emitter was called
            assert mock_emitter.call_count >= 2  # At least GENERATING and SUCCEDED

            # Check first call was GENERATING
            first_call = mock_emitter.call_args_list[0][0][0]
            assert isinstance(first_call, EmitUpdate)
            assert first_call.status == Status.GENERATING

            # Check last call was SUCCEDED
            last_call = mock_emitter.call_args_list[-1][0][0]
            assert isinstance(last_call, AiUpdate)
            assert last_call.status == Status.SUCCEDED
            assert last_call.message == "Test response"

    @pytest.mark.asyncio
    async def test_emitter_called_for_tool_updates(self, test_agent, mock_emitter):
        """Test that emitter is called for tool execution updates"""

        test_agent.emitter = mock_emitter

        with patch.object(test_agent, "client") as mock_client:
            tool_call = MockFunctionCall("add_numbers", '{"a": 1, "b": 2}')
            mock_tool_response = MockOpenAIResponse(function_calls=[tool_call])
            mock_final_response = MockOpenAIResponse(output_text="Done!")

            mock_client.responses.create = AsyncMock(
                side_effect=[mock_tool_response, mock_final_response]
            )

            result = await test_agent.run("Add numbers")

            # Should have tool processing update
            tool_updates = [
                call[0][0]
                for call in mock_emitter.call_args_list
                if isinstance(call[0][0], ToolUpdate)
            ]
            assert len(tool_updates) >= 1

            tool_update = tool_updates[0]
            assert tool_update.tool_call == "add_numbers"
            assert tool_update.tool_args == {"a": 1, "b": 2}
            assert tool_update.status == Status.PROCESSING


class TestAgentContext:
    """Test agent context handling during execution"""

    @pytest.mark.asyncio
    async def test_context_persists_across_tool_calls(self, test_agent):
        """Test that context changes persist across multiple tool calls"""

        with patch.object(test_agent, "client") as mock_client:
            # First tool call updates context
            tool_call1 = MockFunctionCall("update_data", '{"new_data": "updated"}', "call1")
            # Second tool call should see the updated context
            tool_call2 = MockFunctionCall("get_counter", "{}", "call2")

            mock_tool_response = MockOpenAIResponse(function_calls=[tool_call1, tool_call2])
            mock_tool_response = MockOpenAIResponse(function_calls=[tool_call1, tool_call2])
            mock_final_response = MockOpenAIResponse(output_text="Context updated!")

            mock_client.responses.create = AsyncMock(
                side_effect=[mock_tool_response, mock_final_response]
            )

            result = await test_agent.run("Update and check")

            # Verify context was updated
            assert test_agent.context.test_data == "updated"
            assert len(result.tool_responses) == 2

    @pytest.mark.asyncio
    async def test_messages_accumulated_in_context(self, test_agent):
        """Test that messages are properly accumulated in context"""

        with patch.object(test_agent, "client") as mock_client:
            mock_response = MockOpenAIResponse(output_text="Test response")
            mock_client.responses.create = AsyncMock(return_value=mock_response)

            # Initial message count
            initial_count = len(test_agent.context._messages)

            result = await test_agent.run("Test message")

            # Should have user message and assistant response
            final_count = len(test_agent.context._messages)
            assert final_count == initial_count + 2

            # Check message structure
            messages = test_agent.context._messages[-2:]
            assert messages[0]["role"] == "user"
            assert "Test message" in messages[0]["content"]
            assert messages[1]["role"] == "assistant"
            assert messages[1]["content"] == "Test response"


class TestAgentResponseStructure:
    """Test agent response structure and typing"""

    @pytest.mark.asyncio
    async def test_response_model_structure(self, test_agent):
        """Test that response model has correct structure"""

        with patch.object(test_agent, "client") as mock_client:
            tool_call = MockFunctionCall("add_numbers", '{"a": 1, "b": 1}')
            mock_tool_response = MockOpenAIResponse(function_calls=[tool_call])
            mock_final_response = MockOpenAIResponse(output_text="Final response")

            mock_client.responses.create = AsyncMock(
                side_effect=[mock_tool_response, mock_final_response]
            )

            result = await test_agent.run("Test")

            # Check response structure
            assert hasattr(result, "final_output")
            assert hasattr(result, "tool_responses")
            assert result.final_output == "Final response"
            assert isinstance(result.tool_responses, list)
            assert len(result.tool_responses) == 1

            # Check tool response structure
            tool_resp = result.tool_responses[0]
            assert hasattr(tool_resp, "call_depth")
            assert hasattr(tool_resp, "output")
            assert hasattr(tool_resp, "raw_kwargs")
            assert hasattr(tool_resp, "a")  # Tool parameter
            assert hasattr(tool_resp, "b")  # Tool parameter

    @pytest.mark.asyncio
    async def test_call_method_works(self, test_agent):
        """Test that __call__ method works the same as run"""

        with patch.object(test_agent, "client") as mock_client:
            mock_response = MockOpenAIResponse(output_text="Call test response")
            mock_client.responses.create = AsyncMock(return_value=mock_response)

            # Test __call__ method
            result = await test_agent("Call test")

            assert result.final_output == "Call test response"
            assert mock_client.responses.create.called


class TestAgentReasoningHandling:
    """Test handling of reasoning traces from OpenAI"""

    @pytest.mark.asyncio
    async def test_reasoning_traces_handled(self, test_agent):
        """Test that reasoning traces are properly handled"""

        with patch.object(test_agent, "client") as mock_client:
            # Response with reasoning traces
            mock_response = MockOpenAIResponse(
                output_text="Final answer", reasoning=["Step 1: Think", "Step 2: Analyze"]
            )
            mock_client.responses.create = AsyncMock(return_value=mock_response)

            initial_msg_count = len(test_agent.context._messages)

            result = await test_agent.run("Think about this")

            # Reasoning should be added to context messages
            final_msg_count = len(test_agent.context._messages)
            # User message + 2 reasoning + assistant response = 4 new messages
            assert final_msg_count == initial_msg_count + 4

            # Check reasoning messages were added
            messages = test_agent.context._messages
            reasoning_msgs = [msg for msg in messages if msg.get("type") == "reasoning"]
            assert len(reasoning_msgs) == 2
