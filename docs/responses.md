# Agent Responses

PyAgentic agents return structured response objects that contain both the natural language output and detailed information about what happened during execution. This makes it easy to build applications that need both human-readable responses and programmatic access to execution details.

## Response Structure

Every agent response is a Pydantic model with a predictable structure:

- `final_output: str` - The natural language response from the LLM
- `tool_responses: List[ToolResponse]` - Details of any tools called (if agent has tools)
- `agent_responses: List[AgentResponse]` - Responses from linked agents (if agent has linked agents)

The response model is predetermined when the agent class is defined. Each agent automatically gets its own response class that knows exactly what tools and linked agents it can use. This means you get full type safety and IDE autocompletion before you ever run the agent.

## Basic Response

Every agent response includes a `final_output` field containing the LLM's natural language response:

```python
response = await agent("Hello")
response.final_output  # "Hi there! How can I help?"
```

Since responses are Pydantic models, you can serialize them to JSON, validate them, and integrate them seamlessly with FastAPI or other frameworks.

## Tool Responses

When agents use tools, the response automatically includes structured information about each tool call. All tool parameters become accessible as typed fields on the response:

```python
class EmailAgent(BaseAgent):
    @tool("Send email")
    def send_email(self, to: str, urgent: bool = False) -> str: ...

response = await agent("Send urgent email to john about moving to our next apartment")
```

```json
{
  "final_output": "I've sent the email to John. Let me know if there's anything else you need!",
  "tool_responses": [
    {
      "raw_kwargs": "{\"to\":\"john@example.com\",\"message\":\"Subject: Moving Forward with the Next Apartment\\n\\nHi John,\\n\\nI hope this message finds you well. I wanted to discuss our plans for moving to the next apartment. Please let me know when would be a good time for us to chat or meet to go over the specifics.\\n\\nLooking forward to hearing from you soon.\\n\\nBest,\\n[Your Name]\",\"urgent\":false}",
      "call_depth": 0,
      "output": "Email sent",
      "to": "john@example.com",
      "message": "Subject: Moving Forward with the Next Apartment\n\nHi John,\n\nI hope this message finds you well. I wanted to discuss our plans for moving to the next apartment. Please let me know when would be a good time for us to chat or meet to go over the specifics.\n\nLooking forward to hearing from you soon.\n\nBest,\n[Your Name]",
      "urgent": false
    }
  ]
}
```

This gives you both the conversational response and programmatic access to exactly what the agent did.

## Multiple Tools

PyAgentic automatically generates response schemas that can handle any combination of tools your agent might use. The response type is created at class definition time and includes proper typing for all possible tool combinations:

```python
class CustomerAgent(BaseAgent):
    @tool("Get customer")
    def get_customer(self, email: str): ...

    @tool("Update status")
    def update_status(self, id: int, status: str): ...

# Response automatically includes fields for whichever tools were called
```

You can access any tool's parameters through the `tool_responses` list, with full type safety and IDE autocompletion.

## Linked Agents

When agents call other agents, their complete responses are nested within the parent response. This creates a tree structure that captures the full execution flow:

```python
class ReportAgent(BaseAgent):
    database: DatabaseAgent

    @tool("Create chart")
    def make_chart(self, type: str): ...

response = await agent("Create sales chart")
response.final_output                    # Main agent response
response.tool_responses[0].type          # "sales"
response.agent_responses[0].final_output # Database agent response
```

This nested structure lets you trace exactly what each agent did and access the results of any sub-agent calls. The parent agent's response includes both its own tool calls and the complete responses from any linked agents it used.
