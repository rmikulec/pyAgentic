# Structured Outputs

PyAgentic can automatically structure the final output of an agent into a Pydantic model. This is useful when you need the agent's response to follow a specific schema, making it easy to validate and use in downstream applications.

## Defining a Response Model

To specify a structured output, define a Pydantic model and assign it to the `__response_format__` attribute of your agent.

```python
from pyagentic import Agent, tool
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int
    desc: str
    city: str
    state: str

class UserParsingAgent(Agent):
    __system_message__ = "You are an AI that is an expert at parsing user information from any text"
    __response_format__ = UserInfo

    @tool("Returns the city and state of a given zipcode")
    def zipcode_lookup(self, zipcode: str) -> str:
        ...
```

When you run this agent, the `final_output` of the response will be an instance of the `UserInfo` model, rather than a string.

```python
await agent("Im John, im 28 and my postal code is 10012")
```

```json
{
  "final_output": {
    "name": "John",
    "age": 28,
    "desc": "User is John, aged 28, from NYC.",
    "city": "NYC",
    "state": "New York"
  },
  "tool_responses": [
    {
      "raw_kwargs": "{\"zipcode\":\"10012\"}",
      "call_depth": 0,
      "output": "NYC, New York",
      "zipcode": "10012"
    }
  ]
}
```

## How it Works

Under the hood, PyAgentic uses the `parse` capability with a given `__response_format__`, this takes
advantage of the LLMs structured output feature (if supported). Pyagentic also automatically updates its own
`__response_model__` to ensure that the agent's output is always expected.

```json
{
    "$defs": {
        "ToolResponse_get_user_": {
            "properties": {
                "raw_kwargs": {
                    "title": "Raw Kwargs",
                    "type": "string"
                },
                "call_depth": {
                    "title": "Call Depth",
                    "type": "integer"
                },
                "output": {
                    "title": "Output"
                },
                "user_id": {
                    "default": null,
                    "title": "User Id",
                    "type": "string"
                }
            },
            "required": [
                "raw_kwargs",
                "call_depth",
                "output"
            ],
            "title": "ToolResponse[get_user]",
            "type": "object"
        },
        "UserInfo": {
            "properties": {
                "name": {
                    "title": "Name",
                    "type": "string"
                },
                "age": {
                    "title": "Age",
                    "type": "integer"
                },
                "desc": {
                    "title": "Desc",
                    "type": "string"
                }
            },
            "required": [
                "name",
                "age",
                "desc"
            ],
            "title": "UserInfo",
            "type": "object"
        }
    },
    "properties": {
        "final_output": {
            "$ref": "#/$defs/UserInfo"
        },
        "tool_responses": {
            "items": {
                "$ref": "#/$defs/ToolResponse_get_user_"
            },
            "title": "Tool Responses",
            "type": "array"
        }
    },
    "required": [
        "final_output",
        "tool_responses"
    ],
    "title": "UserParsingAgentResponse",
    "type": "object"
}
```

This ensures that the agent's output is a valid JSON object that conforms to the specified Pydantic model, providing type safety and structured data you can rely on.
