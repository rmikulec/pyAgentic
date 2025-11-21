# Structured Outputs

PyAgentic can automatically structure the final output of an agent into a Pydantic model. This is useful when you need the agent's response to follow a specific schema, making it easy to validate and use in downstream applications.

## Defining a Response Model

To specify a structured output, define a Pydantic model and assign it to the `__response_format__` attribute of your agent.

```python
from pyagentic import BaseAgent, tool
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int
    desc: str
    city: str
    state: str

class UserParsingAgent(BaseAgent):
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
    "desc": "Postal code was 10012",
    "city": "NYC",
    "state": "New York"
  },
  "provider_info": {
    "name": "openai",
    "model": "gpt-4o",
    "attributes": {}
  },
  "tool_responses": [
    {
      "raw_kwargs": "{\"zipcode\":\"10012\"}",
      "call_depth": 0,
      "output": "NYC, New York",
      "zipcode": "10012"
    }
  ],
  "state": {
    "instructions": "You are an AI that is an expert at parsing user information from any text",
    "input_template": null
  }
}
```

## How it Works

Under the hood, PyAgentic uses the `parse` capability with a given `__response_format__`, this takes
advantage of the LLMs structured output feature (if supported). Pyagentic also automatically updates its own
`__response_model__` to ensure that the agent's output is always expected.

```json
{
  "$defs": {
    "AgentState_UserParsingAgent_": {
      "properties": {
        "instructions": {
          "title": "Instructions",
          "type": "string"
        },
        "input_template": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Input Template"
        }
      },
      "required": [
        "instructions"
      ],
      "title": "AgentState[UserParsingAgent]",
      "type": "object"
    },
    "ProviderInfo": {
      "properties": {
        "name": {
          "title": "Name",
          "type": "string"
        },
        "model": {
          "title": "Model",
          "type": "string"
        },
        "attributes": {
          "additionalProperties": true,
          "default": null,
          "title": "Attributes",
          "type": "object"
        }
      },
      "required": [
        "name",
        "model"
      ],
      "title": "ProviderInfo",
      "type": "object"
    },
    "ToolResponse_zipcode_lookup_": {
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
        "zipcode": {
          "default": null,
          "title": "Zipcode",
          "type": "string"
        }
      },
      "required": [
        "raw_kwargs",
        "call_depth",
        "output"
      ],
      "title": "ToolResponse[zipcode_lookup]",
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
        },
        "city": {
          "title": "City",
          "type": "string"
        },
        "state": {
          "title": "State",
          "type": "string"
        }
      },
      "required": [
        "name",
        "age",
        "desc",
        "city",
        "state"
      ],
      "title": "UserInfo",
      "type": "object"
    }
  },
  "properties": {
    "final_output": {
      "$ref": "#/$defs/UserInfo"
    },
    "provider_info": {
      "$ref": "#/$defs/ProviderInfo"
    },
    "tool_responses": {
      "items": {
        "$ref": "#/$defs/ToolResponse_zipcode_lookup_"
      },
      "title": "Tool Responses",
      "type": "array"
    },
    "state": {
      "$ref": "#/$defs/AgentState_UserParsingAgent_"
    }
  },
  "required": [
    "final_output",
    "provider_info",
    "tool_responses",
    "state"
  ],
  "title": "UserParsingAgentResponse",
  "type": "object"
}
```

This ensures that the agent's output is a valid JSON object that conforms to the specified Pydantic model, providing type safety and structured data you can rely on.
