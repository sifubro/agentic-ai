'''
https://chatgpt.com/c/692afbd8-09cc-832a-a37a-6c837ba2c2e4
https://chatgpt.com/c/692afbd8-09cc-832a-a37a-6c837ba2c2e4
https://chatgpt.com/c/692afbd8-09cc-832a-a37a-6c837ba2c2e4

✅ Part 1 — What does @tool do in LangChain?

In LangChain, the decorator:

from langchain.tools import tool


is used to turn a Python function into an AI-callable “tool” that a model (GPT-4, Gemini, etc.) can invoke during agent execution.

When you write:

@tool
def add(a: int, b: int) -> int:
    return a + b


LangChain automatically converts add() into a Tool object with:

a name ("add")

a description (from the docstring)

a schema for arguments (from the type hints)

a run() method the agent can call

Then an LLM agent can use the tool:

Thought: I should call the add tool
Action: add
Action Input: {"a": 5, "b": 7}


LangChain runs add(5, 7) and feeds back the result.

⭐ Example: A simple LangChain tool
from langchain.tools import tool

@tool
def greet(name: str) -> str:
    """Greets a person by name."""
    return f"Hello, {name}!"


LLMs will see the tool described in JSON schema:

{
  "name": "greet",
  "description": "Greets a person by name.",
  "parameters": {
    "name": { "type": "string" }
  }
}


Agents can then select and execute it as needed.

🔍 What @tool REALLY does under the hood

Conceptually, the LangChain @tool decorator:

Reads function signature
→ extracts argument names and types

Reads docstring
→ used as the tool’s description

Creates a Tool object
→ wraps the function

Exposes JSON schema to the model
→ so LLM knows how to call it

Controls how the agent invokes the function
→ sync/async variations, error handling, serialization, etc.

So the decorator is basically metadata → wrapper → Tool object.

🧪 Part 2 — Implementing @tool From Scratch

Below is a minimal but functional recreation of LangChain’s @tool.

This shows exactly what the real decorator does conceptually.

✅ 1. A simple Tool class
import inspect
import json

class Tool:
    def __init__(self, name: str, description: str, func):
        self.name = name
        self.description = description
        self.func = func
        self.schema = self._build_schema(func)

    def _build_schema(self, func):
        sig = inspect.signature(func)
        params = {}
        for name, param in sig.parameters.items():
            annotation = param.annotation
            type_name = annotation.__name__ if annotation != inspect._empty else "string"
            params[name] = {"type": type_name}

        return {
            "name": self.name,
            "description": self.description,
            "parameters": params,
        }

    def run(self, **kwargs):
        return self.func(**kwargs)


This captures:

the function

the schema

a .run() method for execution

✅ 2. The @tool decorator implemented manually
def tool(func):
    """
    Minimal re-implementation of LangChain's @tool decorator.
    Turns a function into a Tool instance.
    """
    name = func.__name__
    description = func.__doc__ or ""
    return Tool(name, description, func)


This means any decorated function returns a Tool object instead of a raw function.

🎉 3. Example Usage
@tool
def multiply(a: int, b: int):
    """Multiply two integers."""
    return a * b


Now multiply is a Tool instance:

print(multiply.name)        # "multiply"
print(multiply.description) # "Multiply two integers."
print(json.dumps(multiply.schema, indent=2))


Schema output:

{
  "name": "multiply",
  "description": "Multiply two integers.",
  "parameters": {
    "a": {"type": "int"},
    "b": {"type": "int"}
  }
}


Run it:

result = multiply.run(a=6, b=7)
print(result)   # 42

🧠 Want closer-to-LangChain behavior?

LangChain’s real @tool handles:

✔ async functions
✔ formatting input/output
✔ Pydantic models for argument validation
✔ tool names & overrides
✔ structured output
✔ error handling
✔ docstring parsing
✔ agent integration

If you'd like, I can also:

🔧 Extend the custom @tool to support async
📘 Generate full JSON Schema like LangChain
🧩 Build an agent loop that can call these tools
⚙️ Implement structured tool calls like OpenAI function calling

Just tell me how deep you want to go.

'''