"""
Usage Example

from llm_backend.openai_backend import OpenAIBackend
from llm_backend.agent import create_default_tool_agent

llm = OpenAIBackend("gpt-4")
agent = create_default_tool_agent(llm)

result = agent.run("Search Wikipedia for Nikola Tesla.")
print(result)
"""

# llm_backend/agent.py

import json
from typing import Any, Callable, Dict, List

from .base import LLMBackend


class ToolAgent:
    """
    Tool-using agent for function calling with LLMs.

    Supports OpenAI-style function/tool execution and LLM output augmentation.
    """

    def __init__(self, llm: LLMBackend, tools: Dict[str, Callable]):
        """
        Initialize ToolAgent with an LLM backend and dictionary of tools.

        Args:
            llm (LLMBackend): Any backend implementing the LLMBackend interface.
            tools (Dict[str, Callable]): Tool name -> function mapping.
        """
        self.llm = llm
        self.tools = tools

    def run(self, query: str) -> str:
        """
        Execute a query. The LLM decides if it needs to call a function.

        Args:
            query (str): The user query.

        Returns
        -------
            str: Final response from the LLM (with or without tool use).
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. You can use tools if needed.",
            },
            {"role": "user", "content": query},
        ]

        # Step 1: Call LLM with tool schema
        response = self.llm.chat(
            messages, functions=self.get_function_specs(), function_call="auto"
        )

        # Step 2: If simple string response, return
        if isinstance(response, str):
            return response

        # Step 3: Tool usage flow
        function_call = response.get("function_call")
        if not function_call:
            return response.get("content", "No response.")

        name = function_call.get("name")
        args_json = function_call.get("arguments", "{}")

        try:
            args = json.loads(args_json)
        except json.JSONDecodeError:
            return f"Failed to parse arguments for function '{name}'"

        tool = self.tools.get(name)
        if not tool:
            return f"Unknown tool: {name}"

        try:
            tool_output = tool(**args)
        except Exception as e:
            return f"Tool '{name}' execution failed: {e!s}"

        # Step 4: Send tool result back to LLM
        messages.append({"role": "function", "name": name, "content": str(tool_output)})

        return self.llm.chat(messages)

    def get_function_specs(self) -> List[Dict[str, Any]]:
        """
        Convert tools into OpenAI-compatible function specs.

        Returns
        -------
            List[Dict]: JSON schema descriptions of the tools.
        """
        return [
            {
                "name": name,
                "description": func.__doc__ or "",
                "parameters": getattr(
                    func,
                    "openai_schema",
                    {"type": "object", "properties": {}, "required": []},
                ),
            }
            for name, func in self.tools.items()
        ]


# -------------------------------
# Tool Imports & Agent Factory
# -------------------------------
from llm_backend.utils.tools import search_wikipedia

tools = {
    "search_wikipedia": search_wikipedia,
}


def create_default_tool_agent(llm: LLMBackend) -> ToolAgent:
    """
    Helper to quickly create a tool-enabled agent with default tools.
    """
    return ToolAgent(llm=llm, tools=tools)
