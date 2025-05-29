"""
Usage Example (Async)

import asyncio
from llm_backend.openai_backend import OpenAIBackend
from llm_backend.agent_async import create_default_tool_agent_async

async def main():
    llm = OpenAIBackend("gpt-4")  # Must support async
    agent = create_default_tool_agent_async(llm)
    result = await agent.run("Search Wikipedia for Nikola Tesla")
    print(result)

asyncio.run(main())
"""

# llm_backend/agent_async.py

import json
from collections.abc import Awaitable
from typing import Any, Callable, Dict, List

from .base import LLMBackend


class ToolAgentAsync:
    """
    Async version of ToolAgent for LLMs supporting asynchronous API calls.
    """

    def __init__(self, llm: LLMBackend, tools: Dict[str, Callable]):
        """
        Initialize ToolAgentAsync with an async-capable LLM backend and tools.

        Args:
            llm (LLMBackend): Backend supporting async chat_async method.
            tools (Dict[str, Callable]): Tool name -> function mapping.
        """
        self.llm = llm
        self.tools = tools

    async def run(self, query: str) -> str:
        """
        Execute a query asynchronously. Uses LLM's function-calling to determine tool use.

        Args:
            query (str): The user query.

        Returns
        -------
            str: Final response from LLM (with or without tool output).
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. You can use tools if needed.",
            },
            {"role": "user", "content": query},
        ]

        # Step 1: Call LLM with tool schema
        response = await self.llm.chat_async(
            messages, functions=self.get_function_specs(), function_call="auto"
        )

        # Step 2: If string, return directly
        if isinstance(response, str):
            return response

        # Step 3: Tool usage
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
            # Optional: Support async tools (if needed)
            if isinstance(tool, Awaitable) or getattr(tool, "__await__", False):
                tool_output = await tool(**args)
            else:
                tool_output = tool(**args)
        except Exception as e:
            return f"Tool '{name}' execution failed: {e!s}"

        # Step 4: Send tool result back to LLM
        messages.append({"role": "function", "name": name, "content": str(tool_output)})

        return await self.llm.chat_async(messages)

    def get_function_specs(self) -> List[Dict[str, Any]]:
        """
        Return OpenAI-compatible function specs for tools.

        Returns
        -------
            List[Dict]: Tool metadata for function calling.
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


def create_default_tool_agent_async(llm: LLMBackend) -> ToolAgentAsync:
    """
    Factory function to create ToolAgentAsync with default tools.
    """
    return ToolAgentAsync(llm=llm, tools=tools)
