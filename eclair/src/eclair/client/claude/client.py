"""
Claude MCP Client

Client that connects Claude to the Eclair MCP Server.
"""
from __future__ import annotations

import json

from ..llm import LlmMcpClient, Provider, ToolCall
from ..llm import logger as loop_logger

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False


class ClaudeMCPClient(LlmMcpClient):
    """Client that connects Claude to the Eclair MCP Server."""

    PROVIDER = Provider.CLAUDE

    def __init__(self, mcp_server_url: str = "http://localhost:8080/mcp") -> None:
        if not CLAUDE_AVAILABLE:
            raise ImportError("anthropic is not installed. Install it with: pip install anthropic")
        super().__init__(mcp_server_url)

    def _load_config(self) -> None:
        """Load the shared model settings plus Claude's max_tokens."""
        super()._load_config()
        self.max_tokens = self._provider_config.get("max_tokens", 4096)

    def _build_llm_client(self) -> None:
        """Construct the Anthropic client."""
        if self.api_key:
            self.llm_client = anthropic.Anthropic(api_key=self.api_key)
        else:
            print("No Claude API key found. Only MCP functionality will be available.")

    def _mcp_tools_to_provider_format(self, mcp_tools) -> list[dict]:
        """Convert MCP tools to Anthropic tool definitions."""
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            for tool in mcp_tools
        ]

    def _initial_history(self, prompt: str) -> list[dict]:
        return [{"role": "user", "content": prompt}]

    def _generate_turn(self, history, tools, temperature: float):
        return self.llm_client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=temperature,
            system=self.system_prompt,
            tools=tools,
            messages=history,
        )

    def _parse_tool_calls(self, llm_response) -> list[ToolCall]:
        return [
            ToolCall(id=block.id, name=block.name, arguments=block.input)
            for block in llm_response.content
            if block.type == "tool_use"
        ]

    def _parse_final_text(self, llm_response) -> str:
        return "".join(block.text for block in llm_response.content if block.type == "text")

    def _append_turn(self, history, llm_response, tool_call_results) -> None:
        # Echo the assistant's tool-use turn, then answer with tool_result blocks.
        history.append({"role": "assistant", "content": llm_response.content})
        history.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": json.dumps(self._result_payload(result)),
                }
                for call, result in tool_call_results
            ],
        })


# Example usage (only if running this file directly)
if __name__ == "__main__":
    import argparse
    import asyncio
    import logging

    parser = argparse.ArgumentParser(description="Run the Claude MCP client")
    parser.add_argument(
        "--log-level",
        default="debug",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level for the tool-call loop (default: debug)",
    )
    cli_args = parser.parse_args()

    # Configure only the tool-call loop's logger, so third-party libraries
    # (httpcore, anthropic, ...) keep their default level instead of flooding
    # the output. basicConfig would target the root logger and enable everything.
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    loop_logger.setLevel(getattr(logging, cli_args.log_level.upper()))
    loop_logger.addHandler(handler)

    async def main() -> None:
        try:
            client = ClaudeMCPClient()
            await client.initialize()

            results = await client.ask_llm_with_tools("check if Eclair server is up")
            print(f"{results}")

            await client.close()
        except ImportError as e:
            print(f"Claude client not available: {e}")

    asyncio.run(main())
