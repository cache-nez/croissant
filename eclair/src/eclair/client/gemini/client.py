"""
Gemini MCP Client

Client that connects Gemini to the Eclair MCP Server.
"""
from __future__ import annotations

from ..llm import LlmMcpClient, Provider, ToolCall

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class GeminiMCPClient(LlmMcpClient):
    """Client that connects Gemini to the Eclair MCP Server."""

    PROVIDER = Provider.GEMINI

    def __init__(self, mcp_server_url: str = "http://localhost:8080/mcp") -> None:
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai is not installed. Install it with: pip install google-genai")
        super().__init__(mcp_server_url)

    def _build_llm_client(self) -> None:
        """Construct the Gemini client."""
        if self.api_key:
            self.llm_client = genai.Client(api_key=self.api_key)
        else:
            print("No Gemini API key found. Only MCP functionality will be available.")

    # JSON-Schema keywords that Eclair MCP servers emits but Gemini's function
    # declaration schema (an OpenAPI 3.0 subset) rejects. Left in place they
    # trigger a 400 INVALID_ARGUMENT from the API.
    _UNSUPPORTED_SCHEMA_KEYS = frozenset(
        {"additionalProperties", "$schema", "$defs", "$ref", "$id", "definitions"}
    )

    def _sanitize_schema(self, schema):
        """Recursively drop schema keywords Gemini's tool schema doesn't accept."""
        if isinstance(schema, dict):
            return {
                key: self._sanitize_schema(value)
                for key, value in schema.items()
                if key not in self._UNSUPPORTED_SCHEMA_KEYS
            }
        if isinstance(schema, list):
            return [self._sanitize_schema(item) for item in schema]
        return schema

    def _mcp_tools_to_provider_format(self, mcp_tools) -> list[types.Tool]:
        """Wrap MCP tools as a single Gemini Tool of function declarations."""
        declarations = [
            types.FunctionDeclaration(
                name=tool.name,
                description=tool.description or "",
                parameters=self._sanitize_schema(tool.inputSchema),
            )
            for tool in mcp_tools
        ]
        return [types.Tool(function_declarations=declarations)]

    def _initial_history(self, prompt: str) -> list[types.Content]:
        return [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]

    def _generate_turn(self, history, tools, temperature: float):
        return self.llm_client.models.generate_content(
            model=self.model_name,
            contents=history,
            config=types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=self.system_prompt,
                tools=tools,
            ),
        )

    def _parse_tool_calls(self, llm_response) -> list[ToolCall]:
        return [
            ToolCall(id=call.id or "", name=call.name, arguments=dict(call.args or {}))
            for call in (llm_response.function_calls or [])
        ]

    def _parse_final_text(self, llm_response) -> str:
        return llm_response.text

    def _append_turn(self, history, llm_response, tool_call_results) -> None:
        # The model's turn holds the function-call parts; append it as-is.
        history.append(llm_response.candidates[0].content)
        parts = [
            types.Part.from_function_response(name=call.name, response=self._result_payload(result))
            for call, result in tool_call_results
        ]
        history.append(types.Content(role="user", parts=parts))


# Example usage (only if running this file directly)
if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        client = GeminiMCPClient()
        await client.initialize()

        # Example: Search for datasets
        print("Searching for image datasets...")
        results = await client.search_datasets("image classification")
        print(f"Found datasets: {results}")

        await client.close()

    asyncio.run(main())
