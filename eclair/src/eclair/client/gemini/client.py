"""
Gemini MCP Client

Client that connects Gemini to the Eclair MCP Server.
"""
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

    def __init__(self, mcp_server_url: str = "http://localhost:8080/mcp"):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai is not installed. Install it with: pip install google-genai")
        super().__init__(mcp_server_url)

    def _build_llm_client(self):
        """Construct the Gemini client."""
        if self.api_key:
            self.llm_client = genai.Client(api_key=self.api_key)
        else:
            print("No Gemini API key found. Only MCP functionality will be available.")

    def _convert_tools(self, mcp_tools):
        """Wrap MCP tools as a single Gemini Tool of function declarations."""
        declarations = [
            types.FunctionDeclaration(
                name=tool.name,
                description=tool.description or "",
                parameters=tool.inputSchema,
            )
            for tool in mcp_tools
        ]
        return [types.Tool(function_declarations=declarations)]

    def _init_history(self, prompt: str):
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

    def _parse_tool_calls(self, response) -> list[ToolCall]:
        return [
            ToolCall(id=call.id or "", name=call.name, arguments=dict(call.args or {}))
            for call in (response.function_calls or [])
        ]

    def _final_text(self, response) -> str:
        return response.text

    def _append_turn(self, history, response, results):
        # The model's turn holds the function-call parts; append it as-is.
        history.append(response.candidates[0].content)
        parts = [
            types.Part.from_function_response(name=call.name, response=self._result_payload(result))
            for call, result in results
        ]
        history.append(types.Content(role="user", parts=parts))


# Example usage (only if running this file directly)
if __name__ == "__main__":
    import asyncio

    async def main():
        client = GeminiMCPClient()
        await client.initialize()

        # Example: Search for datasets
        print("Searching for image datasets...")
        results = await client.search_datasets("image classification")
        print(f"Found datasets: {results}")

        await client.close()

    asyncio.run(main())
