"""
Local LLM MCP Client

Client that connects a locally running model to the Eclair MCP Server through an
OpenAI-compatible API (Ollama, vLLM, LM Studio, llama.cpp, LiteLLM, ...).
"""
import json

from ..llm import LlmMcpClient, Provider, ToolCall

try:
    from openai import OpenAI
    LOCAL_AVAILABLE = True
except ImportError:
    LOCAL_AVAILABLE = False

DEFAULT_BASE_URL = "http://localhost:11434/v1"


class LocalMCPClient(LlmMcpClient):
    """Client that connects a local, OpenAI-compatible model to the Eclair MCP Server."""

    PROVIDER = Provider.LOCAL

    def __init__(self, mcp_server_url: str = "http://localhost:8080/mcp"):
        if not LOCAL_AVAILABLE:
            raise ImportError("openai is not installed. Install it with: pip install openai")
        super().__init__(mcp_server_url)

    def _load_config(self):
        """Load the shared model settings plus the OpenAI-compatible base URL."""
        super()._load_config()
        self.base_url = self._provider_config.get("base_url", DEFAULT_BASE_URL)

    def _build_llm_client(self):
        """Construct the OpenAI client pointed at the local server.

        Local servers usually need no real key, so the client is always built
        (a placeholder key is used when none is set).
        """
        self.llm_client = OpenAI(base_url=self.base_url, api_key=self.api_key or "not-needed")

    def _convert_tools(self, mcp_tools):
        """Convert MCP tools to OpenAI function-tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
            for tool in mcp_tools
        ]

    def _init_history(self, prompt: str):
        # OpenAI has no separate system field; the system prompt is the first message.
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

    def _generate_turn(self, history, tools, temperature: float):
        return self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=history,
            tools=tools,
            temperature=temperature,
        )

    def _parse_tool_calls(self, response) -> list[ToolCall]:
        message = response.choices[0].message
        return [
            ToolCall(
                id=call.id,
                name=call.function.name,
                # OpenAI returns arguments as a JSON string, not a dict.
                arguments=json.loads(call.function.arguments or "{}"),
            )
            for call in (message.tool_calls or [])
        ]

    def _final_text(self, response) -> str:
        # content is None on a tool-only turn.
        return response.choices[0].message.content or ""

    def _append_turn(self, history, response, results):
        # Echo the assistant's tool-call turn, then answer with one tool message per call.
        history.append(response.choices[0].message.model_dump(exclude_none=True))
        for call, result in results:
            history.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(self._result_payload(result)),
            })


# Example usage (only if running this file directly)
if __name__ == "__main__":
    import asyncio

    async def main():
        try:
            client = LocalMCPClient()
            await client.initialize()

            # Example: Search for datasets
            print("Searching for image datasets...")
            results = await client.search_datasets("image classification")
            print(f"Found datasets: {results}")

            await client.close()
        except ImportError as e:
            print(f"Local client not available: {e}")

    asyncio.run(main())
