"""
Claude MCP Client

Client that connects Claude to the Eclair MCP Server.
"""
from ..llm import LlmMcpClient, Provider

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False


class ClaudeMCPClient(LlmMcpClient):
    """Client that connects Claude to the Eclair MCP Server."""

    PROVIDER = Provider.CLAUDE

    def __init__(self, mcp_server_url: str = "http://localhost:8080/mcp"):
        if not CLAUDE_AVAILABLE:
            raise ImportError("anthropic is not installed. Install it with: pip install anthropic")
        super().__init__(mcp_server_url)

    def _load_config(self):
        """Load the shared model settings plus Claude's max_tokens."""
        super()._load_config()
        self.max_tokens = self._provider_config.get("max_tokens", 4096)

    def _build_llm_client(self):
        """Construct the Anthropic client."""
        if self.api_key:
            self.llm_client = anthropic.Anthropic(api_key=self.api_key)
        else:
            print("No Claude API key found. Only MCP functionality will be available.")

    async def _generate(self, user_prompt: str, temperature: float) -> str:
        """Generate a response with Claude."""
        try:
            full_prompt = self._build_prompt(user_prompt)
            response = self.llm_client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=temperature,
                system=self.system_prompt,
                messages=[{"role": "user", "content": full_prompt}],
            )
            return response.content[0].text

        except Exception as e:
            return f"⚠️ Claude API error: {e}"


# Example usage (only if running this file directly)
if __name__ == "__main__":
    import asyncio

    async def main():
        try:
            client = ClaudeMCPClient()
            await client.initialize()

            # Example: Search for datasets
            print("Searching for image datasets...")
            results = await client.search_datasets("image classification")
            print(f"Found datasets: {results}")

            await client.close()
        except ImportError as e:
            print(f"Claude client not available: {e}")

    asyncio.run(main())
