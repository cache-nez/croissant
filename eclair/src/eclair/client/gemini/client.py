"""
Gemini MCP Client

Client that connects Gemini to the Eclair MCP Server.
"""
from ..llm import LlmMcpClient, Provider

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

    async def _generate(self, prompt: str, temperature: float) -> str:
        """Generate a response with Gemini."""
        full_prompt = self._build_prompt(prompt)
        response = self.llm_client.models.generate_content(
            model=self.model_name,
            contents=full_prompt,
            config=types.GenerateContentConfig(temperature=temperature),
        )
        return response.text

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
