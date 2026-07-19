"""
Gemini MCP Client

Client that connects Gemini to the Eclair MCP Server.
"""
import warnings

# Suppress known protobuf enum warnings from Google AI library
warnings.filterwarnings("ignore", message="Unrecognized FinishReason enum value", category=UserWarning)
# Suppress protobuf deprecation warning for including_default_value_fields
warnings.filterwarnings("ignore", message=".*including_default_value_fields.*", category=DeprecationWarning)

from ..llm import LlmMcpClient, Provider

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class GeminiMCPClient(LlmMcpClient):
    """Client that connects Gemini to the Eclair MCP Server."""

    PROVIDER = Provider.GEMINI

    def __init__(self, mcp_server_url: str = "http://localhost:8080/mcp"):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai is not installed. Install it with: pip install google-generativeai")
        super().__init__(mcp_server_url)

    def _build_llm_client(self):
        """Configure the Gemini SDK; store the configured module as the client."""
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.llm_client = genai  # the configured module is the client handle
        else:
            print("No Gemini API key found. Only MCP functionality will be available.")

    async def _generate(self, prompt: str, temperature: float) -> str:
        """Generate a response with Gemini."""
        full_prompt = self._build_prompt(prompt)
        model = self.llm_client.GenerativeModel(self.model_name)
        response = model.generate_content(
            full_prompt,
            generation_config=self.llm_client.types.GenerationConfig(temperature=temperature),
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
