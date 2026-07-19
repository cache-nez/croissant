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
        """Generate a response with Gemini, falling back to a basic search."""
        try:
            print("Thinking...")
            full_prompt = self._build_prompt(prompt)
            model = self.llm_client.GenerativeModel(self.model_name)
            response = model.generate_content(
                full_prompt,
                generation_config=self.llm_client.types.GenerationConfig(temperature=temperature),
            )
            return response.text

        except Exception:
            # Fallback: try a simpler approach with a basic dataset search.
            try:
                search_terms = self._extract_search_terms(prompt)
                if search_terms:
                    search_result = await self.mcp_client.call_tool("search-datasets", {"query": search_terms})

                    enhanced_prompt = f"""As a data scientist AI assistant, I searched for datasets related to '{search_terms}' based on your request: "{prompt}"

I found these datasets:
{search_result}

Let me analyze these results and provide recommendations."""

                    model = self.llm_client.GenerativeModel(self.model_name)
                    response = model.generate_content(
                        enhanced_prompt,
                        generation_config=self.llm_client.types.GenerationConfig(temperature=temperature),
                    )
                    return response.text

            except Exception:
                # Final fallback: general guidance without the dataset server.
                model = self.llm_client.GenerativeModel(self.model_name)
                response = model.generate_content(
                    f"I apologize, but I'm having trouble accessing the dataset server right now. However, I can provide general guidance about: {prompt}",
                    generation_config=self.llm_client.types.GenerationConfig(temperature=temperature),
                )
                return f"⚠️ Dataset server unavailable. General guidance:\n\n{response}"

    def _extract_search_terms(self, prompt: str) -> str:
        """Extract relevant search terms from user prompt."""
        # Enhanced search term extraction based on the sophisticated system prompt
        prompt_lower = prompt.lower()

        # Look for specific domain keywords
        if any(keyword in prompt_lower for keyword in ["image", "classification", "computer vision", "cv"]):
            if "animal" in prompt_lower or "cat" in prompt_lower or "dog" in prompt_lower:
                return "image classification animals"
            elif "medical" in prompt_lower:
                return "medical image classification"
            else:
                return "image classification"
        elif any(keyword in prompt_lower for keyword in ["medical", "health", "disease", "clinical"]):
            return "medical health"
        elif any(keyword in prompt_lower for keyword in ["mnist", "digit", "handwritten"]):
            return "mnist handwritten digits"
        elif any(keyword in prompt_lower for keyword in ["nlp", "language", "text", "sentiment"]):
            return "natural language processing text"
        elif any(keyword in prompt_lower for keyword in ["climate", "weather", "temperature"]):
            return "climate weather data"
        else:
            # Extract meaningful words
            words = prompt_lower.split()
            meaningful_words = [w for w in words if len(w) > 3 and w not in [
                "find", "search", "datasets", "about", "with", "that", "have", "need", "want", "looking"
            ]]
            return " ".join(meaningful_words[:3]) if meaningful_words else "data"


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
