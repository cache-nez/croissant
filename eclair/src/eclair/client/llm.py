"""
LLM MCP Client

Shared base class for clients that pair an Eclair MCP server with an LLM.
"""

import inspect
import json
import os
from enum import Enum

from .client import EclairClient

# Load environment variables from a .env file, if present, before any client is
# constructed (so os.getenv picks up .env-defined API keys). Falls back to the
# real process environment when python-dotenv is not installed.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class Provider(Enum):
    """A supported LLM provider and its per-provider constants.

    Each member carries the config.json section key, the environment variable
    holding the API key, and the system-prompt filename (looked up in the
    concrete subclass's own package directory).
    """

    GEMINI = ("gemini", "GEMINI_API_KEY", "gemini.md")
    CLAUDE = ("claude", "CLAUDE_API_KEY", "claude.md")

    def __init__(self, config_key: str, env_var: str, system_prompt_file: str):
        self.config_key = config_key
        self.env_var = env_var
        self.system_prompt_file = system_prompt_file


# Description of the MCP tools, embedded in the prompt so the LLM knows what is
# available. Shared across all providers.
_TOOL_LIST = """Available MCP Tools:
- search-datasets: Search for datasets using a query string
- serve-croissant: Get Croissant metadata for a dataset
- download-dataset: Download dataset files
- datasets-preview-url: Get preview URLs for datasets
- validate-croissant: Validate Croissant metadata"""

_FALLBACK_SYSTEM_PROMPT = (
    "You are a helpful data scientist AI assistant. You have access to MCP tools "
    "for finding and analyzing datasets. Always try to use the available tools to "
    "search for and analyze real data to answer user questions."
)


class LlmMcpClient(EclairClient):
    """Base client that connects an LLM to the Eclair MCP Server.

    Adds LLM scaffolding (config + system-prompt loading, API-key resolution,
    the tool-augmented ask_llm_with_tools flow) on top of the plain MCP
    transport provided by EclairClient. All MCP tool wrappers are inherited
    unchanged.
    """

    # Set by each concrete subclass, e.g. PROVIDER = Provider.GEMINI.
    PROVIDER: Provider = None

    def __init__(self, mcp_server_url: str = "http://localhost:8080/mcp"):
        super().__init__(mcp_server_url)
        # Key is resolved from the environment only (never passed in code).
        self.api_key = os.getenv(self.PROVIDER.env_var)
        self.llm_client = None

        self._load_config()
        self._load_system_prompt()

    def _load_config(self):
        """Load model settings from the provider's section of config.json."""
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.json")
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            self._provider_config = config.get(self.PROVIDER.config_key, {})
        except Exception as e:
            print(f"Warning: Could not load config.json: {e}")
            self._provider_config = {}

        self.model_name = self._provider_config.get("model")
        self.default_temperature = self._provider_config.get("temperature", 0.3)

    def _load_system_prompt(self):
        """Load the system prompt from the subclass package's prompt file."""
        # Resolve relative to the concrete subclass module, so gemini.md lives in
        # the gemini/ package and claude.md in the claude/ package.
        module_dir = os.path.dirname(inspect.getfile(self.__class__))
        system_prompt_path = os.path.join(module_dir, self.PROVIDER.system_prompt_file)
        try:
            with open(system_prompt_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Keep the core instructions but drop markdown headers.
            cleaned_lines = [line for line in content.split("\n") if not line.startswith("#")]
            self.system_prompt = "\n".join(cleaned_lines).strip()
        except Exception as e:
            print(f"Warning: Could not load system prompt from {self.PROVIDER.system_prompt_file}: {e}")
            self.system_prompt = _FALLBACK_SYSTEM_PROMPT

    async def initialize(self):
        """Initialize both the MCP client and the LLM client."""
        await super().initialize()
        self._build_llm_client()

    async def close(self):
        """Clean up resources.

        The MCP client is closed automatically when its context manager exits,
        and the LLM clients need no explicit cleanup.
        """
        pass

    def _build_prompt(self, prompt: str) -> str:
        """Compose the full prompt: system prompt + tool list + user request."""
        return f"""{self.system_prompt}

---

User Request: {prompt}

{_TOOL_LIST}

Instructions: Follow your data scientist workflow by searching for relevant \
datasets and analyzing their metadata based on the user's request. If you need \
to use tools, also execute them.""".strip()

    async def ask_llm_with_tools(self, prompt: str, temperature: float | None = None):
        """Ask the LLM to answer prompt with access to the MCP tools."""
        if not self.llm_client:
            raise ValueError(f"{self.PROVIDER.config_key.title()} client not available (no API key)")

        temp = temperature if temperature is not None else self.default_temperature
        async with self.mcp_client:
            return await self._generate(prompt, temp)

    # --- Provider-specific hooks -------------------------------------------

    def _build_llm_client(self):
        """Construct the provider SDK client into self.llm_client.

        Leaves self.llm_client as None when no API key is set.
        """
        raise NotImplementedError

    async def _generate(self, user_prompt: str, temperature: float) -> str:
        """Generate a response for user_prompt using the provider SDK.

        Builds the full prompt using _build_prompt(user_prompt).
        """
        raise NotImplementedError
