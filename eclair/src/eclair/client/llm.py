"""
LLM MCP Client

Shared base class for clients that pair an Eclair MCP server with an LLM.
"""

import inspect
import json
import os
from dataclasses import dataclass
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

    def __init__(self, config_key: str, env_var: str, system_prompt_file: str) -> None:
        self.config_key = config_key
        self.env_var = env_var
        self.system_prompt_file = system_prompt_file


@dataclass
class ToolCall:
    """A single tool invocation requested by the LLM, normalized across providers.

    id is the provider's correlation handle (required by Claude to match a
    result back to its request; unused by Gemini, which correlates by name).
    """

    id: str
    name: str
    arguments: dict


_FALLBACK_SYSTEM_PROMPT = (
    "You are a helpful data scientist AI assistant. You have access to MCP tools "
    "for finding and analyzing datasets. Always try to use the available tools to "
    "search for and analyze real data to answer user questions."
)


class LlmMcpClient(EclairClient):
    """Base client that connects an LLM to the Eclair MCP Server.

    Adds LLM scaffolding (config + system-prompt loading, API-key resolution,
    the tool-augmented ask_llm_with_tools loop) on top of the plain MCP
    transport provided by EclairClient. All MCP tool wrappers are inherited
    unchanged.

    Concrete subclasses implement the provider-specific hooks at the bottom of
    this class; the tool-call loop itself lives here and is shared.
    """

    # Set by each concrete subclass, e.g. PROVIDER = Provider.GEMINI.
    PROVIDER: Provider = None

    def __init__(self, mcp_server_url: str = "http://localhost:8080/mcp") -> None:
        super().__init__(mcp_server_url)
        # Key is resolved from the environment only (never passed in code).
        self.api_key = os.getenv(self.PROVIDER.env_var)
        self.llm_client = None
        # Provider-format tool declarations, fetched and converted once, lazily.
        self._llm_tools = None

        self._load_config()
        self._load_system_prompt()

    def _load_config(self) -> None:
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

    def _load_system_prompt(self) -> None:
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

    async def initialize(self) -> None:
        """Initialize both the MCP client and the LLM client."""
        await super().initialize()
        self._build_llm_client()

    async def close(self) -> None:
        """Clean up resources.

        The MCP client is closed automatically when its context manager exits,
        and the LLM clients need no explicit cleanup.
        """
        pass

    async def _get_tools(self) -> list:
        """Return the provider-format tool declarations, fetching once and caching.

        list_tools() needs an open MCP session, so this must be called from
        within an active self.mcp_client context.
        """
        if self._llm_tools is None:
            mcp_tools = await self.mcp_client.list_tools()
            self._llm_tools = self._mcp_tools_to_provider_format(mcp_tools)
        return self._llm_tools

    def _result_payload(self, result) -> dict:
        """Turn a fastmcp CallToolResult into a JSON-serializable dict to feed back."""
        if getattr(result, "structured_content", None) is not None:
            return result.structured_content
        text = "\n".join(getattr(c, "text", str(c)) for c in (getattr(result, "content", None) or []))
        return {"result": text}

    async def ask_llm_with_tools(self, prompt: str, temperature: float | None = None) -> str:
        """Ask the LLM to answer prompt, letting it call the MCP tools in a loop.

        The MCP connection is held open for the whole exchange: each turn the LLM
        may request tool calls, which are executed and fed back until it returns
        a plain-text answer.
        """
        if not self.llm_client:
            raise ValueError(f"{self.PROVIDER.config_key.title()} client not available (no API key)")

        temp = temperature if temperature is not None else self.default_temperature
        async with self.mcp_client:
            tools = await self._get_tools()
            history = self._initial_history(prompt)
            while True:
                llm_response = self._generate_turn(history, tools, temp)
                tool_calls = self._parse_tool_calls(llm_response)
                if not tool_calls:
                    return self._parse_final_text(llm_response)

                tool_results = []
                for call in tool_calls:
                    result = await self.mcp_client.call_tool(call.name, call.arguments)
                    tool_results.append((call, result))
                self._append_turn(history, llm_response, tool_results)

    # --- Provider-specific hooks -------------------------------------------

    def _build_llm_client(self) -> None:
        """Construct the provider SDK client into self.llm_client.

        Leaves self.llm_client as None when no API key is set.
        """
        raise NotImplementedError

    def _mcp_tools_to_provider_format(self, mcp_tools) -> list:
        """Convert MCP tool definitions into this provider's tool-declaration format."""
        raise NotImplementedError

    def _initial_history(self, prompt: str) -> list:
        """Return the provider's initial message history for a user prompt."""
        raise NotImplementedError

    def _generate_turn(self, history, tools, temperature: float):
        """Call the LLM for one turn and return the raw provider response."""
        raise NotImplementedError

    def _parse_tool_calls(self, llm_response) -> list[ToolCall]:
        """Return the tool calls the LLM requested this turn (empty when none)."""
        raise NotImplementedError

    def _parse_final_text(self, llm_response) -> str:
        """Extract the final text answer from a response with no tool calls."""
        raise NotImplementedError

    def _append_turn(self, history, llm_response, tool_call_results) -> None:
        """Append the assistant's tool-call turn and the tool results to history.

        tool_call_results is a list of (ToolCall, CallToolResult) pairs.
        """
        raise NotImplementedError
