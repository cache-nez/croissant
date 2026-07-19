"""
Local LLM Client Package

Integration for locally running models exposed through an OpenAI-compatible
API (Ollama, vLLM, LM Studio, llama.cpp, LiteLLM, ...).
"""

try:
    from .client import LocalMCPClient, LOCAL_AVAILABLE
except ImportError:
    LocalMCPClient = None
    LOCAL_AVAILABLE = False

__all__ = ['LocalMCPClient', 'LOCAL_AVAILABLE']
