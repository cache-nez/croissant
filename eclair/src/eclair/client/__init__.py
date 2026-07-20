"""
Eclair Client Package

Contains client utilities for interacting with Eclair servers.
"""

try:
    from .client import EclairClient
except ImportError:
    EclairClient = None

# Optional integrations: never let one break importing the core client. Catch
# broadly (not just ImportError) so that, e.g., a deprecated dependency emitting a
# warning that has been promoted to an error doesn't take the whole package down.
try:
    from .gemini import GeminiMCPClient
except Exception:
    GeminiMCPClient = None

try:
    from .claude import ClaudeMCPClient
except Exception:
    ClaudeMCPClient = None

try:
    from .local import LocalMCPClient
except ImportError:
    LocalMCPClient = None

__all__ = ["EclairClient", "GeminiMCPClient", "ClaudeMCPClient", "LocalMCPClient"]
