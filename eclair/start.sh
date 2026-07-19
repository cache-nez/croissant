#!/bin/bash

# Eclair dataset server quickstart script for development.
#
# Sets up the environment (see install.sh), then launches the server.
set -e

# Set up uv and the project environment.
"$(dirname "$0")/install.sh"

# Launch the server in that environment.
# Note: while the server is bound to 0.0.0.0, clients should connect to 127.0.0.1 or localhost
# because MCP SDK's transport-security layer does not allow 0.0.0.0
uv run eclair-server --host 0.0.0.0 --port 8080 --transport streamable-http
