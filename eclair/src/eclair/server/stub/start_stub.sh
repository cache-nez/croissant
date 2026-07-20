#!/bin/bash

# Launch the local Jetty upstream stub for Eclair development.
#
# Run from anywhere:
#     ./src/eclair/server/stub/start_stub.sh
#
# The stub serves an MCP streamable-http endpoint at http://127.0.0.1:9090/mcp.
# Point eclair/config.json's upstream_server.url at that URL, then start Eclair.
set -e

# cd to the eclair/ project root (four levels up from this script).
cd "$(dirname "$0")/../../../.."

# Ensure the uv environment is set up (see install.sh).
./install.sh

uv run python src/eclair/server/stub/jetty_stub.py
