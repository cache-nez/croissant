#!/bin/bash

# Eclair environment setup for development.
#
# Uses uv to provision a compatible Python (>=3.10) and create a project
# environment automatically, so setup works regardless of the system Python
# version.
set -e

# Ensure uv is available; it manages the Python version and the project venv.
if ! command -v uv &> /dev/null; then
    if [ ! -t 0 ]; then
        echo "uv is not installed and this shell is non-interactive." >&2
        echo "Install uv (https://docs.astral.sh/uv/) and re-run ./install.sh." >&2
        exit 1
    fi

    printf 'uv is required but not installed. Install it now? [y/N] '
    read -r reply
    case "$reply" in
        [Yy] | yes | Yes | YES) ;;
        *)
            echo "Aborting. Install uv from https://docs.astral.sh/uv/ and re-run ./install.sh."
            exit 1
            ;;
    esac

    curl -LsSf https://astral.sh/uv/install.sh | sh

    if ! command -v uv &> /dev/null; then
        echo "uv installation did not complete. See https://docs.astral.sh/uv/." >&2
        exit 1
    fi
fi

# Set up the environment:
# * provision a compatible Python if needed,
# * create an isolated venv,
# * install Eclair and its dependencies (from pyproject.toml).
uv sync
