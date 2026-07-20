## Setup

To run this project, you need 4 components:

1. Local MCP server:

```
cd croissant/eclair
./start.sh
```

2. Remote MCP server: the local MCP server relays most of the calls to it, but the original upstream server on jetty no longer works. Start a stub server (see details about its functionality below):

```
cd croissant/eclair
./start_stub.sh
```

3. Run an OpenAI-compatible LLM locally on port 11434 (update config.json with the model name and an alternative port, if using). I used qwen3-1.7b on llama.cpp:

  1. llama.cpp installation (optional, replace by provider of your choice):

  ```
  brew install llama.cpp
  ```

  or

  ```
  conda install -c conda-forge llama.cpp
  ```

  2. Run llama.cpp or replace with a provider and model of your choice:

  ```
  llama-server -hf unsloth/Qwen3-1.7B-GGUF:Q4_K_M --port 11434
  ```

4. Run the client that bridges the MCP server and the local model! You have two options:

  1. Run a query with `eclair-client` (source: `cli.py`):

  ```
  uv run eclair-client --use-local --tool ask --query "find datasets that include age"
  ```

  2. Debug mode: run `local/client.py` to see the debug logs, including the exchange with the LLM and tools called:

  ```
  uv run python -m eclair.client.local.client
  ```

  If you prefer Gemini, run `uv run python -m eclair.client.gemini.client` or `uv run eclair-client --use-gemini --tool ask --query "find datasets that include age"`. You need to provide `GEMINI_API_KEY` either in an `.env` file or prepend it to the command (`GEMINI_API_KEY=mykey uv run ...`). Analogous for Claude.


## Scope of this project

The intended scope of this project was to studly existing client implementations for Claude and Gemini, identify and document the contracts they use, and implement a client for a locally running LLM. However, once I started the project, I realized that the scope is significantly larger due to key pieces missing from the current implementation:

* Existing clients did not implement the tool call loop: they were not giving the model a way to call the MCP server. They were simply naming the MCP tools in the prompt and returning the LLM's reponse.

* The local MCP server, implemented in this repository, calls an upstream server at `mcp.jetty.io/mcp` for many operations (for example, search-datasets). This upstream server is no longer active.

* The installation instructions are fragile: I couldn't install the project as instructed due to Appple's Python version older than 3.10.

* Docs specify commands that fail: connecting to 0.0.0.0 instead of localhost, `curl` with wrong request format.

* The Gemini client uses deprecated google.generativeai.

Therefore, I had to address these issues in addition to completing the project as described. My contributions include:

* [Robust installation with `uv`](https://github.com/cache-nez/croissant/pull/1): `uv` manages the Python version and installs packages to the `venv`, leaving global Python state untouched.

* [A stub for the upstream MCP server](https://github.com/cache-nez/croissant/pull/2): the stub manages 3 datasets taken from croissant/datasets: simple-join, simple-parquet, and titanic. For example, it answers search-datasets query according to a hardcoded set of keywords for each dataset.

* [Refactoring of existing clients](https://github.com/cache-nez/croissant/pull/3) to follow a universal interface (this is in the original scope of the project),

* [Implementation of a tool-call loop for Gemini and Claude](https://github.com/cache-nez/croissant/pull/4),

* [Bug fixing](https://github.com/cache-nez/croissant/pull/5) of Gemini interactions with the MCP server,

* [Implementation of the tool-call loop for a locally running LLM](https://github.com/cache-nez/croissant/pull/6) (this is in the original scope of the project).

## Design decision

1. Refactor the existing clients before adding support for a local LLM

Because the tool-call loop was not implemented and had to be added to all clients, I chose to first create a uniform interface for the clients, then implement it instead of repeating the logic in each client.

2. Drop the fallback logic from the Gemini client

On exception during LLM call, the Gemini client was falling back to performing a dataset search and retrying the LLM call with the search results. I disagree with this approach: the user's query might not concern dataset search at all, and the exception is likely to be due to misconfiguration or LLM provider being unavailable; I don't see how the exception could be fixed by a manual MCP call. Therefore, I did not include this logic in the revised clients.

3. Implement a generic tool-call loop in the base LLM class and add provider-specific logic as functions specialized in the derived classes.

The tool-call logic is the same for all LLMs: send the query to the LLM, check if it requested to use a tool, add the LLM response and the result of tool calls to the history since the LLM itself is stateless, repeat. The specifics are provider-dependent: how to express the signature of a tool to the LLM, how the LLM requests to use a tool, how the LLM call is performed.

4. Communicate with the locally-running LLM via OpenAI-compatible endpoint.

All commonly-used providers for local LLM inference expose OpenAI-compatible endpoints. Sending the queries to this endpoint allows my client to support all such providers rather than specializing for one.

### Recommended extensions

Here's what I would do if I had more time:

* Detailed error reporting in the client: MCP server unavailable, LLM provider not responding, ...

* Automated tests for the end-to-end workflow (tested manually with the commands provided in the setup section)

* Graceful termination of the MCP server.
