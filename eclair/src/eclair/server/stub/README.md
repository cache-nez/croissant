# Jetty upstream stub

A local, self-contained MCP server that impersonates the upstream **Jetty** server
(`https://mcp.jetty.io/mcp`) that Eclair relays to. It lets you run and test Eclair
fully offline with deterministic, hardcoded answers.

Eclair's relay (`src/eclair/server/tools.py`, `MCPRelay`) forwards exactly four tool
calls to the upstream server. This stub exposes those four:

| Upstream tool         | Args                    | Returns                                   | Eclair tool that uses it |
| --------------------- | ----------------------- | ----------------------------------------- | ------------------------ |
| `search-datasets`     | `{query}`               | list of match objects                     | `search-datasets`        |
| `download-dataset`    | `{collection, dataset}` | dict: metadata, asset_origin, data_path, instructions | `download-dataset` |
| `datasets-preview-url`| `{collection, dataset}` | string URL                                | `datasets-preview-url`   |
| `dataset/mlcroissant` | `{collection, dataset}` | Croissant JSON-LD dict                    | `serve-croissant`        |

Eclair's other tools (`validate-croissant`, `help`, `ping`, `builder-context`,
`pytorch-scaffold`) run locally in Eclair and do not touch the upstream server, so the
stub does not implement them.

## Hardcoded datasets

Answers are built from three real Croissant datasets in this repo, all under
`datasets/1.0/`:

| Collection / dataset       | Source folder                     |
| -------------------------- | --------------------------------- |
| `mlcommons/titanic`        | `datasets/1.0/titanic`            |
| `mlcommons/simple-join`    | `datasets/1.0/simple-join`        |
| `mlcommons/simple-parquet` | `datasets/1.0/simple-parquet`     |

`search-datasets` matches on a hardcoded, content-derived keyword map (see
`KEYWORD_MAP` in `jetty_stub.py`), e.g. a query containing `passenger` returns titanic,
`publications` returns simple-join, `parquet` returns simple-parquet. `age` returns both
titanic and simple-parquet.

## Run

From the `eclair/` directory:

```bash
./src/eclair/server/stub/start_stub.sh     # sets up the uv env, then serves
# or directly:
uv run python src/eclair/server/stub/jetty_stub.py
```

The stub serves `http://127.0.0.1:9090/mcp`. Override the bind address with
`JETTY_STUB_HOST` / `JETTY_STUB_PORT` if those defaults are taken.

## Wire Eclair to the stub

`eclair/config.json` is set to point at the stub by default:

```json
"upstream_server": {
  "url": "http://127.0.0.1:9090/mcp",
  "timeout": 5000
}
```

Start the stub (above), then start Eclair (`./start.sh`). Eclair's relayed tools will
return the hardcoded answers with no network call to `mcp.jetty.io`.

Alternatively, leave `config.json` alone and point the relay at the stub for a single
run with the `ECLAIR_UPSTREAM_URL` env var, which overrides `upstream_server.url`:

```bash
ECLAIR_UPSTREAM_URL=http://127.0.0.1:9090/mcp ./start.sh
```

### Revert to the real Jetty server

Change `upstream_server.url` in `eclair/config.json` back to:

```json
"url": "https://mcp.jetty.io/mcp"
```
