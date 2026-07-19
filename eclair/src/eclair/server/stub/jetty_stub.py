"""
Jetty upstream stub server.

A local, self-contained MCP server that impersonates the upstream Jetty server
(https://mcp.jetty.io/mcp) that Eclair relays to. It returns deterministic,
hardcoded answers built from three real Croissant datasets shipped in this repo
(datasets/1.0/{titanic,simple-join,simple-parquet}).

Eclair's relay (src/eclair/server/tools.py, class MCPRelay) forwards exactly four
tool calls to the upstream server. This stub exposes those four tools:

    - search-datasets      {query}               -> list of match objects
    - download-dataset     {collection, dataset} -> dict
    - datasets-preview-url {collection, dataset} -> string URL
    - dataset/mlcroissant  {collection, dataset} -> Croissant JSON-LD dict
      (this is the upstream name behind Eclair's `serve-croissant`)

Point Eclair at this stub by setting `upstream_server.url` in eclair/config.json
to http://127.0.0.1:9090/mcp, then start this server.

Run (from the eclair/ directory):
    uv run python src/eclair/server/stub/jetty_stub.py
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jetty_stub")

# --- Server binding -------------------------------------------------------
# Bind to 127.0.0.1 (not 0.0.0.0): the MCP SDK transport-security layer rejects
# 0.0.0.0 clients. Port 9090 avoids Eclair's own 8080.
HOST = "127.0.0.1"
PORT = 9090


def _find_datasets_dir() -> str:
    """Walk upward from this file until a `datasets/1.0` directory is found.

    Kept location-independent so the stub keeps working if moved within the repo.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    while True:
        candidate = os.path.join(d, "datasets", "1.0")
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:
            raise FileNotFoundError(f"Could not locate datasets/1.0 above {here}")
        d = parent


# --- Dataset registry -----------------------------------------------------
# The three datasets we hardcode. `folder` is the directory under datasets/1.0/,
# `collection`/`dataset` are the (collection, dataset) keys Eclair passes through.
_DATASETS_DIR = _find_datasets_dir()

_DATASET_SPECS = [
    {"collection": "mlcommons", "dataset": "titanic", "folder": "titanic"},
    {"collection": "mlcommons", "dataset": "simple-join", "folder": "simple-join"},
    {"collection": "mlcommons", "dataset": "simple-parquet", "folder": "simple-parquet"},
]

# search-datasets keyword map (approved). Content-derived keywords -> dataset id.
# A dataset matches if any of its keywords appears in the (lowercased) query.
KEYWORD_MAP: Dict[str, List[str]] = {
    "titanic": ["titanic", "ship", "passenger", "passengers", "embark", "port", "age"],
    "simple-join": [
        "publication", "publications", "author", "authors",
        "machine learning", "neural network", "natural language processing",
        "nlp", "email",
    ],
    "simple-parquet": ["parquet", "person", "persons", "age", "name"],
}


def _load_metadata(folder: str) -> Dict[str, Any]:
    path = os.path.join(_DATASETS_DIR, folder, "metadata.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_registry() -> "Dict[tuple, Dict[str, Any]]":
    """Load metadata.json for each dataset once, keyed by (collection, dataset) lowercased."""
    registry: Dict[tuple, Dict[str, Any]] = {}
    for spec in _DATASET_SPECS:
        metadata = _load_metadata(spec["folder"])
        entry = {
            "collection": spec["collection"],
            "dataset": spec["dataset"],
            "metadata": metadata,
            "data_path": f"{spec['collection']}/{spec['dataset']}",
        }
        registry[(spec["collection"].lower(), spec["dataset"].lower())] = entry
        logger.info("Loaded dataset %s/%s from %s/metadata.json", spec["collection"], spec["dataset"], spec["folder"])
    return registry


REGISTRY = _build_registry()


def _lookup(collection: str, dataset: str) -> Optional[Dict[str, Any]]:
    return REGISTRY.get((str(collection).lower(), str(dataset).lower()))


def _first_record_set_id(metadata: Dict[str, Any]) -> str:
    record_sets = metadata.get("recordSet") or []
    if record_sets and isinstance(record_sets, list):
        return record_sets[0].get("@id", "")
    return ""


def _metadata_summary(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Small subset of the Croissant metadata, matching the upstream search sample shape."""
    return {
        "__type": metadata.get("@type", "sc:Dataset"),
        "conformsTo": metadata.get("conformsTo"),
        "name": metadata.get("name"),
        "description": metadata.get("description"),
        "url": metadata.get("url"),
        "license": metadata.get("license"),
    }


mcp = FastMCP("Jetty Stub Server")


@mcp.tool("search-datasets", description="Search for datasets using a query string")
async def search_datasets(query: str) -> List[Dict[str, Any]]:
    """Return match objects for datasets whose keywords appear in the query."""
    q = (query or "").lower()
    results: List[Dict[str, Any]] = []
    # Iterate in registry (spec) order so multi-matches are deterministic.
    for spec in _DATASET_SPECS:
        dataset_id = spec["dataset"]
        keywords = KEYWORD_MAP.get(dataset_id, [])
        matched = [kw for kw in keywords if kw in q]
        if not matched:
            continue
        entry = REGISTRY[(spec["collection"].lower(), dataset_id.lower())]
        metadata = entry["metadata"]
        description = metadata.get("description") or ""
        results.append({
            "document": {
                "collection_name": spec["collection"],
                "entity_name": dataset_id,
                "entity_type": "dataset",
                "full_name": f"dataset/{spec['collection']}/{dataset_id}",
                "metadata": _metadata_summary(metadata),
            },
            "highlight": {
                "metadata": {
                    "description": {
                        "matched_tokens": matched,
                        "snippet": description[:160],
                    }
                }
            },
            "text_match": 1157451471441100800,
            "text_match_info": {
                "best_field_score": "2211897868288",
                "best_field_weight": 15,
                "fields_matched": len(matched),
                "score": "1157451471441100922",
                "tokens_matched": len(matched),
            },
        })
    return results


@mcp.tool("download-dataset", description="Download a dataset")
async def download_dataset(collection: str, dataset: str) -> Dict[str, Any]:
    """Return metadata + local data path + a short usage snippet."""
    entry = _lookup(collection, dataset)
    if entry is None:
        return {"error": f"Dataset not found: {collection}/{dataset}"}
    metadata = entry["metadata"]
    record_set = _first_record_set_id(metadata)
    instructions = (
        "# pip install mlcroissant\n"
        "import mlcroissant as mlc\n\n"
        f"# Croissant metadata for {entry['data_path']}\n"
        f"ds = mlc.Dataset(jsonld=\"datasets/1.0/{dataset}/metadata.json\")\n"
        f"records = ds.records(record_set=\"{record_set}\")\n"
        "for i, record in enumerate(records):\n"
        "    print(record)\n"
        "    if i >= 4:\n"
        "        break\n"
    )
    return {
        "metadata": metadata,
        "asset_origin": "mlcommons",
        "data_path": entry["data_path"],
        "instructions": instructions,
    }


@mcp.tool("datasets-preview-url", description="Get a download url for a dataset preview")
async def datasets_preview_url(collection: str, dataset: str) -> str:
    """Return a (local) preview URL string for the dataset."""
    entry = _lookup(collection, dataset)
    if entry is None:
        return f"Error: Dataset not found: {collection}/{dataset}"
    return f"http://{HOST}:{PORT}/api/v1/datasets/{entry['data_path']}/preview"


@mcp.tool("dataset/mlcroissant", description="Get the Croissant dataset metadata")
async def dataset_mlcroissant(collection: str, dataset: str) -> Dict[str, Any]:
    """Return the full Croissant JSON-LD metadata (upstream name behind serve-croissant)."""
    entry = _lookup(collection, dataset)
    if entry is None:
        return {"error": f"Dataset not found: {collection}/{dataset}"}
    return entry["metadata"]


def main() -> None:
    mcp.settings.host = HOST
    mcp.settings.port = PORT
    url = f"http://{HOST}:{PORT}/mcp"
    print("🧩 Jetty upstream stub")
    print(f"☁️ Serving at:   {url}")
    print(f"📚 Datasets:     {', '.join(spec['collection'] + '/' + spec['dataset'] for spec in _DATASET_SPECS)}")
    print("   Point eclair/config.json upstream_server.url at the URL above.")
    print("")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
