"""
End-to-end tests for the local Jetty upstream stub, driven through EclairClient.

These tests exercise the full relay path:

    EclairClient  ->  Eclair server (:8080)  ->  Jetty stub (:9090)

The stub (src/eclair/server/stub/jetty_stub.py) impersonates the upstream Jetty
server and returns hardcoded answers built from three in-repo Croissant datasets
(mlcommons/{titanic, simple-join, simple-parquet}). config.json's
upstream_server.url points Eclair's relay at the stub, so calling Eclair's
relayed tools returns the stub's answers.

The `servers` fixture launches the stub and the Eclair server as subprocesses for
the whole session and tears them down at the end.
"""
import asyncio
import json
import os
import socket
import subprocess
import sys
import time

import pytest

from eclair.client import EclairClient

_HERE = os.path.dirname(__file__)
_ECLAIR_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_STUB_PATH = os.path.join(_ECLAIR_ROOT, "src", "eclair", "server", "stub", "jetty_stub.py")

# Eclair server (client target) and stub (relay target) bindings. Deliberately
# not the default 8080/9090 so the tests never collide with a locally running
# Eclair or stub; the tests start their own servers on these dedicated ports.
ECLAIR_HOST, ECLAIR_PORT = "127.0.0.1", 8181
STUB_HOST, STUB_PORT = "127.0.0.1", 9191
ECLAIR_URL = f"http://{ECLAIR_HOST}:{ECLAIR_PORT}/mcp"
STUB_URL = f"http://{STUB_HOST}:{STUB_PORT}/mcp"


def _port_is_free(host: str, port: int) -> bool:
    """True if nothing is currently accepting connections on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) != 0


def _wait_for_port(host: str, port: int, timeout: float = 20.0) -> bool:
    """Block until a TCP connection to host:port succeeds or timeout elapses."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.2)
    return False


def _wait_for_relay(url: str, timeout: float = 30.0) -> bool:
    """Retry a *relayed* tool call until Eclair returns a real stub answer.

    A listening port (or a local `ping`) does not mean Eclair's relay to the stub
    is warm yet; probing search-datasets until a genuine match comes back gates the
    tests on the full EclairClient -> Eclair -> stub path being ready.
    """
    async def _probe() -> None:
        result = await EclairClient(url).search_datasets("titanic ship")
        text = "".join(c.text for c in result.content if getattr(c, "text", None))
        if "document" not in text:
            raise RuntimeError(f"relay not warm yet: {text[:120]}")

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            asyncio.run(_probe())
            return True
        except Exception:
            time.sleep(0.3)
    return False


def _terminate(proc: subprocess.Popen) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def servers():
    """Start the Jetty stub and the Eclair server (relaying to it) for the session.

    The tests own their servers: they assume nothing is already listening on the
    dedicated ports and start fresh subprocesses, wiring the stub port and the
    relay target through env vars (no reliance on config.json). The stub port and
    Eclair's upstream URL are overridden so no committed config is touched.
    """
    for host, port in ((STUB_HOST, STUB_PORT), (ECLAIR_HOST, ECLAIR_PORT)):
        if not _port_is_free(host, port):
            pytest.fail(
                f"Port {host}:{port} is already in use; these tests expect neither the "
                f"Eclair server nor the stub to be running. Stop whatever is on it and retry."
            )

    stub_env = {**os.environ, "JETTY_STUB_HOST": STUB_HOST, "JETTY_STUB_PORT": str(STUB_PORT)}
    stub_proc = subprocess.Popen(
        [sys.executable, _STUB_PATH],
        cwd=_ECLAIR_ROOT,
        env=stub_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    eclair_proc = None
    try:
        if not _wait_for_port(STUB_HOST, STUB_PORT):
            out = stub_proc.communicate(timeout=5)[0] if stub_proc.poll() is not None else ""
            pytest.fail(f"Jetty stub failed to start on {STUB_HOST}:{STUB_PORT}\n{out}")

        # Point Eclair's relay at our stub via env so config.json stays untouched.
        eclair_env = {**os.environ, "ECLAIR_UPSTREAM_URL": STUB_URL}
        eclair_proc = subprocess.Popen(
            ["eclair-server", "--host", ECLAIR_HOST, "--port", str(ECLAIR_PORT),
             "--transport", "streamable-http"],
            cwd=_ECLAIR_ROOT,
            env=eclair_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if not _wait_for_port(ECLAIR_HOST, ECLAIR_PORT) or not _wait_for_relay(ECLAIR_URL):
            out = eclair_proc.communicate(timeout=5)[0] if eclair_proc.poll() is not None else ""
            pytest.fail(f"Eclair server failed to become ready on {ECLAIR_HOST}:{ECLAIR_PORT}\n{out}")
        yield
    finally:
        _terminate(eclair_proc)
        _terminate(stub_proc)


def _value(result):
    """Extract the plain Python value from an EclairClient (fastmcp) CallToolResult.

    MCP wraps a non-object tool return (str, list) as ``{"result": <value>}`` in
    structuredContent, while an object (dict) return is passed through as-is.
    """
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        if list(structured.keys()) == ["result"]:
            return structured["result"]
        return structured
    # Fallback: parse the text content as JSON, else return the raw text.
    texts = [c.text for c in getattr(result, "content", []) if getattr(c, "text", None)]
    joined = "".join(texts)
    try:
        return json.loads(joined)
    except (json.JSONDecodeError, ValueError):
        return joined


def _entity_names(search_result):
    """Dataset names from a search-datasets result (a list of match objects).

    Only reads genuine match objects (those with a `document`), so an empty
    search — which Eclair's relay surfaces as a degenerate `[{"result": []}]`
    wrapper — yields no names.
    """
    value = _value(search_result)
    if isinstance(value, dict):  # single match may arrive unwrapped
        value = [value]
    return [
        item["document"]["entity_name"]
        for item in value
        if isinstance(item, dict) and "document" in item
    ]


@pytest.fixture
def client():
    return EclairClient(ECLAIR_URL)


async def test_ping(servers, client):
    """Sanity: the Eclair stack is up (ping is a local Eclair tool)."""
    result = await client.ping()
    assert "Pong!" in _value(result)


async def test_search_datasets_titanic(servers, client):
    """A titanic-flavored query returns the titanic dataset."""
    result = await client.search_datasets("passengers on the titanic ship")
    names = _entity_names(result)
    assert "titanic" in names
    assert "simple-parquet" not in names


async def test_search_datasets_shared_keyword_matches_multiple(servers, client):
    """'age' is a keyword for both titanic and simple-parquet, so both return."""
    names = _entity_names(await client.search_datasets("record the age of each subject"))
    assert set(names) == {"titanic", "simple-parquet"}


async def test_search_datasets_simple_join(servers, client):
    """Content keywords (publications/nlp) map to simple-join."""
    names = _entity_names(await client.search_datasets("publications by an author about nlp"))
    assert names == ["simple-join"]


async def test_search_datasets_no_match(servers, client):
    """An unrelated query returns no datasets."""
    assert _entity_names(await client.search_datasets("quarterly revenue forecast")) == []


async def test_serve_croissant_titanic(servers, client):
    """serve-croissant relays to the stub's dataset/mlcroissant and returns full JSON-LD."""
    metadata = _value(await client.serve_croissant("mlcommons", "titanic"))
    assert metadata["name"] == "Titanic"
    assert metadata["conformsTo"] == "http://mlcommons.org/croissant/1.0"
    assert len(metadata["recordSet"]) == 3


async def test_download_dataset(servers, client):
    """download-dataset returns metadata + local data_path + usage instructions."""
    info = _value(await client.download_dataset("mlcommons", "simple-join"))
    assert set(info.keys()) == {"metadata", "asset_origin", "data_path", "instructions"}
    assert info["asset_origin"] == "mlcommons"
    assert info["data_path"] == "mlcommons/simple-join"
    assert info["metadata"]["name"] == "simple-join"
    assert "mlcroissant" in info["instructions"]


async def test_datasets_preview_url(servers, client):
    """datasets-preview-url returns a plain URL string for the dataset."""
    url = _value(await client.datasets_preview_url("mlcommons", "simple-parquet"))
    assert isinstance(url, str)
    assert url.endswith("/api/v1/datasets/mlcommons/simple-parquet/preview")


async def test_serve_croissant_unknown_dataset(servers, client):
    """An unknown dataset surfaces the stub's error object through the relay."""
    result = _value(await client.serve_croissant("mlcommons", "does-not-exist"))
    assert isinstance(result, dict)
    assert "error" in result
