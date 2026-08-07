"""The committed front-end build, and how the server serves it.

These replace the old "run the inline script under a node DOM stub" tests with
checks on what actually ships: a committed Vite build under ``frontend/dist``,
a manifest that faithfully describes it, and a server that serves exactly those
files statically and nothing else. No Node is required to run them -- the build
is committed, which is the whole point (the Windows user never builds).
"""

from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path

import pytest

from src import review_assets, review_server
from scripts.verification.album_viewer_fixture import build

REPO = Path(__file__).resolve().parents[1]
DIST = REPO / "frontend" / "dist"


def test_committed_build_is_present():
    """The runtime ships the build; a missing dist is a release error."""
    assert review_assets.build_exists(), (
        f"committed front-end build missing at {DIST}; run `npm ci && npm run build` "
        f"in frontend/ and commit dist/"
    )
    assert (DIST / "review.html").is_file()
    assert list((DIST / "assets").glob("*.js")), "no JS chunks in the committed build"


def test_manifest_matches_what_is_on_disk():
    """review_assets.json names every emitted asset, and only those."""
    manifest = review_assets.manifest_for()
    on_disk = {
        p.relative_to(DIST).as_posix()
        for p in (DIST / "assets").rglob("*")
        if p.is_file()
    }
    assert set(manifest["assets"]) == on_disk
    assert manifest["entry"] == "review.html"


def test_review_html_only_references_manifest_assets():
    """The entry page loads exactly the hashed chunks the allow-list will serve."""
    import re

    html = (DIST / "review.html").read_text(encoding="utf-8")
    # The build dir itself carries no installed manifest; derive it directly.
    allowed = set(review_assets.manifest_for(DIST)["assets"])
    refs = re.findall(r'\./(assets/[^"\']+)', html)
    assert refs, "expected the built page to reference at least one asset"
    for ref in refs:
        assert ref in allowed, f"{ref} referenced but not allow-listed"


def test_install_is_idempotent_and_self_describing(tmp_path):
    first = review_assets.install_into(tmp_path)
    second = review_assets.install_into(tmp_path)
    assert first == second
    written = json.loads((tmp_path / review_assets.MANIFEST_NAME).read_text("utf-8"))
    assert written == second


# --- the served static boundary, end to end -------------------------------

def _serve(output: Path):
    server, url = review_server.start_server(output, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, server.server_port


def _get(port: int, path: str) -> int:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", path)
        return conn.getresponse().status
    finally:
        conn.close()


@pytest.fixture()
def served(tmp_path):
    output = tmp_path / "output"
    build(output, tmp_path / "photos")  # installs the committed UI, like Stage 3
    server, thread, port = _serve(output)
    try:
        yield output, port
    finally:
        server.shutdown()
        server.server_close()
        server.review_data.close()
        thread.join(timeout=5)


def test_server_serves_the_entry_and_its_hashed_assets(served):
    output, port = served
    assert _get(port, "/review.html") == 200
    for asset in review_assets.load_manifest(output)["assets"]:
        assert _get(port, "/" + asset) == 200, asset


def test_server_denies_working_files_and_unlisted_assets(served):
    output, port = served
    # Working files in the output directory are never web-served.
    for name in ("review_assets.json", "review_summary.json", "inventory.sqlite",
                 "delete_local.txt", "summary.txt"):
        assert _get(port, "/" + name) == 404, name
    # A plausible-looking but unlisted asset name is refused (allow-list, not dir).
    assert _get(port, "/assets/review-DEADBEEF.js") == 404
    # Traversal and encoded traversal cannot escape to another file.
    assert _get(port, "/assets/../inventory.sqlite") == 404
    assert _get(port, "/%2e%2e/inventory.sqlite") == 404
