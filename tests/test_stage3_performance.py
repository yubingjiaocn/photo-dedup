"""Stage 3 installs the review UI without ever opening an original photo.

The review page used to be a Python-rendered HTML string with inlined script and
static file:// fallback tiles. It is now a committed Preact build that Stage 3
*copies* into the output directory (``review.html`` + hashed ``assets/*`` +
``review_assets.json``). These tests assert the surviving Stage-3 boundary --
no image is read, the CLI plumbing is intact -- against the new install path.
"""

from PIL import Image

from src import review_assets, review_page, stage3_report


def test_installing_the_review_ui_never_opens_a_source_image(tmp_path, monkeypatch):
    """Copying the built UI touches static files only, never an original photo."""
    def explode(*args, **kwargs):
        raise AssertionError("stage 3 must not open original photos")

    monkeypatch.setattr(Image, "open", explode)
    manifest = review_page.install_review_ui(tmp_path)

    review = tmp_path / "review.html"
    assert review.is_file()
    assert (tmp_path / "assets").is_dir()
    assert (tmp_path / review_assets.MANIFEST_NAME).is_file()
    assert manifest["entry"] == "review.html"
    assert manifest["assets"], "expected at least one hashed asset chunk"


def test_installed_page_loads_only_its_own_hashed_assets(tmp_path):
    """review.html references exactly the assets the manifest allow-lists."""
    review_page.install_review_ui(tmp_path)
    html = (tmp_path / "review.html").read_text(encoding="utf-8")
    allowed = review_assets.allowed_static_paths(tmp_path)
    # Every ./assets/... URL the page pulls is in the served allow-list.
    import re

    for ref in re.findall(r'\./(assets/[^"\']+)', html):
        assert ref in allowed, f"{ref} not in the static allow-list"
    # And the page never reaches for a working file or a source path.
    for forbidden in ("inventory.sqlite", "review_state.json", "file://", "/photos/"):
        assert forbidden not in html


def test_reinstalling_replaces_stale_assets(tmp_path):
    """A rebuild must not leave an orphaned old chunk the manifest no longer names."""
    review_page.install_review_ui(tmp_path)
    orphan = tmp_path / "assets" / "review-DEADBEEF.js"
    orphan.write_text("stale", encoding="utf-8")
    review_page.install_review_ui(tmp_path)
    assert not orphan.exists()
    allowed = review_assets.allowed_static_paths(tmp_path)
    assert "assets/review-DEADBEEF.js" not in allowed


def test_stage3_cli_passes_config_and_root(monkeypatch):
    received = {}
    monkeypatch.setattr(
        stage3_report, "run", lambda **kwargs: received.update(kwargs) or {}
    )

    assert stage3_report.main([
        "--config", "runtime.yaml", "--root", "/photos",
    ]) == 0
    assert received == {
        "config_path": "runtime.yaml", "root_override": "/photos",
    }


def test_missing_committed_build_fails_loudly(tmp_path, monkeypatch):
    """A broken checkout (no committed dist) is an error, not a silent empty page."""
    import pytest

    monkeypatch.setattr(review_assets, "dist_dir", lambda: tmp_path / "does-not-exist")
    with pytest.raises(FileNotFoundError, match="review front-end build"):
        review_page.install_review_ui(tmp_path / "out")
