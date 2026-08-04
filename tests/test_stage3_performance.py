"""Stage 3 must never open an original photo: it uses only the SSD thumb cache."""

from pathlib import Path

import pytest
from PIL import Image

from src import stage3_report, thumbnails


def _rec(index, decision="MAYBE", file_id=None):
    return {
        "file_id": index if file_id is None else file_id,
        "group_id": None,
        "path": f"/photos/{index}.jpg",
        "basename": f"{index}.jpg",
        "width": 4000,
        "height": 3000,
        "quality_score": 1.0,
        "face_count": 0,
        "reason": "review",
        "decision": decision,
        "quality_meta": {},
    }


def _data(maybe=(), unknown=(), groups=()):
    return {
        "groups": list(groups),
        "delete_paths": [],
        "total_delete_bytes": 0,
        "queues": {"MAYBE": list(maybe), "UNKNOWN": list(unknown)},
    }


def test_render_html_never_opens_a_source_image(tmp_path, monkeypatch):
    """Even for a huge queue, rendering must not call Image.open at all."""
    def explode(*args, **kwargs):
        raise AssertionError("stage 3 must not open original photos")

    monkeypatch.setattr(Image, "open", explode)
    data = _data(maybe=[_rec(i) for i in range(1500)],
                 unknown=[_rec(i, "UNKNOWN") for i in range(1200)])

    page = stage3_report.render_html(data, tmp_path, review_limit=100)

    assert "/api/thumb/" in page  # paged UI fetches cached thumbs by file id
    assert "1500" in page and "1200" in page


def _fallback(page: str) -> str:
    """Only the static file:// fallback block, excluding the JS template."""
    return page.rsplit('<div class="row">', 1)[1]


def test_static_fallback_uses_cached_thumbnail_when_present(tmp_path):
    directory = thumbnails.thumbs_dir(tmp_path)
    directory.mkdir(parents=True)
    Image.new("RGB", (32, 24), "navy").save(thumbnails.thumb_path(directory, 7), "JPEG")

    fallback = _fallback(stage3_report.render_html(
        _data(maybe=[_rec(7)]), tmp_path, review_limit=5))

    assert 'src="thumbs/7.jpg"' in fallback
    assert "thumbnail unavailable" not in fallback


def test_static_fallback_reports_missing_thumbnail_instead_of_reading_source(tmp_path):
    fallback = _fallback(stage3_report.render_html(
        _data(maybe=[_rec(9)]), tmp_path, review_limit=5))
    assert "thumbnail unavailable" in fallback
    assert "thumbs/9.jpg" not in fallback


def test_no_thumbs_omits_the_static_fallback_entirely(tmp_path):
    directory = thumbnails.thumbs_dir(tmp_path)
    directory.mkdir(parents=True)
    thumbnails.thumb_path(directory, 4).write_bytes(b"jpeg")
    fallback = _fallback(stage3_report.render_html(
        _data(maybe=[_rec(4)]), tmp_path, review_limit=0))
    assert "thumbs/4.jpg" not in fallback


def test_cached_thumb_uri_only_reports_existing_files(tmp_path):
    assert stage3_report.cached_thumb_uri(tmp_path, 3) is None
    assert stage3_report.cached_thumb_uri(tmp_path, None) is None
    directory = thumbnails.thumbs_dir(tmp_path)
    directory.mkdir(parents=True)
    thumbnails.thumb_path(directory, 3).write_bytes(b"jpeg")
    assert stage3_report.cached_thumb_uri(tmp_path, 3) == "thumbs/3.jpg"


def test_render_html_declares_the_page_size_choices_and_all_view(tmp_path):
    page = stage3_report.render_html(_data(), tmp_path, review_limit=1)
    assert 'data-view="ALL"' in page
    assert 'data-view="MAYBE"' in page
    assert 'data-view="UNKNOWN"' in page
    assert 'data-view="GROUPS"' in page
    assert "/api/original/" in page
    assert "与组内 KEEP 对比" in page
    assert "ArrowLeft" in page and "ArrowRight" in page
    assert "/api/group/" in page
    assert "leftImage.removeAttribute('src')" in page
    for size in (50, 100, 200):
        assert f'value="{size}"' in page


def test_render_html_rejects_a_negative_fallback_limit(tmp_path):
    with pytest.raises(ValueError, match="review_limit"):
        stage3_report.render_html(_data(), tmp_path, review_limit=-1)


def test_performance_panel_is_replaced_in_place(tmp_path):
    review = tmp_path / "review.html"
    review.write_text(
        '<h1>x</h1><div class="notice" id="perf">placeholder</div><div>rest</div>',
        encoding="utf-8",
    )
    assert stage3_report.rewrite_performance_panel(review, "measured 1.5s") is True
    text = review.read_text(encoding="utf-8")
    assert "measured 1.5s" in text
    assert "placeholder" not in text
    assert "<div>rest</div>" in text
    assert stage3_report.rewrite_performance_panel(Path(tmp_path / "missing.html"), "x") is False


def test_stage3_cli_passes_review_limit(monkeypatch):
    received = {}
    monkeypatch.setattr(
        stage3_report, "run", lambda **kwargs: received.update(kwargs) or {}
    )

    assert stage3_report.main(["--config", "runtime.yaml", "--review-limit", "17"]) == 0
    assert received == {
        "config_path": "runtime.yaml", "no_thumbs": False, "review_limit": 17,
        "root_override": None,
    }
