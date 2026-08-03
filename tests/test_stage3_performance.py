import base64
import io

from PIL import Image

from src import stage3_report


def _rec(index, decision="MAYBE"):
    return {
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


def test_large_review_queues_only_generate_limited_thumbnails_and_report_omitted(monkeypatch):
    calls = []
    monkeypatch.setattr(
        stage3_report, "thumb_data_uri",
        lambda path, max_px: calls.append((path, max_px)) or "data:image/jpeg;base64,x",
    )
    data = {
        "groups": [],
        "delete_paths": [],
        "total_delete_bytes": 0,
        "queues": {
            "MAYBE": [_rec(i) for i in range(1500)],
            "UNKNOWN": [_rec(i, "UNKNOWN") for i in range(1200)],
        },
    }

    page = stage3_report.render_html(data, review_limit=1000)

    assert len(calls) == 2000
    assert "total: 1500" in page and "shown: 1000" in page and "omitted: 500" in page
    assert "total: 1200" in page and "omitted: 200" in page


def test_auto_remove_group_thumbnails_keep_existing_semantics(monkeypatch):
    calls = []
    monkeypatch.setattr(
        stage3_report, "thumb_data_uri",
        lambda path, max_px: calls.append(path) or "data:image/jpeg;base64,x",
    )
    keep = _rec("keep", "KEEP")
    remove = _rec("remove", "AUTO_REMOVE")
    data = {
        "groups": [{"id": 1, "type": "exact_dup", "member_count": 2,
                    "keep": keep, "deletes": [remove]}],
        "delete_paths": [remove["path"]],
        "total_delete_bytes": 1,
        "queues": {"MAYBE": [], "UNKNOWN": []},
    }

    stage3_report.render_html(data, review_limit=1)

    assert calls == [keep["path"], remove["path"]]


def test_jpeg_draft_happens_before_convert_and_downsamples(tmp_path, monkeypatch):
    source = tmp_path / "large.jpg"
    Image.new("RGB", (4096, 3072), "navy").save(source, "JPEG")
    original_open = Image.open
    events = []
    drafted_sizes = []

    class TrackedImage:
        def __init__(self, image):
            self.image = image

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.image.close()

        @property
        def format(self):
            return self.image.format

        def draft(self, mode, size):
            events.append("draft")
            result = self.image.draft(mode, size)
            drafted_sizes.append(self.image.size)
            return result

        def convert(self, mode):
            events.append("convert")
            return self.image.convert(mode)

    monkeypatch.setattr(Image, "open", lambda path: TrackedImage(original_open(path)))

    uri = stage3_report.thumb_data_uri(str(source), 200)

    assert events == ["draft", "convert"]
    assert drafted_sizes[0][0] < 4096 and drafted_sizes[0][1] < 3072
    thumb = original_open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    assert max(thumb.size) <= 200


def test_unreadable_thumbnail_still_returns_placeholder_signal(tmp_path):
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not a jpeg")
    assert stage3_report.thumb_data_uri(str(broken)) is None


def test_stage3_cli_passes_review_limit(monkeypatch):
    received = {}
    monkeypatch.setattr(
        stage3_report, "run", lambda **kwargs: received.update(kwargs) or {}
    )

    assert stage3_report.main(["--config", "runtime.yaml", "--review-limit", "17"]) == 0
    assert received == {
        "config_path": "runtime.yaml", "no_thumbs": False, "review_limit": 17
    }
