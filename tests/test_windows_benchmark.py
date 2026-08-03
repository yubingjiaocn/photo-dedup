from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src import windows_benchmark as benchmark


def _sample(tmp_path: Path, count: int = 2) -> Path:
    root = tmp_path / "selected"
    root.mkdir()
    for index in range(count):
        Image.new("RGB", (12, 12), (index, 0, 0)).save(root / f"private-{index}.png")
    return root


def test_requires_explicit_source_and_never_uses_config_root(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        benchmark.run(output=str(tmp_path / "out.json"))


def test_limit_dry_run_and_path_privacy(tmp_path):
    root = _sample(tmp_path)
    result = benchmark.run(sample_dir=str(root), limit=1, dry_run=True, output=str(tmp_path / "out.json"))
    encoded = (tmp_path / "out.json").read_text()
    assert result["sample_count"] == 1
    assert "private-0" not in encoded and str(root) not in encoded


def test_state_identity_mismatch_restarts_not_resumes(tmp_path):
    root, state = _sample(tmp_path), tmp_path / "state.json"
    paths, selection = benchmark._samples(str(root), None, None)
    state.write_text(json.dumps({"state_version": 2, "schema_version": 1, "config_hash": "old",
        "sample_set_hash": benchmark._hash(selection), "model_hash": "old", "runner_spec": "builtin",
        "completed": {benchmark._path_token(paths[0]): {}}}))
    result = benchmark.run(sample_dir=str(root), state=str(state), output=str(tmp_path / "out.json"))
    assert result["resume_status"] == "STATE_MISMATCH_RESTARTED"
    assert result["phases"]["decode"]["processed"] == 2


def test_atomic_writer_keeps_old_output_when_replace_fails(tmp_path, monkeypatch):
    output = tmp_path / "out.json"
    output.write_text('{"old":true}')
    monkeypatch.setattr(benchmark.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("nope")))
    with pytest.raises(OSError):
        benchmark._atomic_json(output, {"new": True})
    assert output.read_text() == '{"old":true}'
    assert not list(tmp_path.glob(".*.tmp"))


def test_skipped_siglip_is_not_processed(tmp_path):
    root = _sample(tmp_path, 1)
    result = benchmark.run(sample_dir=str(root), output=str(tmp_path / "out.json"))
    siglip = result["phases"]["siglip"]
    assert siglip["processed"] == 0 and siglip["skipped"] == 1 and siglip["throughput_per_s"] is None


def test_fixture_rejects_escape_and_bad_types(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "fixtures": [{"id": "x", "file": "../out.png", "buckets": ["x"]}]}))
    with pytest.raises(ValueError, match="escapes"):
        benchmark._samples(None, str(manifest), None)
    manifest.write_text(json.dumps({"schema_version": 1, "fixtures": [{"id": 1, "file": "x.png", "buckets": []}]}))
    with pytest.raises(ValueError, match="string id"):
        benchmark._samples(None, str(manifest), None)


def test_windows_style_path_token_is_opaque():
    token = benchmark._path_token(Path(r"C:\\Users\\Willy\\Photos\\x.jpg"))
    assert len(token) == 16 and "Willy" not in token
