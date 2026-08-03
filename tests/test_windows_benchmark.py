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


def test_interrupt_state_resumes_and_json_is_finite(tmp_path):
    root = _sample(tmp_path)
    class Interrupted:
        model_descriptor = {"fake": "one"}
        calls = 0
        def siglip(self, image):
            self.calls += 1
            if self.calls == 2:
                raise KeyboardInterrupt()
        def stage1(self, image):
            return float("nan")
    runner = Interrupted()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(benchmark, "_load_runner", lambda _: runner)
    state, output = tmp_path / "state.json", tmp_path / "out.json"
    with pytest.raises(KeyboardInterrupt):
        benchmark.run(sample_dir=str(root), state=str(state), output=str(output))
    assert len(json.loads(state.read_text())["completed"]) == 1
    monkeypatch.setattr(benchmark, "_load_runner", lambda _: benchmark.BuiltinRunner())
    result = benchmark.run(sample_dir=str(root), state=str(state), output=str(output))
    assert result["status"] == "COMPLETE" and not state.exists()
    assert "NaN" not in output.read_text()
    monkeypatch.undo()


def test_windows_style_path_token_is_opaque():
    token = benchmark._path_token(Path(r"C:\\Users\\Willy\\Photos\\x.jpg"))
    assert len(token) == 16 and "Willy" not in token
