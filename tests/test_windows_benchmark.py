from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src import windows_benchmark as benchmark
from src.windows_siglip_runner import WindowsSiglipRunner, _settings
from src.siglip_runtime import SiglipShadowProvider, artifact_sha256
from src.siglip_router import prompt_bank_hash


def _sample(tmp_path: Path, count: int = 2) -> Path:
    root = tmp_path / "selected"
    root.mkdir(parents=True)
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
    valid = {name: benchmark._skipped("test") for name in ("decode", "siglip", "stage1")}
    state.write_text(json.dumps({"state_version": 2, "schema_version": 1, "config_hash": "old",
        "sample_set_hash": benchmark._hash(selection), "model_hash": "old", "runner_spec": "builtin",
        "completed": {benchmark._path_token(paths[0]): valid}}))
    result = benchmark.run(sample_dir=str(root), state=str(state), output=str(tmp_path / "out.json"))
    assert result["resume_status"] == "STATE_MISMATCH_RESTARTED"
    assert result["phases"]["decode"]["processed"] == 2


def test_truncated_or_malicious_state_safely_restarts(tmp_path):
    root, state = _sample(tmp_path), tmp_path / "state.json"
    state.write_text('{"completed":')
    result = benchmark.run(sample_dir=str(root), state=str(state), output=str(tmp_path / "out.json"))
    assert result["resume_status"] == "STATE_INVALID_RESTARTED"
    state.write_text(json.dumps({"completed": {"foreign": {"decode": {"seconds": float("inf")}}}}))
    result = benchmark.run(sample_dir=str(root), state=str(state), output=str(tmp_path / "out2.json"))
    assert result["resume_status"] == "STATE_INVALID_RESTARTED"


def test_normal_interrupt_checkpoint_resumes(tmp_path, monkeypatch):
    root, state = _sample(tmp_path), tmp_path / "state.json"
    class Runner:
        calls = 0
        model_descriptor = {"test": "resume"}
        def siglip(self, image):
            self.calls += 1
            if self.calls == 2:
                raise KeyboardInterrupt()
            return {"status": "SKIPPED"}
        def stage1(self, image):
            return {}
    runner = Runner()
    monkeypatch.setattr(benchmark, "_load_runner", lambda _: runner)
    with pytest.raises(KeyboardInterrupt):
        benchmark.run(sample_dir=str(root), state=str(state), output=str(tmp_path / "out.json"))
    runner.calls = 0
    result = benchmark.run(sample_dir=str(root), state=str(state), output=str(tmp_path / "out.json"))
    assert result["resume_status"] == "RESUMED"


def test_cuda_phase_sync_brackets_elapsed_and_sync_failure_is_error(monkeypatch):
    events = []
    class CUDA:
        def synchronize(self):
            events.append("sync")
        def reset_peak_memory_stats(self):
            events.append("reset")
        def max_memory_allocated(self):
            events.append("peak")
            return 1024
    class Torch:
        cuda = CUDA()
    ticks = iter((10.0, 13.0))
    monkeypatch.setattr(benchmark, "_cuda", lambda: Torch())
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: (events.append("clock") or next(ticks)))
    phase, _ = benchmark._phase(lambda: events.append("fn") or {})
    assert events == ["sync", "reset", "clock", "fn", "sync", "clock", "peak"]
    assert phase["seconds"] == 3.0 and phase["errors"] == 0

    class BadCUDA(CUDA):
        calls = 0
        def synchronize(self):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("async failure")
    class BadTorch:
        cuda = BadCUDA()
    monkeypatch.setattr(benchmark, "_cuda", lambda: BadTorch())
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: 20.0)
    phase, _ = benchmark._phase(lambda: {})
    assert phase["status"] == "ERROR" and phase["errors"] == 1


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


def _siglip_settings(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    prompt = Path(__file__).parents[1] / "research" / "siglip_prompt_bank_v1.yaml"
    return {"model_path": str(model), "model_revision": "local-r1",
            "model_sha256": artifact_sha256(model), "prompt_bank_path": str(prompt),
            "prompt_bank_sha256": prompt_bank_hash(prompt), "device": "cpu", "precision": "float32"}


class _Scores:
    def __init__(self): self.image = None
    def score(self, image, prompts, **_):
        self.image = image
        return [0.1] * len(prompts)


def test_siglip_adapter_same_pil_unknown_scores_are_processed_and_no_path_leak(tmp_path):
    values, scores = _siglip_settings(tmp_path), _Scores()
    provider = SiglipShadowProvider(_settings(values), backend=scores)
    runner = WindowsSiglipRunner(values, provider=provider)
    image = Image.new("RGB", (10, 10))
    value = runner.siglip(image)
    assert scores.image is image and value["status"] == "PROCESSED"
    assert value["routing_record"]["scene_context"]["state"] == "UNKNOWN"
    assert str(tmp_path) not in json.dumps(runner.model_descriptor)


def test_siglip_adapter_identity_is_required_and_changes_checkpoint_identity(tmp_path, monkeypatch):
    values = _siglip_settings(tmp_path)
    with pytest.raises(ValueError, match="missing identity"):
        _settings({key: value for key, value in values.items() if key != "model_sha256"})
    root, state = _sample(tmp_path / "samples", 1), tmp_path / "state.json"
    runner = WindowsSiglipRunner(values, provider=SiglipShadowProvider(_settings(values), backend=_Scores()))
    monkeypatch.setattr(benchmark, "_load_runner", lambda _: runner)
    benchmark.run(sample_dir=str(root), state=str(state), output=str(tmp_path / "first.json"))
    # Completed runs remove state; inject a completed state with a descriptor mismatch.
    paths, selection = benchmark._samples(str(root), None, None)
    skipped = {name: benchmark._skipped("test") for name in ("decode", "siglip", "stage1")}
    state.write_text(json.dumps({"state_version": 2, "schema_version": 1,
        "config_hash": benchmark._hash(benchmark.load_config(None).as_dict()),
        "sample_set_hash": benchmark._hash(selection), "model_hash": "old",
        "runner_spec": "x", "completed": {benchmark._path_token(paths[0]): skipped}}))
    result = benchmark.run(sample_dir=str(root), runner_spec="x", state=str(state), output=str(tmp_path / "second.json"))
    assert result["resume_status"] == "STATE_MISMATCH_RESTARTED"


def test_siglip_adapter_provider_failure_is_error_not_throughput(tmp_path, monkeypatch):
    class BadProvider:
        runtime = type("Runtime", (), {"model_revision":"r", "model_sha256":"a" * 64,
            "prompt_sha256":"b" * 64, "device":"cpu", "precision":"float32", "batch_size":1})()
        bank = {"bank_version": 1, "tags": {"x": {}}}
        def __call__(self, _): raise RuntimeError("unavailable")
    runner = WindowsSiglipRunner(_siglip_settings(tmp_path), provider=BadProvider())
    monkeypatch.setattr(benchmark, "_load_runner", lambda _: runner)
    result = benchmark.run(sample_dir=str(_sample(tmp_path / "samples", 1)), output=str(tmp_path / "out.json"))
    assert result["phases"]["siglip"]["errors"] == 1
    assert result["phases"]["siglip"]["throughput_per_s"] is None
