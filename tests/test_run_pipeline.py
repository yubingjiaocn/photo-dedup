from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest
import yaml

from src import db
from src import execute_local
from src import run_pipeline


def _seed_output(output: Path) -> Path:
    """Minimal served output directory: review page + a real (empty) review DB."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "review.html").write_text(
        '<h1>Photo Dedup - Review</h1><div class="notice" id="perf">old</div>',
        encoding="utf-8",
    )
    conn = db.open_db(output / "inventory.sqlite")
    conn.commit()
    conn.close()
    return output


def test_pipeline_order_limit_output_and_never_execute(tmp_path, monkeypatch):
    root = tmp_path / "photos"
    root.mkdir()
    output = tmp_path / "review"
    calls = []

    def record(name, result):
        def fake(*args, **kwargs):
            config = yaml.safe_load(Path(kwargs["config_path"]).read_text(encoding="utf-8"))
            calls.append((name, kwargs, config))
            if name == "stage3":
                _seed_output(output)
            return result

        return fake

    monkeypatch.setattr(run_pipeline.stage0_inventory, "run", record(
        "stage0", {"files": 2, "library_still_images": 20}
    ))
    monkeypatch.setattr(
        run_pipeline.stage1_features, "run",
        record("stage1", {"processed": 2, "thumbnails": {"created": 2, "cache_files": 2,
                                                         "cache_bytes": 4096,
                                                         "average_bytes": 2048.0}}),
    )
    monkeypatch.setattr(run_pipeline.stage2_cluster, "run", record("stage2", {"groups": 1}))
    monkeypatch.setattr(
        run_pipeline.stage3_report,
        "run",
        record("stage3", {"groups": 1, "maybe": 1, "unknown": 0, "delete_files": 0,
                          "all_items": 2, "thumbnails": {"recorded_ok": 2}}),
    )
    monkeypatch.setattr(
        execute_local,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not execute")),
    )

    result = run_pipeline.run(
        str(root), str(output), backend="stub", limit=2, review_limit=37
    )

    assert [call[0] for call in calls] == ["stage0", "stage1", "stage2", "stage3"]
    assert calls[0][1]["limit"] == 2
    assert calls[1][1]["limit"] == 2
    assert calls[1][1]["backend_override"] == "stub"
    assert calls[3][1]["review_limit"] == 37
    assert all(call[2]["paths"]["output_dir"] == str(output.resolve()) for call in calls)
    assert all(call[2]["paths"]["db"] == str(output.resolve() / "inventory.sqlite") for call in calls)
    # Runtime config enables the SSD thumbnail cache without touching config.yaml.
    assert all(call[2]["features"]["thumbnails"]["enabled"] is True for call in calls)
    assert calls[1][2]["features"]["thumbnails"]["max_px"] == 320
    assert result["review_html"] == str(output.resolve() / "review.html")
    assert result["performance"]["inventory_still_images"] == 20


def test_pipeline_reports_stage_times_throughput_and_rough_eta(tmp_path, monkeypatch):
    root = tmp_path / "photos"
    root.mkdir()
    output = tmp_path / "review"

    def stage(name, result):
        def fake(*args, **kwargs):
            if name == "stage3":
                _seed_output(output)
            return result

        return fake

    monkeypatch.setattr(run_pipeline.stage0_inventory, "run", stage("stage0", {"files": 5}))
    monkeypatch.setattr(
        run_pipeline.stage1_features, "run",
        stage("stage1", {"processed": 5, "thumbnails": {"created": 5, "cache_files": 5,
                                                        "cache_bytes": 5 * 30000,
                                                        "average_bytes": 30000.0}}),
    )
    monkeypatch.setattr(run_pipeline.stage2_cluster, "run", stage("stage2", {"groups": 0}))
    monkeypatch.setattr(
        run_pipeline.stage3_report, "run",
        stage("stage3", {"groups": 0, "maybe": 0, "unknown": 0, "delete_files": 0,
                         "all_items": 5, "output_dir": str(output),
                         "thumbnails": {"recorded_ok": 5}}),
    )
    monkeypatch.setattr(run_pipeline, "_still_image_count", lambda _path: 5)

    result = run_pipeline.run(str(root), str(output), backend="stub")

    performance = result["performance"]
    assert set(performance["stage_seconds"]) == {"stage0", "stage1", "stage2", "stage3"}
    assert performance["stage1_processed"] == 5
    assert performance["stage1_images_per_second"] > 0
    assert performance["stage0_files_per_second"] > 0
    assert performance["eta_hours"] is not None
    assert "ROUGH LINEAR ESTIMATE ONLY" in performance["eta_disclaimer"]
    # ETA scope is this inventory, never a hardcoded 1 TiB library.
    assert performance["inventory_still_images"] == 5
    assert "1 TiB" not in performance["eta_basis"]

    text = (output / "performance.txt").read_text(encoding="utf-8")
    assert "stage0 wall time" in text and "stage3 wall time" in text
    assert "files/s" in text and "images/s" in text
    assert "ROUGH full-library Stage 1 ETA" in text
    assert "Thumbnail cache (SSD)" in text
    assert "SSD free space" in text

    page = (output / "review.html").read_text(encoding="utf-8")
    assert "Observed performance and thumbnail disk usage" in page
    assert "ROUGH full-library Stage 1 ETA" in page


def test_low_disk_space_warns_but_does_not_abort(tmp_path, monkeypatch, capsys):
    output = tmp_path / "out"
    output.mkdir()
    monkeypatch.setattr(run_pipeline.pipeline_report, "free_bytes", lambda _path: 3 * 1024 ** 3)
    plan = run_pipeline.pipeline_report.print_thumbnail_plan(output, 100000, 30000.0)
    assert plan["low_space"] is True
    captured = capsys.readouterr().out
    assert "less than 20 GiB free" in captured
    assert "advisory only" in captured
    # 50 GiB is never demanded anywhere in the notice.
    assert "50 GiB" not in captured


def test_main_reports_clear_error_for_missing_root(tmp_path, capsys):
    output = tmp_path / "out"
    status = run_pipeline.main(
        ["--root", str(tmp_path / "missing"), "--output", str(output)]
    )
    assert status == 1
    assert "photo root is not a directory" in capsys.readouterr().err
    log = (output / "photo-dedup.log").read_text(encoding="utf-8")
    assert "[diagnostics] Python:" in log
    assert "FileNotFoundError" in log
    assert "photo root is not a directory" in log


def test_main_writes_successful_console_output_to_diagnostic_log(tmp_path, monkeypatch):
    root = tmp_path / "photos"
    root.mkdir()
    output = tmp_path / "output"

    def fake_run(*_args, **_kwargs):
        print("pipeline diagnostic marker")
        return {"report": {"output_dir": str(output)}}

    monkeypatch.setattr(run_pipeline, "run", fake_run)
    status = run_pipeline.main(
        ["--root", str(root), "--output", str(output), "--no-serve"]
    )

    assert status == 0
    log = (output / "photo-dedup.log").read_text(encoding="utf-8")
    assert "pipeline diagnostic marker" in log
    assert "backend='torch'" in log


def test_server_is_local_output_root_and_returns_review(tmp_path):
    output = _seed_output(tmp_path / "output")
    (tmp_path / "secret.txt").write_text("outside", encoding="utf-8")
    server, url = run_pipeline.start_review_server(output)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert server.server_address[0] == "127.0.0.1"
        assert url.endswith("/review.html")
        assert b"Photo Dedup - Review" in urlopen(url).read()
        for probe in ("/../secret.txt", "/inventory.sqlite", "/thumbs/1.jpg"):
            with pytest.raises(HTTPError) as excinfo:
                urlopen(url.rsplit("/", 1)[0] + probe)
            assert excinfo.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        server.review_data.close()
        thread.join()


def test_server_refuses_to_start_without_a_review_database(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "review.html").write_text("review", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="review database does not exist"):
        run_pipeline.start_review_server(output)


def test_main_serves_only_after_pipeline_and_honors_no_open(tmp_path, monkeypatch):
    root = tmp_path / "photos"
    root.mkdir()
    output = tmp_path / "output"
    calls = []

    def fake_run(*args, **kwargs):
        calls.append("pipeline")
        return {"report": {"output_dir": str(output)}}

    def fake_serve(path, port, open_browser):
        calls.append(("serve", path, port, open_browser))

    monkeypatch.setattr(run_pipeline, "run", fake_run)
    monkeypatch.setattr(run_pipeline, "serve_review", fake_serve)
    status = run_pipeline.main(
        ["--root", str(root), "--output", str(output), "--port", "8765", "--no-open"]
    )
    assert status == 0
    assert calls == ["pipeline", ("serve", output, 8765, False)]


def test_main_passes_config_through_to_run(tmp_path, monkeypatch):
    """``--config`` is optional and reaches :func:`run` as the tunables base."""
    root = tmp_path / "photos"
    root.mkdir()
    output = tmp_path / "output"
    tunables = tmp_path / "my-thresholds.yaml"
    tunables.write_text("cluster:\n  dinov2_threshold: 0.93\n", encoding="utf-8")
    received = {}

    def fake_run(*args, **kwargs):
        received.update(kwargs)
        return {"report": {"output_dir": str(output)}}

    monkeypatch.setattr(run_pipeline, "run", fake_run)
    status = run_pipeline.main([
        "--root", str(root), "--output", str(output),
        "--config", str(tunables), "--no-serve",
    ])
    assert status == 0
    assert received["config_path"] == str(tunables)


def test_main_config_is_optional(tmp_path, monkeypatch):
    root = tmp_path / "photos"
    root.mkdir()
    output = tmp_path / "output"
    received = {}

    def fake_run(*args, **kwargs):
        received.update(kwargs)
        return {"report": {"output_dir": str(output)}}

    monkeypatch.setattr(run_pipeline, "run", fake_run)
    assert run_pipeline.main(
        ["--root", str(root), "--output", str(output), "--no-serve"]
    ) == 0
    assert received["config_path"] is None


@pytest.mark.parametrize("argv", [[], ["--root", "R"], ["--output", "O"]])
def test_main_requires_root_and_output(argv, capsys):
    """Same mandatory shape as ``rebuild_review``; a config cannot supply paths."""
    with pytest.raises(SystemExit) as excinfo:
        run_pipeline.main(argv)
    assert excinfo.value.code == 2
    assert "required" in capsys.readouterr().err


def test_explicit_config_supplies_tunables_but_never_paths(tmp_path, monkeypatch):
    """``--root``/``--output`` win over any ``paths:`` the config declares.

    A config whose ``paths.db`` pointed somewhere else used to be the documented
    way to run; now it is inert, so a run can only ever write the database that
    belongs to the ``--output`` the user typed.
    """
    root = tmp_path / "photos"
    root.mkdir()
    output = tmp_path / "review"
    decoy = tmp_path / "decoy"
    tunables = tmp_path / "tunables.yaml"
    tunables.write_text(yaml.safe_dump({
        "paths": {"root": str(decoy / "photos"), "db": str(decoy / "other.sqlite"),
                  "output_dir": str(decoy)},
        "cluster": {"dinov2_threshold": 0.931, "burst_window_seconds": 17},
        "quality": {"weight_iqa": 0.55},
    }), encoding="utf-8")
    seen = []

    def record(name, result):
        def fake(*args, **kwargs):
            config = yaml.safe_load(Path(kwargs["config_path"]).read_text(encoding="utf-8"))
            seen.append((name, config))
            if name == "stage3":
                _seed_output(output)
            return result

        return fake

    monkeypatch.setattr(run_pipeline.stage0_inventory, "run", record("stage0", {"files": 1}))
    monkeypatch.setattr(run_pipeline.stage1_features, "run", record("stage1", {
        "processed": 1, "thumbnails": {"created": 1, "cache_files": 1,
                                       "cache_bytes": 2048, "average_bytes": 2048.0}}))
    monkeypatch.setattr(run_pipeline.stage2_cluster, "run", record("stage2", {"groups": 0}))
    monkeypatch.setattr(run_pipeline.stage3_report, "run", record("stage3", {
        "groups": 0, "maybe": 0, "unknown": 0, "delete_files": 0, "all_items": 1,
        "thumbnails": {"recorded_ok": 1}}))

    run_pipeline.run(
        str(root), str(output), backend="stub", thumb_px=256, config_path=str(tunables)
    )

    assert [name for name, _ in seen] == ["stage0", "stage1", "stage2", "stage3"]
    for _name, config in seen:
        # paths: from the CLI, always
        assert config["paths"]["root"] == str(root.resolve())
        assert config["paths"]["db"] == str(output.resolve() / "inventory.sqlite")
        assert config["paths"]["output_dir"] == str(output.resolve())
        assert str(decoy) not in yaml.safe_dump(config["paths"])
        # tunables: from the config, unchanged
        assert config["cluster"]["dinov2_threshold"] == 0.931
        assert config["cluster"]["burst_window_seconds"] == 17
        assert config["quality"]["weight_iqa"] == 0.55
        # Stage 1 knobs: from the CLI, because run_pipeline owns Stage 1
        assert config["features"]["backend"] == "stub"
        assert config["features"]["thumbnails"]["max_px"] == 256
    assert not decoy.exists()


def test_paths_with_spaces_survive_round_trip(tmp_path, monkeypatch):
    """``C:\\photo review`` style paths: quoting must be all the user needs."""
    root = tmp_path / "My Photos 2026"
    root.mkdir()
    output = tmp_path / "photo review out"
    seen = []

    def record(name, result):
        def fake(*args, **kwargs):
            seen.append(yaml.safe_load(
                Path(kwargs["config_path"]).read_text(encoding="utf-8")))
            if name == "stage3":
                _seed_output(output)
            return result

        return fake

    monkeypatch.setattr(run_pipeline.stage0_inventory, "run", record("stage0", {"files": 1}))
    monkeypatch.setattr(run_pipeline.stage1_features, "run", record("stage1", {
        "processed": 1, "thumbnails": {"created": 1, "cache_files": 1,
                                       "cache_bytes": 2048, "average_bytes": 2048.0}}))
    monkeypatch.setattr(run_pipeline.stage2_cluster, "run", record("stage2", {"groups": 0}))
    monkeypatch.setattr(run_pipeline.stage3_report, "run", record("stage3", {
        "groups": 0, "maybe": 0, "unknown": 0, "delete_files": 0, "all_items": 1,
        "thumbnails": {"recorded_ok": 1}}))

    result = run_pipeline.run(str(root), str(output), backend="stub")

    assert seen and all(c["paths"]["root"] == str(root.resolve()) for c in seen)
    assert all(
        c["paths"]["db"] == str(output.resolve() / "inventory.sqlite") for c in seen
    )
    assert result["review_html"] == str(output.resolve() / "review.html")


def test_main_passes_review_limit_and_thumb_px(tmp_path, monkeypatch):
    root = tmp_path / "photos"
    root.mkdir()
    output = tmp_path / "output"
    received = {}

    def fake_run(*args, **kwargs):
        received.update(kwargs)
        return {"report": {"output_dir": str(output)}}

    monkeypatch.setattr(run_pipeline, "run", fake_run)
    status = run_pipeline.main(
        [
            "--root", str(root), "--output", str(output),
            "--review-limit", "23", "--thumb-px", "256", "--no-serve",
        ]
    )
    assert status == 0
    assert received["review_limit"] == 23
    assert received["thumb_px"] == 256
