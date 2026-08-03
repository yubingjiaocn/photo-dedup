from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlopen

import yaml

from src import execute_local
from src import run_pipeline


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
                (output / "review.html").write_text("review", encoding="utf-8")
            return result

        return fake

    monkeypatch.setattr(run_pipeline.stage0_inventory, "run", record("stage0", {"files": 2}))
    monkeypatch.setattr(
        run_pipeline.stage1_features, "run", record("stage1", {"processed": 2})
    )
    monkeypatch.setattr(run_pipeline.stage2_cluster, "run", record("stage2", {"groups": 1}))
    monkeypatch.setattr(
        run_pipeline.stage3_report,
        "run",
        record("stage3", {"groups": 1, "maybe": 1, "unknown": 0, "delete_files": 0}),
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
    assert result["review_html"] == str(output.resolve() / "review.html")


def test_main_reports_clear_error_for_missing_root(tmp_path, capsys):
    status = run_pipeline.main(
        ["--root", str(tmp_path / "missing"), "--output", str(tmp_path / "out")]
    )
    assert status == 1
    assert "photo root is not a directory" in capsys.readouterr().err


def test_server_is_local_output_root_and_returns_review(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "review.html").write_text("review-page", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("outside", encoding="utf-8")
    server, url = run_pipeline.start_review_server(output)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert server.server_address[0] == "127.0.0.1"
        assert url.endswith("/review.html")
        assert urlopen(url).read() == b"review-page"
        try:
            urlopen(url.rsplit("/", 1)[0] + "/../secret.txt")
        except HTTPError as exc:
            assert exc.code == 404
        else:
            raise AssertionError("server escaped its output root")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


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


def test_main_passes_review_limit(tmp_path, monkeypatch):
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
            "--review-limit", "23", "--no-serve",
        ]
    )
    assert status == 0
    assert received["review_limit"] == 23
