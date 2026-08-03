from pathlib import Path

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

    result = run_pipeline.run(str(root), str(output), backend="stub", limit=2)

    assert [call[0] for call in calls] == ["stage0", "stage1", "stage2", "stage3"]
    assert calls[0][1]["limit"] == 2
    assert calls[1][1]["limit"] == 2
    assert calls[1][1]["backend_override"] == "stub"
    assert all(call[2]["paths"]["output_dir"] == str(output.resolve()) for call in calls)
    assert all(call[2]["paths"]["db"] == str(output.resolve() / "inventory.sqlite") for call in calls)
    assert result["review_html"] == str(output.resolve() / "review.html")


def test_main_reports_clear_error_for_missing_root(tmp_path, capsys):
    status = run_pipeline.main(
        ["--root", str(tmp_path / "missing"), "--output", str(tmp_path / "out")]
    )
    assert status == 1
    assert "photo root is not a directory" in capsys.readouterr().err
