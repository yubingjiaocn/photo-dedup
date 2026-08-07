from pathlib import Path

from src import serve_review


def test_missing_output_fails_cleanly(tmp_path, capsys):
    status = serve_review.main(["--output", str(tmp_path / "missing")])
    assert status == 1
    assert "output directory not found" in capsys.readouterr().err


def test_missing_inventory_fails_before_install(tmp_path, monkeypatch, capsys):
    output = tmp_path / "output"
    output.mkdir()
    called = []
    monkeypatch.setattr(serve_review.review_assets, "install_into",
                        lambda _path: called.append(True))
    status = serve_review.main(["--output", str(output)])
    assert status == 1
    assert called == []
    assert "inventory DB not found" in capsys.readouterr().err


def test_refreshes_ui_then_serves_existing_output(tmp_path, monkeypatch):
    output = tmp_path / "review output"
    output.mkdir()
    (output / "inventory.sqlite").write_bytes(b"existing-db")
    calls = []

    def install(path: Path):
        calls.append(("install", path))
        return {"assets": ["assets/review.js", "assets/review.css"]}

    def serve(path: Path, port: int, open_browser: bool):
        calls.append(("serve", path, port, open_browser))

    monkeypatch.setattr(serve_review.review_assets, "install_into", install)
    monkeypatch.setattr(serve_review, "serve_review", serve)

    assert serve_review.main([
        "--output", str(output), "--port", "8765", "--no-open",
    ]) == 0
    resolved = output.resolve()
    assert calls == [
        ("install", resolved),
        ("serve", resolved, 8765, False),
    ]
    assert (output / "inventory.sqlite").read_bytes() == b"existing-db"


def test_runtime_failure_is_reported(tmp_path, monkeypatch, capsys):
    output = tmp_path / "output"
    output.mkdir()
    (output / "inventory.sqlite").write_bytes(b"db")
    monkeypatch.setattr(serve_review.review_assets, "install_into",
                        lambda _path: {"assets": []})
    monkeypatch.setattr(serve_review, "serve_review",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("busy")))
    assert serve_review.main(["--output", str(output)]) == 1
    assert "busy" in capsys.readouterr().err
