"""Local trial: multi-keep / keep-all review actions and the data-only export."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from urllib.error import HTTPError

import pytest

from src import local_trial_export, review_state
from tests.test_review_actions import _Served, _build_output_with_groups, _get, _post


def _with_shas(output):
    conn = sqlite3.connect(output / "inventory.sqlite")
    for (fid,) in conn.execute("SELECT id FROM files").fetchall():
        conn.execute("UPDATE features SET content_sha256=? WHERE file_id=?",
                     (hashlib.sha256(str(fid).encode()).hexdigest(), fid))
    conn.commit()
    conn.close()


def _post_error(base, payload):
    with pytest.raises(HTTPError) as err:
        _post(base, payload)
    return json.loads(err.value.read())["error"]


def test_keep_multiple_and_keep_all_persist_reload_and_export(tmp_path):
    output, photos = _build_output_with_groups(tmp_path, 6)
    _with_shas(output)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in photos.iterdir()}
    with _Served(output) as base:
        assert _post(base, {"group_id": 1, "action": "keep", "file_ids": [2, 1]})["ok"]
        assert _post(base, {"group_id": 2, "action": "pick", "file_id": 4})["ok"]
        assert _post(base, {"group_id": 3, "action": "mark"})["ok"]
        assert "not a member" in _post_error(base, {"group_id": 2, "action": "keep", "file_ids": [1]})
        assert "unique" in _post_error(base, {"group_id": 2, "action": "keep", "file_ids": [3, 3]})
        assert "requires" in _post_error(base, {"group_id": 2, "action": "keep", "file_ids": []})
        assert "must not include file_ids" in _post_error(
            base, {"group_id": 2, "action": "pick", "file_id": 4, "file_ids": [4]})

    # Reopen without any scan: the multi-keep is still there and still DONE.
    with _Served(output) as base:
        page = _get(base, "/api/page?view=GROUPS&queue=DONE&page=1&page_size=100")
        stored = page["review_state"]["1"]
        assert stored["action"] == "keep" and stored["file_ids"] == [1, 2]
        assert stored["keep_all"] is True
        assert stored["file_sha256"] == [hashlib.sha256(b"1").hexdigest(),
                                         hashlib.sha256(b"2").hexdigest()]
        assert page["queue_counts"] == {"PENDING": 0, "LATER": 1, "DONE": 2}

    manifest = local_trial_export.write_export(output)
    rows = [json.loads(line) for line in
            (output / "local_trial_export" / manifest["data_file"]).read_text("utf-8").splitlines()]
    assert manifest["counts"] == {"groups_exported": 2, "groups_later": 1, "groups_stale": 0,
                                  "selected": 3, "nonselected": 1}
    assert {(r["group_id"], r["file_id"], r["selected"]) for r in rows} == {
        (1, 1, True), (1, 2, True), (2, 3, False), (2, 4, True)}
    assert all("path" not in r and r["content_sha256"] for r in rows)
    assert manifest["contains_photos"] is False and manifest["model_calls"] == 0
    # Originals are byte- and mtime-identical.
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in photos.iterdir()}


def test_export_reports_changed_sha_and_members_as_stale(tmp_path):
    output, _ = _build_output_with_groups(tmp_path, 4)
    _with_shas(output)
    with _Served(output) as base:
        _post(base, {"group_id": 1, "action": "keep", "file_ids": [1]})
        _post(base, {"group_id": 2, "action": "accept"})
    conn = sqlite3.connect(output / "inventory.sqlite")
    conn.execute("UPDATE features SET content_sha256=? WHERE file_id=1", ("0" * 64,))
    conn.execute("DELETE FROM group_members WHERE group_id=2 AND file_id=4")
    conn.commit()
    conn.close()
    result = local_trial_export.build_records(output)
    reasons = {s["group_id"]: s["reason"] for s in result["stale"]}
    assert "content_sha256 changed" in reasons[1]
    assert "members changed" in reasons[2]
    assert result["records"] == []


def test_legacy_pick_untouched_and_malformed_keep_dropped_on_load(tmp_path):
    output, _ = _build_output_with_groups(tmp_path, 4)
    legacy = {"action": "pick", "file_id": 2, "timestamp": 1, "member_fingerprint": "x"}
    bad_keep = {"action": "keep", "file_ids": [2, 1], "file_sha256": [None, None]}
    smuggled = {"action": "pick", "file_id": 3, "file_ids": [3, 4]}
    (output / "review_state.json").write_text(json.dumps(
        {"version": 1, "groups": {"1": legacy, "2": bad_keep, "3": smuggled}, "history": []}),
        encoding="utf-8")
    state = review_state.ReviewState(output)
    assert state.get_group(1) == legacy
    assert state.get_group(2) is None and state.get_group(3) is None
    with pytest.raises(ValueError):
        state.apply_decision(1, {"action": "keep", "file_ids": [1, 2], "file_sha256": [None]})
    assert review_state.queue_of({"action": "keep"}) == "DONE"


def test_export_cli_refuses_missing_output(tmp_path, capsys):
    assert local_trial_export.main(["--output", str(tmp_path / "nope")]) == 1
    assert "[export][ERROR]" in capsys.readouterr().err


def test_launcher_refuses_output_inside_root_or_app_and_reopen_never_scans(tmp_path, monkeypatch):
    from src import local_trial, run_pipeline, serve_review

    root = tmp_path / "照片 新日期"
    root.mkdir()
    assert local_trial.main(["start", "--root", str(root), "--output", str(root / "out")]) == 2
    assert local_trial.main(["start", "--root", str(root),
                             "--output", str(local_trial.APP_DIR / "output-x")]) == 2
    assert not (local_trial.APP_DIR / "output-x").exists()
    calls = []
    monkeypatch.setattr(run_pipeline, "main", lambda a: calls.append(("scan", a)) or 0)
    monkeypatch.setattr(serve_review, "main", lambda a: calls.append(("serve", a)) or 0)
    out = tmp_path / "输出 out"
    assert local_trial.main(["start", "--root", str(root), "--output", str(out), "--no-open"]) == 0
    kind, args = calls[-1]
    assert kind == "scan" and args[args.index("--backend") + 1] == "torch"
    assert args[args.index("--limit") + 1] == "100"
    out.mkdir()
    (out / "inventory.sqlite").write_bytes(b"")
    (out / "review_summary.json").write_text("{}", encoding="utf-8")
    assert local_trial.main(["start", "--root", str(root), "--output", str(out)]) == 0
    assert calls[-1][0] == "serve"
    assert local_trial.main(["reopen", "--output", str(out)]) == 0
    assert calls[-1][0] == "serve"
    assert local_trial.main(["start", "--root", str(root), "--output", str(out),
                             "--rescan", "--smoke-stub"]) == 0
    assert calls[-1][1][calls[-1][1].index("--backend") + 1] == "stub"


def test_finished_review_refuses_different_root_before_serving(tmp_path, monkeypatch):
    from src import local_trial, root_scope, serve_review

    output, photos = _build_output_with_groups(tmp_path, 4)
    conn = sqlite3.connect(output / "inventory.sqlite")
    conn.row_factory = sqlite3.Row
    root_scope.bind(conn, photos)
    conn.commit()
    conn.close()
    (output / "review_summary.json").write_text("{}", encoding="utf-8")
    another_root = tmp_path / "other-date"
    another_root.mkdir()
    calls = []
    monkeypatch.setattr(serve_review, "main", lambda a: calls.append(a) or 0)
    assert local_trial.main(["start", "--root", str(another_root),
                             "--output", str(output)]) == 2
    assert calls == []
    assert local_trial.main(["start", "--root", str(photos),
                             "--output", str(output)]) == 0
    assert len(calls) == 1
