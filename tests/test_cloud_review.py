"""Offline integration against the actual 194-group batch; no model transport."""
import copy
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from src.cloud_review import CloudReviewData, digest

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "artifacts/cloud-judge-batch/authorized-scene-run-v3"
MANIFEST = ROOT / "artifacts/cloud-judge-batch/frozen.json"


def load(output, batch=BATCH, manifest=MANIFEST):
    if not MANIFEST.exists():
        pytest.skip("private real cloud fixture not installed")
    # Full legacy fixture includes sealed holdouts. Learning-loop development
    # exercises the same save chain with explicit nonholdout manifests instead.
    from src.review_events import HoldoutPolicy
    frozen = json.loads(MANIFEST.read_bytes())
    policy = HoldoutPolicy(frozen)
    if any(g["group"] in policy.groups for g in json.loads(Path(manifest).read_bytes())["groups"]):
        pytest.skip("sealed holdout: use test_learning_loop development-only coverage")
    return CloudReviewData(output, batch, manifest)


def protected_hashes():
    paths = list(BATCH.rglob("*"))
    for inv in json.loads(MANIFEST.read_bytes())["inventories"]:
        path = Path(inv["path"])
        paths.append(path)
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
            paths.extend(Path(r[0]) for r in conn.execute("select path from files"))
    return {str(p): digest(p.read_bytes()) for p in paths if p.is_file()}


def test_real_batch_actions_are_sidecar_only_and_restore(tmp_path):
    data = load(tmp_path)
    before = protected_hashes()
    groups = data.page(page_size=200)["items"]
    assert len(groups) == 194
    assert sum(len(g["members"]) for g in groups) == 510
    assert sum(g["ai"]["status"] == "select" for g in groups) == 185
    assert sum(g["ai"]["status"] == "abstain" for g in groups) == 2
    assert sum(g["ai"]["status"] == "manual" for g in groups) == 7
    for group in groups:
        assert len(group["ai"]["discriminators"]) <= 3
        for m in group["members"]:
            assert data.image_bytes(group["group_id"], m["file_id"], m["content_sha256"]).startswith(b"\xff\xd8")
    group = groups[0]
    payload = {"batch_id": data.batch_id, "group_id": group["group_id"]}
    for action, extra in [("accept", {}), ("modify", {"selected": data._frames(group)}),
                          ("keep_all", {}), ("feedback", {"inaccurate": True, "ignored_reasons": [0]}),
                          ("later", {}), ("accept", {})]:
        result = data.apply_action({**payload, "action": action, **extra})
        assert result.get("ok"), result
    assert result["group"]["review"]["feedback"] is True
    assert result["group"]["review"]["ignored_reasons"] == [0]
    assert data.status()["processed"] == 1
    assert data.page(queue="ACCEPTED")["total"] == 1
    assert data.page(queue="AI_ISSUES")["total"] == 9
    restored = load(tmp_path)
    assert restored.group(1)["review"] == data.group(1)["review"]
    assert protected_hashes() == before  # every original, DB and cloud raw byte
    assert len(data.journal.events()) == 6
    assert all(str(p.relative_to(tmp_path)).startswith(("review-state/", "review-events-v1/")) for p in tmp_path.rglob("*") if p.is_file())


def test_reject_identity_and_unsupported_actions(tmp_path):
    data = load(tmp_path)
    base = {"batch_id": data.batch_id, "group_id": 1}
    wrong = copy.deepcopy(data._frames(data.group(1)))
    wrong[0]["content_sha256"] = "0" * 64
    for extra in [{"action": "modify", "selected": wrong}, {"action": "modify", "selected": []},
                  {"action": "execute_local"}, {"action": "accept", "batch_id": "stale"},
                  {"action": "feedback", "inaccurate": True, "ignored_reasons": [99]}]:
        assert "error" in data.apply_action({**base, **extra})
    failed = next(g for g in data.page(page_size=200)["items"] if g["ai"]["status"] == "manual")
    assert not failed["ai"]["recommended"]
    assert "error" in data.apply_action({**base, "group_id": failed["group_id"], "action": "accept"})
    assert not data.state_path.exists()


def test_reject_stale_saved_state_without_overwrite(tmp_path):
    data = load(tmp_path)
    assert data.apply_action({"batch_id": data.batch_id, "group_id": 1, "action": "accept"})["ok"]
    saved = json.loads(data.state_path.read_bytes())
    for mutation in ("batch", "identity"):
        altered = copy.deepcopy(saved)
        if mutation == "batch":
            altered["batch_id"] = "old"
        else:
            next(iter(altered["groups"].values()))["members"][0]["content_sha256"] = "0" * 64
        data.state_path.write_text(json.dumps(altered))
        before = data.state_path.read_bytes()
        reopened = load(tmp_path)
        assert reopened.status()["state_warning"]
        assert reopened.status()["processed"] == 0
        assert "error" in reopened.apply_action({"batch_id": reopened.batch_id, "group_id": 1, "action": "accept"})
        assert data.state_path.read_bytes() == before


def test_reject_symlink_state_directory(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    (output / "review-state").symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        load(output)


def test_changed_raw_batch_refuses_existing_action(tmp_path):
    # Tiny copied batch; mutations target copies only.
    manifest = json.loads(MANIFEST.read_bytes())
    manifest["groups"] = [g for g in manifest["groups"] if g["group"] == "2026-04-04-G002"]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    batch = tmp_path / "batch"
    alias = manifest["groups"][0]["group"]
    shutil.copytree(BATCH / alias, batch / alias)
    data = load(tmp_path / "output", batch, manifest_path)
    raw = batch / alias / "raw.txt"
    raw.write_text(raw.read_text() + " ")
    assert "error" in data.apply_action({"batch_id": data.batch_id, "group_id": 1, "action": "accept"})
    with pytest.raises(ValueError, match="hash mismatch"):
        load(tmp_path / "output", batch, manifest_path)


def test_cloud_cli_requires_manifest(monkeypatch, tmp_path):
    from src import serve_review
    with pytest.raises(SystemExit):
        serve_review.main(["--output", str(tmp_path), "--cloud-batch", str(BATCH)])
    called = []
    monkeypatch.setattr(serve_review, "serve_cloud_review", lambda *args, **kwargs: called.append((args, kwargs)))
    assert serve_review.main(["--output", str(tmp_path), "--cloud-batch", str(BATCH),
                              "--cloud-manifest", str(MANIFEST), "--no-open"]) == 0
    assert called


def test_output_inside_original_directory_refused():
    inv = json.loads(MANIFEST.read_bytes())["inventories"][0]
    with sqlite3.connect(Path(inv["path"]).as_uri() + "?mode=ro", uri=True) as conn:
        source = Path(conn.execute("select path from files limit 1").fetchone()[0])
    with pytest.raises(ValueError, match="original photo directory"):
        load(source.parent / "never-created-review-output")
    assert not (source.parent / "never-created-review-output").exists()


def test_live_database_change_refuses_action_on_copy(tmp_path):
    manifest = json.loads(MANIFEST.read_bytes())
    manifest["groups"] = [g for g in manifest["groups"] if g["group"] == "2026-04-04-G002"]
    manifest["inventories"] = manifest["inventories"][:1]
    inv = manifest["inventories"][0]
    copied = tmp_path / "inventory.sqlite"
    shutil.copyfile(inv["path"], copied)
    inv["path"] = str(copied)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    # Output cannot be inside the inventory directory, so use a sibling fixture dir.
    inventory_dir = tmp_path / "source-db"
    inventory_dir.mkdir()
    copied.rename(inventory_dir / copied.name)
    copied = inventory_dir / copied.name
    inv["path"] = str(copied)
    manifest_path.write_text(json.dumps(manifest))
    data = load(tmp_path / "review-output", BATCH, manifest_path)
    with sqlite3.connect(copied) as conn:
        conn.execute("update features set content_sha256=? where file_id=?", ("0" * 64, manifest["groups"][0]["frames"][0]["file_id"]))
    assert "error" in data.apply_action({"batch_id": data.batch_id, "group_id": 1, "action": "accept"})
    assert not data.state_path.exists()
