"""Learning loop gates use development-only real inputs or synthetic copies."""

import copy
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from src import learning_bundle as b
from src import learning_shadow as s
from src.cloud_review import CloudReviewData
from src.learning_io import atomic_json, canonical, file_sha, read_json, sha256
from src.readonly_features import decode_dino, readonly_inventory
from src.review_events import EventJournal, HoldoutPolicy, make_event, validate_event

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "artifacts/cloud-judge-batch/frozen.json"
BATCH = ROOT / "artifacts/cloud-judge-batch/authorized-scene-run-v3"
DEV = ["2026-04-04-G002", "2026-04-04-G004", "2026-04-04-G006", "2026-04-04-G007"]


@pytest.fixture
def inputs(tmp_path):
    if not FROZEN.exists():
        pytest.skip("private development inputs absent")
    frozen = read_json(FROZEN)
    policy = HoldoutPolicy(frozen)
    selected = copy.deepcopy(frozen)
    selected["groups"] = [g for g in frozen["groups"] if g["group"] in DEV]
    manifest = tmp_path / "development.json"
    atomic_json(manifest, selected)
    events = []
    for n, spec in enumerate(selected["groups"], 1):
        event = make_event(
            spec,
            {"status": "modified", "selected": spec["frames"][:1], "action": "modify"},
            BATCH,
            "fixture",
            f"fixture:{spec['group']}",
        )
        event["sequence"] = n
        events.append(event)
    return frozen, policy, selected, manifest, events


@pytest.fixture
def bundle(tmp_path, inputs):
    frozen, policy, _, _, events = inputs
    path = tmp_path / "bundle"
    b.export_bundle(events, frozen, BATCH, path, policy)
    return path


def test_event_schema_differences_and_ids(inputs):
    event = inputs[4][0]
    validate_event(event)
    for field in ("members", "timestamp", "session", "snapshots", "ai"):
        changed = copy.deepcopy(event)
        changed.pop(field)
        with pytest.raises(jsonschema.ValidationError):
            validate_event(changed)
    changed = copy.deepcopy(event)
    changed["human"]["added"] = []
    changed["human"]["removed"] = []
    changed["human"]["final"] = []
    with pytest.raises(ValueError):
        validate_event(changed)


def test_journal_concurrent_idempotent_and_immutable(tmp_path, inputs):
    journal = EventJournal(tmp_path / "events")
    events = inputs[4]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(journal.append, events * 2))
    before = {p.name: file_sha(p) for p in journal.root.glob("*.json")}
    for event in events:
        journal.append(event)
    assert len(journal.events()) == 4
    assert before == {p.name: file_sha(p) for p in journal.root.glob("*.json")}
    bad = copy.deepcopy(events[0])
    bad["human"]["feedback"] = True
    with pytest.raises(ValueError, match="idempotency"):
        journal.append(bad)


def test_ui_save_outbox_recovery_and_no_reload_duplicates(
    tmp_path, inputs, monkeypatch
):
    output = tmp_path / "review"
    data = CloudReviewData(output, BATCH, inputs[3])
    payload = {
        "batch_id": data.batch_id,
        "group_id": 1,
        "action": "keep_all",
        "request_id": "request-1",
    }
    append = data.journal.append

    def fail(_):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(data.journal, "append", fail)
    assert "error" in data.apply_action(payload)
    assert read_json(data.state_path)["pending_event"]
    monkeypatch.setattr(data.journal, "append", append)
    restored = CloudReviewData(output, BATCH, inputs[3])
    assert restored.apply_action(payload)["ok"]
    assert len(restored.journal.events()) == 1
    assert restored.apply_action(payload)["ok"]
    again = CloudReviewData(output, BATCH, inputs[3])
    assert len(again.journal.events()) == 1
    assert again.group(1)["review"]["status"] == "modified"


def test_two_instances_do_not_lose_state(tmp_path, inputs):
    output = tmp_path / "review"
    a = CloudReviewData(output, BATCH, inputs[3])
    c = CloudReviewData(output, BATCH, inputs[3])
    for data, gid in ((a, 1), (c, 2)):
        assert data.apply_action(
            {"batch_id": data.batch_id, "group_id": gid, "action": "keep_all"}
        )["ok"]
    assert len(read_json(a.state_path)["groups"]) == 2
    assert [e["sequence"] for e in a.journal.events()] == [1, 2]


def test_bundle_full_features_catalog_and_pair_labels(bundle, inputs):
    checked = b.validate_bundle(bundle, inputs[1])
    assert checked["groups"] == 4
    manifest = read_json(bundle / "manifest.json")
    for entry in manifest["groups"].values():
        records = read_json(bundle / entry["features"])["rows"]
        assert "quality_meta" in records[0]["columns"]
        assert "dinov2_embedding" in records[0]["columns"]
    assert not list(bundle.rglob("*.sqlite"))
    assert not (bundle / "originals").exists()
    assert read_json(bundle / "pair-labels.json")


@pytest.mark.parametrize(
    "mutation",
    [
        "checksum",
        "unlisted",
        "symlink",
        "holdout_flag",
        "holdout_identity",
        "split",
        "pair",
        "absolute",
        "catalog",
    ],
)
def test_bundle_rejects_tampering(bundle, inputs, mutation):
    manifest = read_json(bundle / "manifest.json")
    _group, entry = next(iter(manifest["groups"].items()))
    event = read_json(bundle / entry["event"])
    if mutation == "checksum":
        (bundle / entry["event"]).write_text("{}")
    elif mutation == "unlisted":
        (bundle / "secret.txt").write_text("secret")
    elif mutation == "symlink":
        (bundle / "escape").symlink_to(FROZEN)
    elif mutation == "split":
        entry["split"] = "train"
    elif mutation == "pair":
        atomic_json(bundle / "pair-labels.json", [])
        manifest["files"]["pair-labels.json"] = file_sha(bundle / "pair-labels.json")
    elif mutation == "absolute":
        manifest["hostname"] = "private"
    elif mutation == "catalog":
        catalog = read_json(bundle / "source-catalog.json")
        catalog["assets"][0]["asset_id"] = "https://temporary/url"
        atomic_json(bundle / "source-catalog.json", catalog)
        manifest["files"]["source-catalog.json"] = file_sha(
            bundle / "source-catalog.json"
        )
    else:
        if mutation == "holdout_flag":
            event["holdout"] = True
        else:
            # Membership metadata only; no holdout labels, artifacts or photos.
            held = next(
                g for g in inputs[0]["groups"] if g["group"] in inputs[1].groups
            )
            event["group"] = held["group"]
            event["holdout"] = False
        atomic_json(bundle / entry["event"], event)
        manifest["files"][entry["event"]] = file_sha(bundle / entry["event"])
    atomic_json(bundle / "manifest.json", manifest)
    with pytest.raises((ValueError, jsonschema.ValidationError)):
        b.validate_bundle(bundle, inputs[1])


def test_artifact_import_rollback_shadow_deterministic(tmp_path, bundle, inputs):
    artifact = tmp_path / "fixture"
    output = tmp_path / "output"
    value = s.fixture_artifact(bundle, artifact, inputs[1])
    first = s.import_artifact(artifact, bundle, output, inputs[1])
    result = s.shadow_context(bundle, output, DEV[0], inputs[1])
    assert result == s.shadow_context(bundle, output, DEV[0], inputs[1])
    assert 0 < len(result["context"]["exemplars"]) <= 3
    assert result["bytes"] <= 4096
    assert not result["context"]["model_called"]
    assert result["shadow_recommendation"] is None
    value["preference_profile"]["version"] = "fixture-2"
    atomic_json(artifact / "artifact.json", value)
    second = s.import_artifact(artifact, bundle, output, inputs[1])
    assert first != second
    assert s.rollback(output)["active"] == first["version"]
    assert s.rollback(output, none=True)["active"] is None
    with pytest.raises(ValueError):
        s.shadow_context(bundle, output, DEV[0], inputs[1])


@pytest.mark.parametrize(
    "mutation", ["execute", "feature", "dataset", "exemplar", "rule", "code"]
)
def test_artifact_rejects_unsafe_or_incompatible(tmp_path, bundle, inputs, mutation):
    artifact = tmp_path / "fixture"
    value = s.fixture_artifact(bundle, artifact, inputs[1])
    if mutation == "execute":
        value["permissions"] = ["delete"]
    elif mutation == "feature":
        value["feature_schema"] = "L2"
    elif mutation == "dataset":
        value["dataset_bundle_sha256"] = "a" * 64
    elif mutation == "exemplar":
        value["retrieval_exemplars"][0]["group"] = next(iter(inputs[1].groups))
    elif mutation == "rule":
        value["preference_profile"]["preferences"] = [
            {"criterion": "exec('delete')", "weight": 1}
        ]
    else:
        value["files"] = {"run.py": "a" * 64}
        (artifact / "run.py").write_text("raise Exception")
    atomic_json(artifact / "artifact.json", value)
    with pytest.raises((ValueError, jsonschema.ValidationError)):
        s.import_artifact(artifact, bundle, tmp_path / "out", inputs[1])


def test_selective_original_supplement_dry_run_and_copy(tmp_path, bundle, inputs):
    asset = read_json(bundle / "source-catalog.json")["assets"][0]
    request = {
        "schema": "original-request-v1",
        "dataset_bundle_sha256": b.validate_bundle(bundle, inputs[1])[
            "dataset_bundle_sha256"
        ],
        "items": [
            {
                k: asset[k]
                for k in ("date", "file_id", "content_sha256", "provider", "asset_id")
            }
            | {"reason": "fixture hard example"}
        ],
    }
    path = tmp_path / "request.json"
    atomic_json(path, request)
    target = tmp_path / "supplement"
    preview = b.export_original_supplement(path, bundle, inputs[0], target, inputs[1])
    assert preview["photos"] == 1 and preview["bytes"] > 0 and not target.exists()
    result = b.export_original_supplement(
        path, bundle, inputs[0], target, inputs[1], confirm=True
    )
    assert result["photos"] == 1
    assert len(list((target / "originals").iterdir())) == 1
    assert file_sha(next((target / "originals").iterdir())) == asset["content_sha256"]


def test_ro_connection_wal_and_dino(tmp_path):
    path = tmp_path / "test.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE x (id INTEGER)")
    before = file_sha(path)
    with readonly_inventory(path) as conn:
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO x VALUES (1)")
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")
    Path(str(path) + "-wal").write_bytes(b"nonempty")
    with pytest.raises(ValueError, match="WAL"), readonly_inventory(path):
        pass
    assert before == file_sha(path)
    assert np.isclose(
        np.linalg.norm(decode_dino(np.ones(768, dtype="<f2").tobytes())), 1
    )
    for blob in (
        b"",
        np.zeros(768, dtype="<f2").tobytes(),
        np.full(768, np.nan, dtype="<f2").tobytes(),
    ):
        with pytest.raises(ValueError):
            decode_dino(blob)


def test_canonical_event_id(inputs):
    event = inputs[4][0]
    assert event["event_id"] == sha256(event["idempotency_key"].encode())
    assert canonical(event) == canonical(json.loads(canonical(event)))
