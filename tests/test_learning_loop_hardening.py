"""Regression checks for import attachments, journal boundaries and frozen policy."""
import copy

import pytest

from src import learning_bundle as b
from src import learning_shadow as s
from src.cloud_review import CloudReviewData
from src.learning_io import atomic_json, file_sha, read_json
from src.review_events import EventJournal
from tests.test_learning_loop import BATCH
from tests.test_learning_loop import bundle as bundle_fixture
from tests.test_learning_loop import inputs as inputs_fixture

bundle = bundle_fixture
inputs = inputs_fixture


def test_import_preserves_optional_request_bytes(tmp_path, bundle, inputs):
    artifact = tmp_path / "fixture"
    value = s.fixture_artifact(bundle, artifact, inputs[1])
    asset = read_json(bundle / "source-catalog.json")["assets"][0]
    request = {"schema": "original-request-v1", "dataset_bundle_sha256": value["dataset_bundle_sha256"],
               "items": [{k: asset[k] for k in ("date", "file_id", "content_sha256", "provider", "asset_id")} | {"reason": "fixture"}]}
    import json
    (artifact / "original-request.json").write_text(json.dumps(request, indent=4))
    expected = file_sha(artifact / "original-request.json")
    value["files"] = {"original-request.json": expected}
    value["original_request"] = {"path": "original-request.json", "sha256": expected}
    atomic_json(artifact / "artifact.json", value)
    output = tmp_path / "out"
    result = s.import_artifact(artifact, bundle, output, inputs[1])
    imported = output / "learners/versions" / result["version"]
    assert file_sha(imported / "original-request.json") == expected
    s.validate_artifact(imported, bundle, inputs[1])


def test_journal_rejects_replaced_directory_symlink(tmp_path, inputs):
    root = tmp_path / "journal"
    journal = EventJournal(root)
    other = tmp_path / "other"
    other.mkdir()
    root.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        journal.append(inputs[4][0])
    assert not list(other.iterdir())


def test_holdout_sha_alias_rejected_without_holdout_label_access(inputs):
    policy = inputs[1]
    event = copy.deepcopy(inputs[4][0])
    event["members"][0]["content_sha256"] = next(iter(policy.shas))
    assert policy.blocked(event["group"], event["date"], event["members"])


def test_unreferenced_payload_rejected_even_with_checksum(bundle, inputs):
    manifest = read_json(bundle / "manifest.json")
    (bundle / "extra.bin").write_bytes(b"not an authorized training input")
    manifest["files"]["extra.bin"] = file_sha(bundle / "extra.bin")
    manifest["payload_bytes"] += (bundle / "extra.bin").stat().st_size
    atomic_json(bundle / "manifest.json", manifest)
    with pytest.raises(ValueError, match="unreferenced"):
        b.validate_bundle(bundle, inputs[1])


def test_save_idempotency_conflict_rejected(tmp_path, inputs):
    data = CloudReviewData(tmp_path / "output", BATCH, inputs[3])
    payload = {"batch_id": data.batch_id, "group_id": 1, "action": "keep_all", "request_id": "same-request"}
    assert data.apply_action(payload)["ok"]
    assert "error" in data.apply_action({**payload, "group_id": 2})
    assert len(data.journal.events()) == 1


@pytest.mark.parametrize("value", [
    "/opt", "/root/.aws", "/data", "/srv", "file:///root/.aws", "s3://bucket/key",
    "http://example.test", "custom+proto://locator", "(/home/private/file)",
    "'/home/private/file'", '"/opt/private"', r"C:\private\file", r"\\host\share",
    {"api_key": "x"}, {"aws_secret_access_key": "x"}, {"session_token": "x"},
    {"Cookie": "x"}, {"myPASSWORDfield": "x"}, {"PRIVATE_KEY_data": "x"},
    {"inputTokens": "secret"}, {"token_count": 1},
])
def test_hygiene_malicious_values(value):
    with pytest.raises(ValueError):
        b.hygiene(value)


def test_provider_usage_narrow_allowlist():
    b.hygiene({"usage": {"inputTokens": 12, "outputTokens": 4, "totalTokens": 16,
                        "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0}})


def test_default_holdout_refused_before_any_batch_read(tmp_path, monkeypatch):
    from tests.test_learning_loop import FROZEN
    def forbidden(*args):
        pytest.fail("batch was read before holdout rejection")
    monkeypatch.setattr(CloudReviewData, "_read", forbidden)
    with pytest.raises(ValueError, match="holdout manifest refused"):
        CloudReviewData(tmp_path / "out", BATCH, FROZEN)


def test_live_holdout_save_is_sealed_without_holdout_reads(tmp_path, inputs, monkeypatch):
    # Only development assets are loaded. Assign a trusted blocked alias to the
    # in-memory fixture; never copy, open, or decode any real holdout artifact.
    data = CloudReviewData(tmp_path / "out", BATCH, inputs[3])
    group = data._groups[1]
    group["alias"] = "2026-04-04-G001"
    group["_spec"] = {**group["_spec"], "group": group["alias"]}
    import src.review_events as events
    real_read_json = events.read_json
    schema_path = events.ROOT / "schemas/review-event-log-v1.schema.json"
    schema_reads = []

    def schema_only(path):
        if path != schema_path:
            pytest.fail("sealed AI/artifact JSON read")
        schema_reads.append(path)
        return real_read_json(path)

    # Exercise cold validator initialization even when earlier tests warmed it.
    events.event_validator.cache_clear()
    try:
        with monkeypatch.context() as guard:
            guard.setattr(events, "artifact_ref", lambda *a: pytest.fail("sealed artifact read"))
            guard.setattr(events, "read_json", schema_only)
            result = data.apply_action({"batch_id": data.batch_id, "group_id": 1, "action": "keep_all"})
            assert result["ok"]
            assert schema_reads == [schema_path]
        event = data.journal.events()[0]
        assert event["holdout"] and not event["assets"]
        assert event["snapshots"] == {"L0": None, "L1": None}
        assert event["ai"]["decision"] == "unavailable_sealed"
        assert all(event["ai"][k] is None for k in ("prompt_sha256", "model", "contract", "receipt_ref"))
        assert not any(event["ai"][k] for k in ("recommended", "reasons", "evidence_refs", "artifacts"))
    finally:
        events.event_validator.cache_clear()


def test_crash_then_direct_export_recovers_pending(tmp_path, inputs, monkeypatch):
    data = CloudReviewData(tmp_path / "out", BATCH, inputs[3])
    monkeypatch.setattr(data.journal, "append", lambda e: (_ for _ in ()).throw(OSError("crash")))
    assert "error" in data.apply_action({"batch_id": data.batch_id, "group_id": 1, "action": "keep_all"})
    journal = EventJournal(data.journal.root)
    result = b.export_bundle(journal.export_events(), inputs[0], BATCH, tmp_path / "export", inputs[1])
    assert result["groups"] == 1
    assert read_json(data.state_path)["pending_event"] is None
    assert len(journal.export_events()) == 1


def test_export_refuses_failed_pending_recovery(tmp_path, inputs, monkeypatch):
    journal = EventJournal(tmp_path / "out" / "review-events-v1")
    atomic_json(journal.root.parent / "review-state/cloud-review.json", {"pending_event": inputs[4][0]})
    monkeypatch.setattr(journal, "append", lambda e: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError, match="disk"):
        journal.export_events()


def test_reload_recovers_pending_without_save(tmp_path, inputs, monkeypatch):
    data = CloudReviewData(tmp_path / "out", BATCH, inputs[3])
    monkeypatch.setattr(data.journal, "append", lambda e: (_ for _ in ()).throw(OSError("crash")))
    assert "error" in data.apply_action({"batch_id": data.batch_id, "group_id": 1, "action": "keep_all"})
    loaded = CloudReviewData(tmp_path / "out", BATCH, inputs[3])
    assert len(loaded.journal.events()) == 1
    assert read_json(loaded.state_path)["pending_event"] is None


@pytest.mark.parametrize("field,value", [("file_id", 999999), ("content_sha256", "f" * 64), ("asset_id", "outside")])
def test_import_rejects_outside_original_attachment(tmp_path, bundle, inputs, field, value):
    artifact = tmp_path / "fixture"
    doc = s.fixture_artifact(bundle, artifact, inputs[1])
    asset = read_json(bundle / "source-catalog.json")["assets"][0]
    item = {k: asset[k] for k in ("date", "file_id", "content_sha256", "provider", "asset_id")} | {"reason": "fixture", field: value}
    request = {"schema": "original-request-v1", "dataset_bundle_sha256": doc["dataset_bundle_sha256"], "items": [item]}
    atomic_json(artifact / "original-request.json", request)
    sha = file_sha(artifact / "original-request.json")
    doc.update(files={"original-request.json": sha}, original_request={"path": "original-request.json", "sha256": sha})
    atomic_json(artifact / "artifact.json", doc)
    with pytest.raises(ValueError, match="allowlist"):
        s.import_artifact(artifact, bundle, tmp_path / "out", inputs[1])
    assert not (tmp_path / "out").exists()


def test_export_failure_cleans_staging_and_never_publishes(tmp_path, inputs, monkeypatch):
    monkeypatch.setattr(b, "checked_copy", lambda *a: (_ for _ in ()).throw(OSError("copy failure")))
    with pytest.raises(OSError, match="copy failure"):
        b.export_bundle(inputs[4], inputs[0], BATCH, tmp_path / "target", inputs[1])
    assert not (tmp_path / "target").exists()
    assert not list(tmp_path.glob(".target.staging-*"))


def test_changed_registry_requires_new_trust_and_invalidates_old_bundle(tmp_path, bundle, inputs):
    from src.learning_io import canonical, sha256
    from src.review_events import HoldoutPolicy
    changed = copy.deepcopy(inputs[0])
    changed["dataset_note"] = "different dataset"
    with pytest.raises(ValueError, match="fingerprint"):
        HoldoutPolicy(changed)
    trust = tmp_path / "trust.json"
    atomic_json(trust, {"frozen_sha256": sha256(canonical(changed)), "split_sha256": inputs[1].hashes})
    policy = HoldoutPolicy(changed, trust_path=trust)
    with pytest.raises(ValueError, match="provenance"):
        b.validate_bundle(bundle, policy)


def test_original_attachment_wrong_bundle_sha_rejected(tmp_path, bundle, inputs):
    artifact = tmp_path / "fixture"
    doc = s.fixture_artifact(bundle, artifact, inputs[1])
    asset = read_json(bundle / "source-catalog.json")["assets"][0]
    request = {"schema": "original-request-v1", "dataset_bundle_sha256": "0" * 64,
               "items": [{k: asset[k] for k in ("date", "file_id", "content_sha256", "provider", "asset_id")} | {"reason": "fixture"}]}
    atomic_json(artifact / "original-request.json", request)
    sha = file_sha(artifact / "original-request.json")
    doc.update(files={"original-request.json": sha}, original_request={"path": "original-request.json", "sha256": sha})
    atomic_json(artifact / "artifact.json", doc)
    with pytest.raises(ValueError, match="bundle mismatch"):
        s.import_artifact(artifact, bundle, tmp_path / "out", inputs[1])
    assert not (tmp_path / "out").exists()


def test_pending_already_appended_recovers_exactly_once(tmp_path, inputs):
    journal = EventJournal(tmp_path / "out/review-events-v1")
    event = journal.append(inputs[4][0])
    state = journal.root.parent / "review-state/cloud-review.json"
    atomic_json(state, {"pending_event": event})
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda _: EventJournal(journal.root).recover_pending(), range(3)))
    assert len(journal.events()) == 1
    assert read_json(state)["pending_event"] is None


def test_bundle_isolation_checked_before_staging_parent_creation(tmp_path, inputs, monkeypatch):
    # Synthetic protected batch, no source writes even on rejected nested target.
    batch = tmp_path / "protected"
    batch.mkdir()
    with pytest.raises(ValueError, match="protected"):
        b.export_bundle(inputs[4], inputs[0], batch, batch / "new-parent/target", inputs[1])
    assert not list(batch.iterdir())
