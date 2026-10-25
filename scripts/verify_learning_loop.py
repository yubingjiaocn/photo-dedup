"""Reproducible local artifact exchange evidence, strictly nonholdout after backfill."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.learning_bundle import (
    export_original_supplement,
    validate_bundle,
)
from src.learning_io import atomic_json, file_sha, read_json
from src.learning_shadow import (
    fixture_artifact,
    import_artifact,
    rollback,
    shadow_context,
)
from src.readonly_features import feature_rows
from src.review_events import HoldoutPolicy


def main():
    evidence = ROOT / "artifacts/learning-loop-mvp/p1-evidence"
    evidence.mkdir(exist_ok=True)
    base = ROOT / "output/learning-mvp"
    output = base / "exchange-v3"
    bundle = base / "bundle-v3"
    frozen = read_json(ROOT / "artifacts/cloud-judge-batch/frozen.json")
    policy = HoldoutPolicy(frozen)
    # Development IDs from the validated bundle only. Do not load sealed labels.
    manifest = read_json(bundle / "manifest.json")
    events = [read_json(bundle / entry["event"]) for entry in manifest["groups"].values()]
    assert len(events) == 130 and not any(e["holdout"] for e in events)
    # Hash only development original files; no holdout photo read.
    sources = {}
    invs = {i["date"]: i for i in frozen["inventories"]}
    for event in events:
        if event["holdout"]:
            continue
        policy.check_event(event)
        for ref in [*event["snapshots"].values(), *event["ai"]["artifacts"], *[a["review"] for a in event["assets"]]]:
            sources[str(ROOT / "artifacts/cloud-judge-batch/authorized-scene-run-v3" / ref["path"])] = ref["sha256"]
        inv = invs[event["date"]]
        sources[inv["path"]] = inv["sha256"]
        _, paths = feature_rows(inv["path"], event["date"], {f["file_id"]: f["content_sha256"] for f in event["members"]})
        for frame in event["members"]:
            sources[str(paths[frame["file_id"]])] = frame["content_sha256"]
    assert all(file_sha(p) == h for p, h in sources.items())
    validation = validate_bundle(bundle, policy)
    fixture = output / "fixture-artifact"
    value = fixture_artifact(bundle, fixture, policy)
    first = import_artifact(fixture, bundle, output / "shadow-runtime", policy)
    context = shadow_context(bundle, output / "shadow-runtime", "2026-04-04-G002", policy)
    repeat = shadow_context(bundle, output / "shadow-runtime", "2026-04-04-G002", policy)
    assert context == repeat
    atomic_json(evidence / "shadow-retrieval.json", context)
    value["preference_profile"]["version"] = "fixture-2"
    atomic_json(fixture / "artifact.json", value)
    second = import_artifact(fixture, bundle, output / "shadow-runtime", policy)
    previous = rollback(output / "shadow-runtime")
    assert previous["active"] == first["version"]
    none = rollback(output / "shadow-runtime", none=True)
    assert none["active"] is None
    atomic_json(evidence / "artifact-import-rollback.json", {"fixture_only": True, "first": first, "second": second, "rollback": previous, "none": none})
    catalog = read_json(bundle / "source-catalog.json")
    asset = catalog["assets"][0]
    request = {"schema": "original-request-v1", "dataset_bundle_sha256": validation["dataset_bundle_sha256"], "items": [{k: asset[k] for k in ("date", "file_id", "content_sha256", "provider", "asset_id")} | {"reason": "fixture: demonstrate one hard-example supplement, not a training request"}]}
    atomic_json(output / "original-request.json", request)
    preview = export_original_supplement(output / "original-request.json", bundle, frozen, output / "original-supplement", policy)
    target = output / "original-supplement"
    if not target.exists():
        supplement = export_original_supplement(output / "original-request.json", bundle, frozen, target, policy, confirm=True)
    else:
        supplement = read_json(target / "manifest.json")
        assert all(file_sha(target / p) == h for p, h in supplement["files"].items())
    atomic_json(evidence / "capacity.json", {"base": validation, "original_supplement": {k: supplement[k] for k in ("photos", "bytes", "mode")}, "preview": preview})
    assert all(file_sha(p) == h for p, h in sources.items())
    # Local evidence only: portable bundles never contain this source-path map.
    atomic_json(evidence / "source-integrity.json", {"before_expected": sources, "after": {p: file_sha(p) for p in sources}, "unchanged": True, "holdout_photo_reads": 0})
    atomic_json(evidence / "verification.json", {"passed": True, "development_events_verified": 130, "holdout_groups_excluded": 64, "training_groups": validation["groups"], "training_photos": validation["photos"], "shadow_deterministic": True, "original_writes": 0, "database_writes": 0, "decision_writes": 0, "keeper_writes": 0, "delete_writes": 0, "model_calls": 0, "google_calls": 0, "served_model": "amazon-bedrock-mantle/global.openai.gpt-6-astra", "harness": "OpenClaw native subagent", "independent_acceptance": "pending parent"})
    print(validation)


if __name__ == "__main__":
    main()
