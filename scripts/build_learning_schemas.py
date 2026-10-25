"""Regenerate checked-in learning JSON schemas (no dataset inputs)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHA = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
TEXT = {"type": "string", "minLength": 1}


def obj(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def arr(item):
    return {"type": "array", "items": item}


def nullable(value):
    return {"anyOf": [value, {"type": "null"}]}


def save(name, schema):
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": name + ".schema.json",
        **schema,
    }
    (ROOT / "schemas" / (name + ".schema.json")).write_text(
        json.dumps(schema, indent=2) + "\n"
    )


frame = obj({"file_id": {"type": "integer", "minimum": 1}, "content_sha256": SHA})
ref = obj({"path": TEXT, "sha256": SHA})
frames = arr(frame)
event = obj(
    {
        "schema": {"const": "review-event-log-v1"},
        "event_id": SHA,
        "idempotency_key": TEXT,
        "sequence": {"type": "integer", "minimum": 1},
        "timestamp": {"type": "string", "format": "date-time"},
        "session": TEXT,
        "group": {"type": "string", "pattern": "^[A-Za-z0-9_-]+$"},
        "date": {"type": "string", "format": "date"},
        "members": {**frames, "minItems": 1},
        "holdout": {"type": "boolean"},
        "ai": obj(
            {
                "decision": TEXT,
                "recommended": frames,
                "reasons": arr({"type": "object"}),
                "evidence_refs": arr({"type": "object"}),
                "prompt_sha256": nullable(SHA),
                "model": nullable(TEXT),
                "contract": nullable(TEXT),
                "receipt_ref": nullable(ref),
                "artifacts": arr(ref),
            }
        ),
        "human": obj(
            {
                "status": {"enum": ["pending", "later", "accepted", "modified"]},
                "final": frames,
                "added": frames,
                "removed": frames,
                "replaced": {"type": "boolean"},
                "ignored_reasons": arr({"type": "integer", "minimum": 0}),
                "feedback": {"type": "boolean"},
                "action": {
                    "enum": [
                        "accept",
                        "modify",
                        "keep_all",
                        "later",
                        "feedback",
                        "backfill",
                    ]
                },
            }
        ),
        "snapshots": obj({"L0": nullable(ref), "L1": nullable(ref)}),
        "assets": arr(
            obj({"original": frame, "review": ref, "long_edge": {"const": 2048}})
        ),
        "provenance": {
            "type": "object",
            "required": ["kind"],
            "properties": {
                "kind": {"enum": ["live_review", "frozen_label_snapshot"]},
                "snapshot_sha256": SHA,
            },
            "additionalProperties": False,
        },
    }
)
event["allOf"] = [{
    "if": {"properties": {"holdout": {"const": True}}},
    "then": {"properties": {
        "ai": {"properties": {
            "decision": {"const": "unavailable_sealed"},
            **{k: {"const": None} for k in ("prompt_sha256", "model", "contract", "receipt_ref")},
            **{k: {"maxItems": 0} for k in ("recommended", "reasons", "evidence_refs", "artifacts")},
        }},
        "assets": {"maxItems": 0},
        "snapshots": {"properties": {"L0": {"const": None}, "L1": {"const": None}}},
    }},
}]
save("review-event-log-v1", event)
locator = obj(
    {
        "provider": {"enum": ["local", "google_photos"]},
        "asset_id": TEXT,
        "google_media_item_id": nullable(TEXT),
        "date": TEXT,
        "file_id": {"type": "integer", "minimum": 1},
        "content_sha256": SHA,
        "width": {"type": "integer", "minimum": 1},
        "height": {"type": "integer", "minimum": 1},
        "mtime_ns": {"type": "integer", "minimum": 0},
        "size_bytes": {"type": "integer", "minimum": 0},
        "review_asset": ref,
    }
)
save(
    "source-catalog-v1",
    obj({"schema": {"const": "source-catalog-v1"}, "assets": arr(locator)}),
)
request = obj(
    {
        "schema": {"const": "original-request-v1"},
        "dataset_bundle_sha256": SHA,
        "items": {
            **arr(
                obj(
                    {
                        "date": TEXT,
                        "file_id": {"type": "integer", "minimum": 1},
                        "content_sha256": SHA,
                        "provider": {"enum": ["local", "google_photos"]},
                        "asset_id": TEXT,
                        "reason": {"type": "string", "minLength": 1, "maxLength": 500},
                    }
                )
            ),
            "minItems": 1,
            "maxItems": 32,
        },
    }
)
save("original-request-v1", request)
profile = obj(
    {
        "version": TEXT,
        "preferences": {
            **arr(
                obj(
                    {
                        "criterion": {
                            "enum": [
                                "distinct_moments",
                                "expressions",
                                "interaction",
                                "composition",
                                "technical_quality",
                                "uncertainty",
                            ]
                        },
                        "weight": {"type": "number", "minimum": -1, "maximum": 1},
                    }
                )
            ),
            "maxItems": 12,
        },
    }
)
artifact = obj(
    {
        "schema": {"const": "learner-artifact-v1"},
        "dataset_bundle_sha256": SHA,
        "trainer": obj(
            {
                "code_sha256": SHA,
                "model": TEXT,
                "version": TEXT,
                "fixture": {"type": "boolean"},
            }
        ),
        "feature_schema": {"const": "full-db-features-v1"},
        "metrics": {
            "type": "object",
            "additionalProperties": {"type": ["number", "null"]},
        },
        "gates": obj(
            {"holdout_excluded": {"const": True}, "shadow_only": {"const": True}}
        ),
        "preference_profile": profile,
        "retrieval_exemplars": arr(obj({"event_id": SHA, "group": TEXT})),
        "files": {"type": "object", "additionalProperties": SHA},
        "original_request": nullable(ref),
    }
)
save("learner-artifact-v1", artifact)
manifest = obj({
    "schema": {"const": "bundle-exporter-v1"}, "feature_schema": {"const": "full-db-features-v1"},
    "mode": {"const": "review_2048_only"}, "event_range": {"type": "array", "minItems": 2, "maxItems": 2, "prefixItems": [{"type": "integer", "minimum": 1}, nullable({"type": "integer", "minimum": 1})]},
    "groups": {"type": "object", "minProperties": 1, "additionalProperties": obj({"event": TEXT, "features": TEXT, "split": {"const": "development"}, "date": TEXT, "database_sha256": SHA})},
    "files": {"type": "object", "additionalProperties": SHA}, "external_refs": {"type": "object", "additionalProperties": SHA},
    "provenance": obj({"frozen_sha256": SHA, "split_sha256": arr(SHA), "holdout_events_excluded": {"type": "integer", "minimum": 0}, "feature_source": {"const": "shared_readonly_full_database"}, "originals_included": {"const": False}}),
    "photos": {"type": "integer", "minimum": 1}, "unique_source_photos": {"type": "integer", "minimum": 1}, "payload_bytes": {"type": "integer", "minimum": 1},
})
save("bundle-exporter-v1", manifest)
