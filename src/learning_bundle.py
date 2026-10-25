"""Portable review-only training bundles. Originals require a separate bounded request."""

from __future__ import annotations

import base64
import itertools
import json
import re
import shutil
from pathlib import Path

import jsonschema
from PIL import Image

from .learning_io import (
    atomic_json,
    canonical,
    file_sha,
    read_json,
    safe_path,
    sha256,
    staged_directory,
)
from .readonly_features import FEATURE_SCHEMA, feature_rows, readonly_inventory
from .review_events import ROOT, frame_set, identity, validate_event


def validate_schema(value, name):
    jsonschema.Draft202012Validator(
        read_json(ROOT / "schemas" / (name + ".schema.json"))
    ).validate(value)


USAGE_COUNTS = frozenset({
    "inputtokens", "outputtokens", "totaltokens", "cachereadinputtokens",
    "cachewriteinputtokens", "cachecreationinputtokens", "input_tokens",
    "output_tokens", "total_tokens", "cache_read_input_tokens",
    "cache_creation_input_tokens",
})


def hygiene(value):
    """Reject portable locators and secret-bearing keys; usage counts are integers."""
    if isinstance(value, dict):
        # Full DB binary feature schema: encoded bytes are data, never locators.
        if set(value) in ({"encoding", "data"}, {"encoding", "data", "bytes"}) and value["encoding"] == "base64":
            decoded = base64.b64decode(value["data"], validate=True)
            if ("bytes" in value and (type(value["bytes"]) is not int or value["bytes"] != len(decoded))) or base64.b64encode(decoded).decode() != value["data"]:
                raise ValueError("noncanonical binary feature")
            return
        for key, item in value.items():
            normalized = key.lower()
            if normalized in USAGE_COUNTS:
                if type(item) is not int or item < 0:
                    raise ValueError("usage token count must be a nonnegative integer")
            elif any(part in normalized for part in (
                "token", "secret", "api_key", "access_key", "credential",
                "password", "cookie", "authorization", "private_key",
            )) or normalized in {"hostname", "username", "download_url", "base_url"}:
                raise ValueError("private host/credential field")
            hygiene(key)
            hygiene(item)
    elif isinstance(value, list):
        for item in value:
            hygiene(item)
    elif isinstance(value, str) and (
        re.search(r"[A-Za-z][A-Za-z0-9+.-]*://", value)
        or re.search(r"(?:^|[\s\"'`(\[{=:：])(?:/|[A-Za-z]:[\\/]|\\\\)", value)
        or re.search(r"ip-\d+-\d+-\d+-\d+", value)
    ):
        raise ValueError("absolute path, URL or hostname in bundle")


def checked_copy(source, target, expected):
    if file_sha(source) != expected:
        raise ValueError("source hash mismatch")
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    if file_sha(source) != expected or file_sha(target) != expected:
        raise ValueError("copy source changed or checksum mismatch")


def pair_labels(event):
    selected = frame_set(event["human"]["final"])
    return [
        {
            "date": event["date"],
            "group": event["group"],
            "event_id": event["event_id"],
            "a": a,
            "b": b,
            "label": "both"
            if identity(a) in selected and identity(b) in selected
            else "a"
            if identity(a) in selected
            else "b"
            if identity(b) in selected
            else "neither",
        }
        for a, b in itertools.combinations(event["members"], 2)
    ]


def bundle_digest(manifest):
    return sha256(canonical(manifest))


def export_bundle(events, frozen, batch, destination, policy, start=1, end=None):
    # Check isolation before even creating staging parents/locks.
    target = Path(destination).resolve()
    protected = [Path(batch).resolve()]
    for inv in frozen["inventories"]:
        protected.append(Path(inv["path"]).resolve().parent)
        with readonly_inventory(inv["path"]) as conn:
            protected.extend(Path(row[0]).resolve().parent for row in conn.execute("SELECT path FROM files"))
    if any(target.is_relative_to(path) for path in protected):
        raise ValueError("bundle output overlaps protected input")
    with staged_directory(destination) as staging:
        return _export_bundle(events, frozen, batch, staging, policy, start, end)


def _export_bundle(events, frozen, batch, destination, policy, start=1, end=None):
    destination = Path(destination)
    if destination.exists():
        raise ValueError("bundle destination must be new")
    selected = {}
    excluded = 0
    for event in events:
        validate_event(event)
        if event["sequence"] < start or (end is not None and event["sequence"] > end):
            continue
        if event["holdout"] or policy.blocked(
            event["group"], event["date"], event["members"]
        ):
            excluded += 1
            continue
        policy.check_event(event)
        selected[event["group"]] = event
    selected = {
        g: e
        for g, e in selected.items()
        if e["human"]["status"] in ("accepted", "modified")
    }
    if not selected:
        raise ValueError("no reviewed nonholdout events")
    # Validate output isolation against all DBs and original directories, no images read.
    for inv in frozen["inventories"]:
        db_path = Path(inv["path"]).resolve()
        if destination.resolve().is_relative_to(
            db_path.parent
        ) or destination.resolve().is_relative_to(Path(batch).resolve()):
            raise ValueError("bundle output overlaps protected input")
    destination.mkdir(parents=True)
    files, groups, catalog, pairs, external_refs = {}, {}, {}, [], {}
    inventories = {i["date"]: i for i in frozen["inventories"]}

    def save(name, value):
        hygiene(value)
        path = safe_path(destination, name)
        atomic_json(path, value)
        files[name] = file_sha(path)
        return name

    for group, event in sorted(selected.items()):
        inv = inventories[event["date"]]
        if file_sha(inv["path"]) != inv["sha256"]:
            raise ValueError("inventory source hash mismatch")
        identities = dict(identity(f) for f in event["members"])
        records, sources = feature_rows(inv["path"], event["date"], identities)
        with readonly_inventory(inv["path"]) as conn:
            meta = {
                fid: dict(
                    conn.execute(
                        "SELECT width,height,mtime_ns,size_bytes FROM files WHERE id=?",
                        (fid,),
                    ).fetchone()
                )
                for fid in identities
            }
        for source in sources.values():
            if destination.resolve().is_relative_to(source.resolve().parent):
                raise ValueError("bundle output overlaps original directory")
        prefix = f"groups/{group}"
        event_path = save(prefix + "/event.json", event)
        features_path = save(
            prefix + "/features.json",
            {
                "schema": FEATURE_SCHEMA,
                "database_sha256": inv["sha256"],
                "rows": records,
            },
        )
        refs = [
            *event["snapshots"].values(),
            *event["ai"]["artifacts"],
            *[a["review"] for a in event["assets"]],
        ]
        if not event["assets"] or any(v is None for v in event["snapshots"].values()):
            raise ValueError("review assets and L0/L1 required")
        for ref in refs:
            source = safe_path(batch, ref["path"])
            name = ref["path"]
            if name in files:
                if files[name] != ref["sha256"]:
                    raise ValueError("conflicting source ref")
                continue
            # Receipt state may contain a local import path. Preserve its content
            # address as an external provenance ref; provider/raw/validated remain copied.
            if source.name == "state.json":
                if file_sha(source) != ref["sha256"]:
                    raise ValueError("state source hash mismatch")
                external_refs[name] = ref["sha256"]
                continue
            if source.suffix in (".json", ".txt"):
                text = source.read_text()
                hygiene(json.loads(text) if source.suffix == ".json" else text)
            checked_copy(source, safe_path(destination, name), ref["sha256"])
            files[name] = ref["sha256"]
        for asset in event["assets"]:
            f = asset["original"]
            fid, sha = identity(f)
            with Image.open(safe_path(destination, asset["review"]["path"])) as image:
                if image.format != "JPEG" or max(image.size) > 2048 or image.getexif():
                    raise ValueError("review asset must be JPEG <=2048")
            key = (event["date"], fid, sha)
            catalog[key] = {
                "provider": "local",
                "asset_id": f"{event['date']}:{fid}:{sha}",
                "google_media_item_id": None,
                "date": event["date"],
                **f,
                **meta[fid],
                "review_asset": asset["review"],
            }
        groups[group] = {
            "event": event_path,
            "features": features_path,
            "split": "development",
            "date": event["date"],
            "database_sha256": inv["sha256"],
        }
        pairs.extend(pair_labels(event))
        if file_sha(inv["path"]) != inv["sha256"]:
            raise ValueError("inventory changed during export")
    save(
        "source-catalog.json",
        {"schema": "source-catalog-v1", "assets": list(catalog.values())},
    )
    save("pair-labels.json", pairs)
    manifest = {
        "schema": "bundle-exporter-v1",
        "feature_schema": FEATURE_SCHEMA,
        "mode": "review_2048_only",
        "event_range": [start, end],
        "groups": groups,
        "files": files,
        "external_refs": external_refs,
        "provenance": {
            "split_sha256": policy.hashes,
            "frozen_sha256": policy.frozen_sha256,
            "holdout_events_excluded": excluded,
            "feature_source": "shared_readonly_full_database",
            "originals_included": False,
        },
        "photos": len(catalog),
        "unique_source_photos": len({key[2] for key in catalog}),
        "payload_bytes": sum(
            safe_path(destination, name).stat().st_size for name in files
        ),
    }
    atomic_json(destination / "manifest.json", manifest)
    return validate_bundle(destination, policy)


def validate_bundle(root, policy):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    validate_schema(manifest, "bundle-exporter-v1")
    if (
        manifest["schema"] != "bundle-exporter-v1"
        or manifest["feature_schema"] != FEATURE_SCHEMA
        or manifest["mode"] != "review_2048_only"
    ):
        raise ValueError("bundle schema/mode incompatible")
    hygiene(manifest)
    external = manifest.get("external_refs", {})
    if any(name not in {f"{g}/state.json" for g in manifest["groups"]} or not re.fullmatch(r"[0-9a-f]{64}", value) for name, value in external.items()):
        raise ValueError("only source-state content addresses may be external")
    if (
        manifest["provenance"]["split_sha256"] != policy.hashes
        or manifest["provenance"].get("frozen_sha256") != policy.frozen_sha256
        or manifest["provenance"]["originals_included"] is not False
        or manifest["provenance"]["feature_source"] != "shared_readonly_full_database"
    ):
        raise ValueError("bundle provenance mismatch")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {"manifest.json"} or any(
        p.is_symlink() for p in root.rglob("*")
    ):
        raise ValueError("unlisted file or symlink in bundle")
    for name, expected in manifest["files"].items():
        path = safe_path(root, name)
        if file_sha(path) != expected:
            raise ValueError("bundle checksum mismatch")
        if path.suffix in (".json", ".txt"):
            text = path.read_text()
            hygiene(json.loads(text) if path.suffix == ".json" else text)
            if any(identity in text for identity in policy.shas | policy.groups):
                raise ValueError("holdout identity in bundled content")
    catalog = read_json(root / "source-catalog.json")
    validate_schema(catalog, "source-catalog-v1")
    allowed, expected_pairs = {}, []
    referenced = {"source-catalog.json", "pair-labels.json"}
    for group, entry in manifest["groups"].items():
        event = validate_event(read_json(safe_path(root, entry["event"])))
        policy.check_event(event)
        if (
            group != event["group"]
            or entry["date"] != event["date"]
            or entry["split"] != "development"
            or event["human"]["status"] not in ("accepted", "modified")
        ):
            raise ValueError("bundle split/group/review mismatch")
        referenced.update((entry["event"], entry["features"]))
        if entry["database_sha256"] != policy.inventories[event["date"]]["sha256"]:
            raise ValueError("trusted database provenance mismatch")
        features = read_json(safe_path(root, entry["features"]))
        if (
            features["schema"] != FEATURE_SCHEMA
            or features["database_sha256"] != entry["database_sha256"]
        ):
            raise ValueError("feature provenance mismatch")
        if {(r["date"], *identity(r["columns"])) for r in features["rows"]} != {
            (event["date"], *identity(f)) for f in event["members"]
        }:
            raise ValueError("feature identity mismatch")
        from .readonly_features import group_embedding

        required_columns = {"file_id", "content_sha256", "phash", "dinov2_embedding", "quality_score", "quality_meta", "face_count", "faces_json", "status", "local_quality_embedding"}
        if any(not required_columns <= r["columns"].keys() for r in features["rows"]):
            raise ValueError("full DB feature columns required; L2 projection prohibited")
        group_embedding(features["rows"])
        for layer in ("L0", "L1"):
            ref = event["snapshots"][layer]
            if not ref or read_json(safe_path(root, ref["path"])).get("layer") != layer:
                raise ValueError("snapshot layer mismatch")
        for ref in [
            *event["snapshots"].values(),
            *event["ai"]["artifacts"],
            *[a["review"] for a in event["assets"]],
        ]:
            if (
                ref is None
                or (manifest["files"] | manifest.get("external_refs", {})).get(
                    ref["path"]
                )
                != ref["sha256"]
            ):
                raise ValueError("event source hash mismatch")
            referenced.add(ref["path"])
        for asset in event["assets"]:
            allowed[(event["date"], *identity(asset["original"]))] = asset["review"]
        expected_pairs.extend(pair_labels(event))
    if referenced != set(manifest["files"]) | set(external):
        raise ValueError("unreferenced payload in bundle")
    keys = set()
    for asset in catalog["assets"]:
        key = (asset["date"], *identity(asset))
        if (
            key in keys
            or key not in allowed
            or asset["review_asset"] != allowed[key]
            or asset["content_sha256"] in policy.shas
        ):
            raise ValueError("catalog identity/holdout mismatch")
        keys.add(key)
        if (
            "://" in asset["asset_id"]
            or asset["google_media_item_id"]
            and "://" in asset["google_media_item_id"]
        ):
            raise ValueError("temporary URL is not a stable asset ID")
        with Image.open(safe_path(root, asset["review_asset"]["path"])) as image:
            if image.format != "JPEG" or max(image.size) > 2048 or image.getexif():
                raise ValueError("invalid review asset")
    if (
        keys != set(allowed)
        or manifest["photos"] != len(keys)
        or manifest["unique_source_photos"] != len({k[2] for k in keys})
    ):
        raise ValueError("catalog count mismatch")
    if read_json(root / "pair-labels.json") != expected_pairs:
        raise ValueError("derived pair label mismatch")
    size = sum(safe_path(root, n).stat().st_size for n in manifest["files"])
    if size != manifest["payload_bytes"]:
        raise ValueError("bundle byte count mismatch")
    return {
        "valid": True,
        "dataset_bundle_sha256": bundle_digest(manifest),
        "groups": len(manifest["groups"]),
        "photos": len(keys),
        "unique_source_photos": manifest["unique_source_photos"],
        "payload_bytes": size,
        "total_bytes": size + (root / "manifest.json").stat().st_size,
        "mode": manifest["mode"],
    }


def export_original_supplement(
    request_path, bundle, frozen, destination, policy, confirm=False
):
    validation = validate_bundle(bundle, policy)
    request = read_json(request_path)
    validate_original_request(request, bundle, validation)
    inventories = {i["date"]: i for i in frozen["inventories"]}
    plan = []
    for item in request["items"]:
        if item["provider"] != "local":
            raise ValueError(
                "Google fetch not implemented; M2 API/Picker authorization required"
            )
        inv = inventories[item["date"]]
        if file_sha(inv["path"]) != inv["sha256"]:
            raise ValueError("inventory changed")
        _, sources = feature_rows(
            inv["path"], item["date"], {item["file_id"]: item["content_sha256"]}
        )
        source = sources[item["file_id"]]
        plan.append((item, source))
    summary = {
        "photos": len(plan),
        "bytes": sum(s.stat().st_size for _, s in plan),
        "mode": "selective_original_supplement",
        "warning": "Originals may contain EXIF/location and incur upload/storage costs",
        "confirmed": confirm,
    }
    if not confirm:
        return summary
    destination = Path(destination)
    if destination.exists():
        raise ValueError("supplement destination must be new")
    protected = [
        Path(bundle).resolve(),
        *[Path(i["path"]).resolve().parent for i in frozen["inventories"]],
        *[s.resolve().parent for _, s in plan],
    ]
    if any(destination.resolve().is_relative_to(p) for p in protected):
        raise ValueError("supplement overlaps protected source")
    with staged_directory(destination) as staging:
        destination = staging
        destination.mkdir(parents=True)
        files = {}
        for item, source in plan:
            name = (
                f"originals/{item['date']}-{item['file_id']}-{item['content_sha256']}.bin"
            )
            checked_copy(source, safe_path(destination, name), item["content_sha256"])
            files[name] = item["content_sha256"]
        atomic_json(
            destination / "manifest.json",
            {
                "schema": "original-supplement-v1",
                "dataset_bundle_sha256": validation["dataset_bundle_sha256"],
                "request": request,
                "files": files,
                **summary,
            },
        )
    return summary


def validate_original_request(request, bundle, validation):
    """Validate attachment identity before import, without fetching any originals."""
    validate_schema(request, "original-request-v1")
    hygiene(request)
    if request["dataset_bundle_sha256"] != validation["dataset_bundle_sha256"]:
        raise ValueError("original request bundle mismatch")
    catalog = {
        (a["date"], *identity(a)): a
        for a in read_json(Path(bundle) / "source-catalog.json")["assets"]
    }
    seen = set()
    for item in request["items"]:
        key = (item["date"], *identity(item))
        asset = catalog.get(key)
        if (key in seen or not asset or asset["asset_id"] != item["asset_id"]
                or asset["provider"] != item["provider"]):
            raise ValueError("original request not in bundle allowlist")
        seen.add(key)
