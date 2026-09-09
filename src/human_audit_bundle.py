"""Local-only cached-thumbnail export and CLI for human audit bundles."""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path

from PIL import Image

from .human_audit import (
    canonical, development_scope, digest, labels_v2, load_json, make_plan, parse_labels, read_labels, report, seal,
    validate_bundle, validate_labels,
)
from .human_audit_migration import (
    CURRENT_BUNDLE, authorized_note_migration, bind_migration_manifest,
)


def safe_root(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("cache/output symlink forbidden")
    if not path.is_dir():
        raise ValueError("cache root must be an existing explicit directory")
    return path.resolve()


def cached_thumbnail(root, member):
    """Only numeric JPEG cache entries; never consult source paths or DBs."""
    path = root / f"{member}.jpg"
    if path.is_symlink():
        raise ValueError("thumbnail symlink forbidden")
    if not path.exists():
        return None
    if not path.is_file() or path.stat().st_size > 2_000_000:
        raise ValueError("invalid/oversized cached thumbnail")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as handle:
        data = handle.read(2_000_001)
    if len(data) > 2_000_000:
        raise ValueError("oversized cached thumbnail")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "JPEG" or max(image.size) > 2048:
                raise ValueError("expected bounded JPEG thumbnail")
            image.load()
            # Re-encode pixels only: EXIF/comments and embedded data never leave cache.
            clean = io.BytesIO()
            image.convert("RGB").save(clean, "JPEG", quality=88)
            return clean.getvalue()
    except OSError:
        return None


def export_bundle(*, groups_path, audit_path, scopes_path, provenance_path,
                  cache_roots, output, seed, fixture_path=None):
    scopes = development_scope(load_json(scopes_path))
    # Validate authorization and roots before any group input or media reads.
    if set(cache_roots) != set(scopes):
        raise ValueError("exactly one explicit cache root per authorized dataset required")
    roots = {key: safe_root(value) for key, value in cache_roots.items()}
    output = Path(output).absolute()
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("output symlink forbidden")
    if output.exists():
        raise ValueError("output must be a new directory; bundles are immutable")
    for root in roots.values():
        if output == root or root in output.parents or output in root.parents:
            raise ValueError("output must be separate from thumbnail caches")
    plan = make_plan(load_json(groups_path)["groups"], load_json(audit_path), scopes,
                     load_json(provenance_path), seed,
                     load_json(fixture_path) if fixture_path else None)
    media = {}
    for task in plan["tasks"]:
        task["media"] = []
        for member, alias in zip(task["member_ids"], task["aliases"], strict=True):
            data = cached_thumbnail(roots[task["dataset"]], member)
            name = f'thumbs/{task["task_id"]}-{alias}.jpg'
            if data is not None:
                media[name] = data
            task["media"].append({"alias": alias, "file": name if data is not None else None,
                                  "sha256": hashlib.sha256(data).hexdigest() if data is not None else None})
        task["media_complete"] = all(m["file"] is not None for m in task["media"])
    return publish_bundle(plan, media, output)


def publish_bundle(plan, media, output, initial_labels=None):
    """Seal the exact UI (including seed labels) before publishing a new directory."""
    plan["label_schema_version"] = 2
    public = {"schema_version": 2, "bundle_id": "0" * 64,
              "legacy_bundle_id": plan.get("legacy_bundle_id"),
              "initial_labels": initial_labels or [],
              "migration_records": plan.get("note_migration", {}).get("records", {}), "tasks": [
        {"task_id": t["task_id"], "media_complete": t["media_complete"],
         "members": [{"alias": m["alias"], "file": m["file"]} for m in t["media"]]}
        for t in plan["tasks"]]}
    template = Path(__file__).with_name("human_audit_ui.html").read_text(encoding="utf-8")
    page = template.replace("__TASK_DATA__", canonical(public).replace("<", "\\u003c"))
    plan["reviewer_content_sha256"] = hashlib.sha256(page.encode()).hexdigest()
    bundle = seal(plan)
    page = page.replace('"bundle_id":"' + "0" * 64 + '"',
                        '"bundle_id":"' + bundle["bundle_id"] + '"')
    # All validation and cache reads finish before publishing anything.
    output.mkdir(parents=True, exist_ok=False)
    reviewer = output / "reviewer"
    (reviewer / "thumbs").mkdir(parents=True)
    for name, data in media.items():
        (reviewer / name).write_bytes(data)
    (reviewer / "index.html").write_text(page, encoding="utf-8")
    (output / "manifest.json").write_text(canonical(bundle) + "\n", encoding="utf-8")
    (output / "empty-labels.jsonl").write_text("", encoding="utf-8")
    return bundle


def verify_media(bundle, directory):
    """Report replay rejects changed/missing cache exports, not just stale labels."""
    validate_bundle(bundle)
    directory = safe_root(directory)
    page_path = directory / "index.html"
    if page_path.is_symlink():
        raise ValueError("reviewer page symlink forbidden")
    page = page_path.read_text(encoding="utf-8")
    marker = '"bundle_id":"' + bundle["bundle_id"] + '"'
    if page.count(marker) != 1:
        raise ValueError("reviewer bundle marker mismatch")
    normalized = page.replace(marker, '"bundle_id":"' + "0" * 64 + '"')
    if hashlib.sha256(normalized.encode()).hexdigest() != bundle["reviewer_content_sha256"]:
        raise ValueError("reviewer page fingerprint mismatch")
    if "legacy_labels_sha256" in bundle:
        source_copy = directory.parent / "imported-v1-labels.jsonl"
        if source_copy.is_symlink() or not source_copy.is_file():
            raise ValueError("original v1 label artifact missing or symlink")
        if hashlib.sha256(source_copy.read_bytes()).hexdigest() != bundle["legacy_labels_sha256"]:
            raise ValueError("original v1 label artifact fingerprint mismatch")
    if "note_migration" in bundle:
        migration_path = directory.parent / "migration-manifest.json"
        if migration_path.is_symlink():
            raise ValueError("migration manifest symlink forbidden")
        manifest = load_json(migration_path)
        if (manifest["source_jsonl_sha256"] != bundle["legacy_labels_sha256"]
                or manifest["target_bundle_id"] != bundle["bundle_id"]
                or any(r["v2_result"]["bundle_id"] != bundle["bundle_id"] for r in manifest["records"])):
            raise ValueError("migration manifest bundle mismatch")
        normalized_manifest = bind_migration_manifest(manifest, CURRENT_BUNDLE)
        if digest(normalized_manifest) != bundle["note_migration"]["manifest_sha256"]:
            raise ValueError("migration manifest fingerprint mismatch")
    for task in bundle["tasks"]:
        for media in task["media"]:
            if media["file"] is None:
                continue
            expected = f'thumbs/{task["task_id"]}-{media["alias"]}.jpg'
            if media["file"] != expected:
                raise ValueError("invalid media reference")
            path = directory / expected
            if any(p.is_symlink() for p in (path, path.parent)):
                raise ValueError("exported media symlink forbidden")
            if hashlib.sha256(path.read_bytes()).hexdigest() != media["sha256"]:
                raise ValueError("exported media fingerprint mismatch")


def upgrade_bundle(source, labels_path, output, *, note_authorization=None):
    """Re-present verified v1 bytes and labels; never resample or open source caches.

    A sealed lineage ID authorizes v1 imports only from this exact source bundle.
    V2 exports bind to the new bundle. Note rules require explicit authorization.
    """
    source = safe_root(source)
    output = Path(output).absolute()
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("output symlink forbidden")
    if output.exists() or source in output.parents or output in source.parents:
        raise ValueError("output must be a new separate directory; bundles are immutable")
    labels_path = Path(labels_path).absolute()
    if any(p.is_symlink() for p in (labels_path, *labels_path.parents)) or (source / "manifest.json").is_symlink():
        raise ValueError("migration input symlink forbidden")
    old = validate_bundle(load_json(source / "manifest.json"))
    if old.get("legacy_bundle_id") or old.get("label_schema_version", 1) != 1:
        raise ValueError("upgrade requires an original v1 bundle, not a v2/migrated bundle")
    verify_media(old, source / "reviewer")
    labels_bytes = labels_path.read_bytes()
    labels = parse_labels(labels_bytes.decode("utf-8"))
    validate_labels(old, labels)
    if any(x["schema_version"] != 1 for x in labels):
        raise ValueError("upgrade requires legacy v1 labels")
    plan = copy.deepcopy(old)
    del plan["bundle_id"]
    plan["legacy_bundle_id"] = old["bundle_id"]
    plan["legacy_labels_sha256"] = hashlib.sha256(labels_bytes).hexdigest()
    initial_labels, migration_manifest = labels, None
    if note_authorization is not None:
        initial_labels, migration_manifest, plan["note_migration"] = authorized_note_migration(
            old, labels_bytes, note_authorization)
    media = {}
    for task in plan["tasks"]:
        for item in task["media"]:
            if item["file"] is not None:
                data = (source / "reviewer" / item["file"]).read_bytes()
                # Verify the exact bytes we will publish, not an earlier read.
                if hashlib.sha256(data).hexdigest() != item["sha256"]:
                    raise ValueError("exported media fingerprint mismatch")
                media[item["file"]] = data
    new = publish_bundle(plan, media, output, initial_labels=initial_labels)
    (output / "imported-v1-labels.jsonl").write_bytes(labels_bytes)
    if migration_manifest is not None:
        bound_manifest = bind_migration_manifest(migration_manifest, new["bundle_id"])
        (output / "migration-manifest.json").write_text(canonical(bound_manifest) + "\n", encoding="utf-8")
        labels = [r["v2_result"] for r in bound_manifest["records"]]
    converted = labels_v2(new, labels)
    (output / "labels-v2.jsonl").write_text("".join(canonical(x) + "\n" for x in converted), encoding="utf-8")
    (output / "imported-report.json").write_text(canonical(report(new, labels)) + "\n", encoding="utf-8")
    verify_media(new, output / "reviewer")
    return new


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("export")
    for flag in ("groups", "audit", "scopes", "provenance", "output", "seed"):
        build.add_argument(f"--{flag}", required=True)
    build.add_argument("--cache", nargs=2, action="append", required=True, metavar=("DATASET", "THUMBS"))
    build.add_argument("--fixture")
    check = commands.add_parser("report")
    check.add_argument("--bundle", required=True)
    check.add_argument("--labels", required=True)
    check.add_argument("--output", required=True)
    upgrade = commands.add_parser("upgrade")
    for flag in ("bundle", "labels", "output"):
        upgrade.add_argument(f"--{flag}", required=True)
    migrate = commands.add_parser("migrate-notes", help="explicitly authorized deterministic v1 note migration")
    for flag in ("bundle", "labels", "output", "authorization-id"):
        migrate.add_argument(f"--{flag}", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            if len(dict(args.cache)) != len(args.cache):
                raise ValueError("duplicate cache dataset")
            bundle = export_bundle(groups_path=args.groups, audit_path=args.audit,
                                   scopes_path=args.scopes, provenance_path=args.provenance,
                                   cache_roots=dict(args.cache), output=args.output, seed=args.seed,
                                   fixture_path=args.fixture)
            print(f'Created {len(bundle["tasks"])} pending tasks; bundle {bundle["bundle_id"]}')
        elif args.command in {"upgrade", "migrate-notes"}:
            authorization = args.authorization_id if args.command == "migrate-notes" else None
            bundle = upgrade_bundle(args.bundle, args.labels, args.output, note_authorization=authorization)
            if "note_migration" in bundle:
                print(f'Authorized note migration: {canonical(bundle["note_migration"]["counts"])}')
            print(f'Created v2 reviewer: {len(bundle["tasks"])} tasks with editable legacy records; bundle {bundle["bundle_id"]}')
        else:
            path = Path(args.bundle)
            bundle = validate_bundle(load_json(path / "manifest.json"))
            verify_media(bundle, path / "reviewer")
            value = report(bundle, read_labels(args.labels))
            # Exclusive creation: never overwrite labels, inputs, or earlier reports.
            with Path(args.output).open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            print(f'{value["status"]}: risk={value["weighted_error_rate"]}, upper95={value["upper_95"]}')
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
