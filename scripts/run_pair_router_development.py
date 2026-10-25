"""Read-only 88-pair development replay. No holdout data or cloud transport.

Only explicit development file IDs are queried from frozen SQLite inventories.
The fixed policy is evaluated, not fitted, in leave-one-date-out folds.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from src import pair_exposure_resolver as resolver
from src import photo_exposure
from src.readonly_features import readonly_inventory

ROOT = Path(__file__).resolve().parents[1]
TUNING = ROOT / "artifacts/pair-router-tuning"
SOURCE = ROOT / "artifacts/cloud-judge-batch/authorized-scene-run-v3"
OUT = TUNING / "candidate"
FROZEN = ROOT / "artifacts/cloud-judge-batch/frozen.json"


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def safe_asset(case, asset):
    path = (case / asset["path"]).resolve()
    if not path.is_relative_to(case.resolve()):
        raise RuntimeError("source path escape")
    if digest(path) != asset["sha256"]:
        raise RuntimeError("source asset SHA mismatch")
    return path


def load_group(group, sources):
    case = SOURCE / group
    request_path = case / "request.json"
    sources[str(request_path)] = digest(request_path)
    request = read_json(request_path)
    context = request["context"]
    bindings = context["evidence"]["frame_bindings"]
    times = {f"I{x['frame']['file_id']}@{x['frame']['content_sha256']}": x["capture_time"]
             for x in context["temporal"]}
    frames = []
    for asset in request["images"]:
        path = safe_asset(case, asset)
        sources[str(path)] = digest(path)
        if asset["kind"] != "full_group_page":
            continue
        if len(asset["cells"]) != 1:
            raise RuntimeError("pair page must bind one source")
        ident = asset["cells"][0]["frame"]
        if ident not in bindings.values():
            raise RuntimeError("frame binding mismatch")
        with Image.open(path) as image:
            desc = photo_exposure.descriptor(image)
        frames.append({"identity": ident, "time": times[ident], "descriptor": desc,
                       "page_pixels": image.width * image.height})
    frames.sort(key=lambda f: f["identity"])
    if len(frames) != 2 or set(bindings.values()) != {f["identity"] for f in frames}:
        raise RuntimeError("development pair identity mismatch")
    evidence = {}
    for item in context["evidence"]["full_frame"]["evidence"].values():
        values = item.get("values", {})
        evidence[item["criterion"]] = {ident: values[alias] for alias, ident in bindings.items() if alias in values}
    # Hash dev old AI artifacts for integrity only; replay labels already contain baseline.
    for name in ("validated.json", "raw.json", "state.json"):
        path = case / name
        if path.exists():
            sources[str(path)] = digest(path)
    return frames, evidence


def fetch_development_rows(path, identities):
    """Single ID-bound SELECT, explicit columns, SQLite-enforced read-only."""
    ids = sorted(identities)
    if not ids or any(type(i) is not int or i <= 0 for i in ids):
        raise ValueError("invalid development ID allowlist")
    wal = Path(str(path) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise RuntimeError("frozen inventory has nonempty WAL; refuse ambiguous snapshot")
    with readonly_inventory(path) as conn:
        query = ("SELECT fe.file_id, fe.content_sha256, fe.dinov2_embedding, f.path "
                 "FROM features fe JOIN files f ON f.id=fe.file_id "
                 f"WHERE fe.file_id IN ({','.join('?' for _ in ids)})")
        rows = conn.execute(query, ids).fetchall()
    if {r[0] for r in rows} != set(ids) or len(rows) != len(ids):
        raise RuntimeError("missing or duplicate development DB identity")
    for file_id, sha, _blob, _source in rows:
        if sha != identities[file_id]:
            raise RuntimeError("development DB SHA identity mismatch")
    return rows


def aggregate(rows):
    tp = sum(len(r["predicted"] & r["human"]) for r in rows)
    missing = sum(len(r["human"] - r["predicted"]) for r in rows)
    extra = sum(len(r["predicted"] - r["human"]) for r in rows)
    precision, recall = tp / max(tp + extra, 1), tp / max(tp + missing, 1)
    exact = sum(r["predicted"] == r["human"] for r in rows)
    return {"groups": len(rows), "exact": exact, "exact_rate": exact / len(rows) if rows else None,
            "precision": precision if rows else None, "recall": recall if rows else None,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12) if rows else None,
            "missing": missing, "extra": extra}


def zero_error_bound(n):
    return 1 - 0.05 ** (1 / n) if n else None


def evaluate(rows, profile):
    plans, scored, local = [], [], []
    for row in rows:
        result = resolver.resolve(row["frames"], row["evidence"], profile)
        predicted = (row["baseline"] if result["vlm_required"] else
                     {x["file_id"] for x in result["local_selected"]})
        score = {"predicted": predicted, "human": row["human"]}
        scored.append(score)
        if not result["vlm_required"]:
            local.append(score)
        plans.append({"group": row["group"], **result, "human": sorted(row["human"]),
                      "final_replay": sorted(predicted), "page_pixels": sum(f["page_pixels"] for f in row["frames"])})
    leaks = sum(len(r["human"]) == 2 for r in local)
    errors = sum(r["human"] != r["predicted"] for r in local)
    both = sum(len(r["human"]) == 2 for r in scored)
    n, local_n = len(rows), len(local)
    vlm = n - local_n
    pair_projected = 262 * vlm / n
    total_projected = 248 + pair_projected
    metrics = {"profile": profile, "groups": n, "local_select_one": local_n,
               "local_keep_both": 0, "local_errors": errors, "keep_both_leaks": leaks,
               "keep_both_groups": both, "keep_both_exposure_recall": (both - leaks) / both if both else None,
               "vlm_groups": vlm, "vlm_unique_photos": 2 * vlm,
               "high_resolution_parts": 2 * vlm,
               "high_resolution_pixels": sum(p["page_pixels"] for p in plans if p["vlm_required"]),
               "local_quality": aggregate(local), "hybrid_replay_quality": aggregate(scored),
               "local_error_one_sided_95_upper_if_zero": zero_error_bound(local_n) if errors == 0 else None,
               "keep_both_leak_one_sided_95_upper_if_zero": zero_error_bound(both) if leaks == 0 else None,
               "capacity_projection": {"library_input_photos": 2065, "ungrouped_no_vlm": 1555,
                   "large_group_photos_all_exposed": 248, "all_pair_photos": 262,
                   "all_pair_groups": 131, "nonpair_vlm_groups": 63,
                   "unique_photos": total_projected, "vlm_groups": 63 + 131 * vlm / n,
                   "unique_photos_per_1000": total_projected / 2065 * 1000,
                   "vlm_groups_per_1000": (63 + 131 * vlm / n) / 2065 * 1000,
                   "saved_photos_per_1000": (510 - total_projected) / 2065 * 1000}}
    return plans, metrics


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write_json("checkpoint.json", {"status": "running", "phase": "authorized development inputs"})
    split = read_json(TUNING / "split.json")
    labels = read_json(TUNING / "development-labels.json")["groups"]
    groups, holdout = split["development_groups"], split["holdout_groups"]
    if len(groups) != 88 or len(set(groups)) != 88 or set(groups) != set(labels):
        raise RuntimeError("development allowlist mismatch")
    if len(holdout) != 43 or set(groups) & set(holdout):
        raise RuntimeError("holdout metadata mismatch")
    sources = {str(p): digest(p) for p in (TUNING / "split.json", TUNING / "development-labels.json", FROZEN)}
    loaded = {g: load_group(g, sources) for g in groups}
    # Only inventory metadata is consumed; frozen group records/labels are unused.
    inventories = {i["date"]: i for i in read_json(FROZEN)["inventories"]}
    db_audit = []
    for date in sorted({g[:10] for g in groups}):
        ids = {}
        for group in groups:
            if group[:10] == date:
                for frame in loaded[group][0]:
                    rec = resolver.identity_record(frame["identity"])
                    ids[rec["file_id"]] = rec["content_sha256"]
        inventory = inventories[date]
        path = Path(inventory["path"])
        before = digest(path)
        if before != inventory["sha256"]:
            raise RuntimeError("frozen DB hash mismatch")
        sources[str(path)] = before
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(path) + suffix)
            if sidecar.exists():
                sources[str(sidecar)] = digest(sidecar)
        records = fetch_development_rows(path, ids)
        blobs = {}
        validation = []
        for file_id, sha, blob, source_path in records:
            ident = f"I{file_id}@{sha}"
            original = Path(source_path)
            if digest(original) != sha:
                raise RuntimeError("original SHA mismatch")
            sources[str(original)] = sha
            blobs[ident] = blob
            try:
                resolver.decode_dino(blob)
                state = "valid"
            except ValueError as exc:
                state = str(exc)  # resolver routes this pair to VLM
            validation.append({"file_id": file_id, "content_sha256": sha,
                               "blob_bytes": len(blob) if isinstance(blob, bytes) else None,
                               "blob_sha256": hashlib.sha256(blob).hexdigest() if isinstance(blob, bytes) else None,
                               "validation": state})
        for group in groups:
            if group[:10] == date:
                loaded[group][1]["raw_dino_blobs"] = {f["identity"]: blobs[f["identity"]] for f in loaded[group][0]}
        db_audit.append({"date": date, "path": str(path), "sha256": before,
                         "queried_file_ids": sorted(ids), "query_mode": "ro+immutable+query_only; ID-bound SELECT",
                         "embeddings": validation})
    rows = [{"group": g, "frames": loaded[g][0], "evidence": loaded[g][1],
             "human": set(labels[g]["human_selected"]), "baseline": set(labels[g]["baseline_ai"])} for g in groups]
    for row in rows:
        ids = {resolver.identity_record(f["identity"])["file_id"] for f in row["frames"]}
        if not row["human"] or not row["human"] <= ids or not row["baseline"] <= ids:
            raise RuntimeError("development label identity mismatch")
    policy = {"resolver_version": resolver.VERSION, "locked_profile": "A",
              "profiles": resolver.PROFILES, "families": resolver.FAMILIES, "criteria": resolver.CRITERIA,
              "family_rule": "one vote per provenance family; conflict/tie/missing abstains; not statistical independence",
              "excluded_alias": "quality_score equals MUSIQ; never votes",
              "local_routes": ["local_select_one"], "local_keep_both": "forbidden",
              "dino": {"dtype": "<f2", "dimensions": 768, "bytes": 1536, "minimum_norm_exclusive": resolver.DINO_MIN_NORM, "normalization": "float64 L2 before cosine; stored vectors need not be unit length"},
              "scope": "88 development groups / 176 date-scoped file identities; 43 holdout metadata only",
              "actual_served_model": "amazon-bedrock-mantle/global.openai.gpt-6-astra",
              "harness": "OpenClaw native subagent; direct file tools and deterministic local Python; no ACP or external VLM"}
    write_json("policy.json", policy)
    frontier, all_plans = [], []
    for profile in resolver.PROFILES:
        plans, metrics = evaluate(rows, profile)
        frontier.append(metrics)
        all_plans.extend(plans)
    locked = frontier[0]
    if locked["keep_both_leaks"] or locked["local_errors"]:
        raise RuntimeError("Profile A failed zero-leak/high-precision gate; do not lock")
    folds = []
    for date in sorted({r["group"][:10] for r in rows}):
        test = [r for r in rows if r["group"][:10] == date]
        train = [r for r in rows if r["group"][:10] != date]
        _, train_metrics = evaluate(train, "A")
        _, test_metrics = evaluate(test, "A")
        folds.append({"held_out_date": date, "training_groups": len(train),
                      "training_gate_zero_errors_and_leaks": train_metrics["local_errors"] == train_metrics["keep_both_leaks"] == 0,
                      "test": test_metrics})
    write_json("leave-one-date-out.json", {"method": "fixed predeclared A; no fitting, threshold search or label-based fold selection",
                 "limitation": "post-design date-stratified stress test, not independent nested validation", "folds": folds})
    write_json("plans.json", {"groups": all_plans})
    write_json("frontier.json", {"resolver_version": resolver.VERSION, "locked_candidate": "A",
                 "baseline_v3": aggregate([{"predicted": r["baseline"], "human": r["human"]} for r in rows]),
                 "frontier": frontier, "holdout": "43 names/count only; no labels/images/old AI opened",
                 "bound_interpretation": "one-sided exact binomial upper 1-0.05**(1/n) for zero events; null when n=0. IID assumption and development selection bias prevent production guarantee.",
                 "capacity_interpretation": "route-rate projection to 131 pairs plus all 248 nonpair grouped photos; not measured holdout/production; every source within a sheet counts"})
    write_json("teacher-ledger.json", {"new_calls": 0, "automatic_retries": 0,
                 "baseline": "development-labels.json baseline_ai; no teacher result parsing needed",
                 "external_vlm": "not called"})
    after = {p: digest(Path(p)) for p in sources}
    if sources != after:
        raise RuntimeError("source/DB/original mutation")
    write_json("source-integrity.json", {"unchanged": True, "source_paths": len(sources),
                 "before": sources, "after": after, "databases": db_audit,
                 "writes": dict.fromkeys(("original", "database", "v3", "decision", "keeper", "delete"), 0)})
    write_json("checkpoint.json", {"status": "replay_complete", "phase": "await tests and final hash lock",
                 "local_groups": locked["local_select_one"], "keep_both_leaks": locked["keep_both_leaks"]})
    print(json.dumps({"frontier": frontier}, indent=2))


if __name__ == "__main__":
    main()
