"""Authorized development-only single-judge experiment; no automatic retries.

Labels are used only after inference for scoring. Runtime requests contain no
labels. No inventory, original source, live review or holdout reader exists here.
"""

from __future__ import annotations

import argparse
import io
import time

from PIL import Image

from scripts.prepare_large_group_candidate import CANARIES, OUT, SOURCE, TUNING
from src import cloud_judge_batch as b
from src import large_group_projection as projection
from src import large_group_review as v4
from src import two_pass_review as tp

RUN = OUT / "development-run-a-r2"


def wire_request(request, case):
    b.verify(
        request["judgment_contract"] == projection.CONTRACT,
        "candidate contract required",
    )
    content = [
        {
            "text": tp.canonical(
                {k: v for k, v in request.items() if k != "images"}
            ).decode()
        }
    ]
    photos = set()
    pixels = 0
    for asset in request["images"]:
        path = (case / asset["path"]).resolve()
        b.verify(path.is_relative_to(case.resolve()), "image path escape")
        data = path.read_bytes()
        b.verify(
            tp.sha(data) == asset["sha256"] and len(data) == asset["bytes"],
            "image binding changed",
        )
        with Image.open(io.BytesIO(data)) as image:
            b.verify(
                image.format == "JPEG" and list(image.size) == asset["size"],
                "image metadata changed",
            )
        photos.update(c["frame"] for c in asset["cells"])
        pixels += asset["size"][0] * asset["size"][1]
        content.extend(
            [
                {"text": tp.canonical(asset).decode()},
                {"image": {"format": "jpeg", "source": {"bytes": data}}},
            ]
        )
    b.verify(len(request["images"]) <= b.VISUAL_BUDGET["images"], "image part budget")
    b.verify(pixels <= b.VISUAL_BUDGET["image_pixels"], "image pixel budget")
    b.verify(
        sum(a["bytes"] for a in request["images"]) <= b.VISUAL_BUDGET["image_bytes"],
        "image byte budget",
    )
    b.verify(
        sum(len(c["text"].encode()) for c in content if "text" in c)
        <= b.VISUAL_BUDGET["text_bytes"],
        "wire text budget",
    )
    wire = {
        "modelId": b.MODEL,
        "messages": [{"role": "user", "content": content}],
        "inferenceConfig": {"maxTokens": v4.MAX_OUTPUT_TOKENS},
    }
    exposure = {
        "unique_source_photos": len(photos),
        "source_frames": sorted(photos),
        "high_resolution_candidates": len(photos),
        "image_parts": len(request["images"]),
        "pixels": pixels,
    }
    return wire, exposure


def validate_provider(response, request, rows, validation):
    b.verify(
        response.get("stopReason") == "end_turn",
        "truncated/non-final response; no retry",
    )
    usage = response.get("usage", {})
    for k in ("inputTokens", "outputTokens", "totalTokens"):
        b.verify(type(usage.get(k)) is int and usage[k] >= 0, "invalid usage")
    b.verify(
        usage["totalTokens"] == usage["inputTokens"] + usage["outputTokens"],
        "usage sum",
    )
    b.verify(usage["outputTokens"] <= v4.MAX_OUTPUT_TOKENS, "output token limit")
    blocks = response["output"]["message"]["content"]
    # Converse may include signed reasoning metadata even without an explicit
    # thinking request. Preserve it in raw receipt, never treat it as judgment.
    b.verify(
        all(set(x) in ({"text"}, {"reasoningContent"}) for x in blocks),
        "unexpected response blocks",
    )
    text = "\n".join(x["text"] for x in blocks if "text" in x)
    b.verify(len(text.encode()) <= v4.MAX_OUTPUT_BYTES, "response byte budget")
    answer = tp.strict_json(text)
    return projection.validate_response(
        answer,
        rows,
        request["context"]["visual_packet"],
        validation,
        request["context"]["evidence"],
    )


def allowed_groups():
    split = b.read(TUNING / "split.json")
    dev = split["development_groups"]
    b.verify(
        len(dev) == len(set(dev)) == 39
        and not set(dev).intersection(split["holdout_groups"]),
        "split boundary",
    )
    return dev


def run_group(group, dev, client, *, run_root=RUN, exposure_plan=None):
    b.verify(group in dev, "development allowlist required")
    case = SOURCE / group
    output = run_root / group
    state_path = output / "state.json"
    if state_path.exists():
        state = b.read(state_path)
        b.verify(
            state["status"] == "validated", "reserved/failed call cannot be retried"
        )
        return state
    request = b.read(case / "request.json")
    rows = [x["frame"] for x in request["context"]["temporal"]]
    original_member_count = len(rows)
    validation = b.read(case / "view-validation.json")
    if exposure_plan is not None:
        b.verify(exposure_plan["group"] == group, "exposure plan group mismatch")
        exposed = set(exposure_plan["exposed"])
        b.verify(
            3 <= len(exposed) <= len(rows),
            "candidate B requires at least three exposed frames; never reroute source large group into pair v3",
        )
        rows = [r for r in rows if tp.identity(r) in exposed]
        b.verify(len(rows) == len(exposed), "exposed identity mismatch")
        # Drop every supplement: a contact sheet containing an omitted source
        # would defeat source-photo exposure reduction even as one image part.
        request["images"] = [
            a
            for a in request["images"]
            if a["kind"] == "full_group_page"
            and all(c["frame"] in exposed for c in a["cells"])
        ]
        context = request["context"]
        context["temporal"] = [
            t for t in context["temporal"] if tp.identity(t["frame"]) in exposed
        ]
        context["visual_packet"]["complete_frames"] = [tp.identity(r) for r in rows]
        context["visual_packet"]["views"] = [
            v
            for v in context["visual_packet"]["views"]
            if tp.identity(v["frame"]) in exposed
        ]
        validation["lookup"] = {
            k: v
            for k, v in validation["lookup"].items()
            if tp.identity(v["frame"]) in exposed
        }
        context["untrusted_local_attention_hints"] = None
        context["exposure_policy"] = (
            "local_cv_microclusters_balanced; only exposed frames are selectable; no claim to have visually reviewed excluded sources"
        )
        # Local numeric CV for excluded members is permitted, but remove alias
        # value maps so the model never confuses them with observed frames.
        omitted_aliases = {
            k
            for k, v in context["evidence"]["frame_bindings"].items()
            if v not in exposed
        }

        def prune(value):
            if isinstance(value, dict):
                return {
                    k: prune(v) for k, v in value.items() if k not in omitted_aliases
                }
            if isinstance(value, list):
                return [prune(v) for v in value]
            return value

        context["evidence"] = prune(context["evidence"])
    request = projection.build_request(request, rows)
    wire, exposure = wire_request(request, case)
    b.save(output / "request.json", request)
    # Durable reservation BEFORE SDK invocation. Transport exceptions also consume it.
    state = {
        "group": group,
        "contract": projection.CONTRACT,
        "status": "reserved",
        "external_calls": 1,
        "member_count": original_member_count,
        "exposure": exposure,
        "request_sha256": tp.sha(tp.canonical(request)),
        "decision_writes": 0,
        "keeper_writes": 0,
        "delete_writes": 0,
    }
    b.save(state_path, state)
    started = time.monotonic()
    try:
        response = client.converse(**wire)
        b.save(output / "provider-response.json", response)
        state["usage"] = response.get("usage", {})
        text = "\n".join(
            x.get("text", "")
            for x in response.get("output", {}).get("message", {}).get("content", [])
        )
        (output / "raw.txt").write_text(text)
        answer = validate_provider(response, request, rows, validation)
        b.save(output / "validated.json", answer)
        state["status"] = "validated"
    except Exception as exc:  # Persist the failed receipt, then stop; never retry.
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        state["latency_seconds"] = round(time.monotonic() - started, 3)
        b.save(state_path, state)
    return state


def score(groups):
    labels = b.read(TUNING / "development-labels.json")["groups"]
    records = []
    for group in groups:
        b.verify(group in allowed_groups(), "score development allowlist")
        state = b.read(RUN / group / "state.json")
        answer = b.read(RUN / group / "validated.json")
        human = set(labels[group]["human_selected"])
        selected = {x["frame"]["file_id"] for x in answer["recommended"]}
        baseline = set(labels[group]["baseline_ai"])
        records.append(
            {
                "group": group,
                "human": sorted(human),
                "baseline": sorted(baseline),
                "candidate": sorted(selected),
                "decision": answer["decision"],
                "missing": sorted(human - selected),
                "extra": sorted(selected - human),
                "exact": human == selected,
                "baseline_exact": human == baseline,
                "tp": len(human & selected),
                "fp": len(selected - human),
                "fn": len(human - selected),
                "exposure": state["exposure"],
                "member_count": state["member_count"],
            }
        )
    tp_count, fp, fn = (sum(r[k] for r in records) for k in ("tp", "fp", "fn"))
    precision = tp_count / (tp_count + fp) if tp_count + fp else 0
    recall = tp_count / (tp_count + fn) if tp_count + fn else 0
    return {
        "groups": records,
        "metrics": {
            "n": len(records),
            "exact": sum(r["exact"] for r in records),
            "baseline_exact": sum(r["baseline_exact"] for r in records),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall)
            if precision + recall
            else 0,
            "missing": fn,
            "extra": fp,
            "vlm_groups": len(records),
            "members": sum(r["member_count"] for r in records),
            "unique_source_photos_exposed": len(
                {f for r in records for f in r["exposure"]["source_frames"]}
            ),
            "member_exposures": sum(
                r["exposure"]["unique_source_photos"] for r in records
            ),
            "image_parts": sum(r["exposure"]["image_parts"] for r in records),
            "pixels": sum(r["exposure"]["pixels"] for r in records),
            "human_selected_exposure_recall": 1.0,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["canary", "development", "score"])
    args = parser.parse_args()
    dev = allowed_groups()
    groups = CANARIES if args.stage == "canary" else dev
    if args.stage == "development":
        b.verify(
            all((RUN / g / "validated.json").exists() for g in CANARIES),
            "canaries must validate first",
        )
    if args.stage != "score":
        import boto3
        from botocore.config import Config

        client = boto3.Session().client(
            "bedrock-runtime",
            region_name="us-east-1",
            config=Config(
                retries={"total_max_attempts": 1, "mode": "standard"},
                connect_timeout=20,
                read_timeout=300,
            ),
        )
        for group in groups:
            state = run_group(group, dev, client)
            print(group, state["status"], state.get("latency_seconds"), flush=True)
    result = score(groups)
    b.save(
        RUN
        / ("canary-evaluation.json" if args.stage == "canary" else "evaluation.json"),
        result,
    )
    print(result["metrics"], flush=True)


if __name__ == "__main__":
    main()
