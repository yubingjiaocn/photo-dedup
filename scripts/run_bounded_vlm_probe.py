"""One-call bounded Sonnet/Terra resolution and semantic-observation probes."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path

import boto3
from botocore.config import Config
from PIL import Image

from src.vlm_review import assert_model_context_safe

ALLOWED_MODELS = {
    "global.anthropic.claude-sonnet-5-5",
    "global.openai.gpt-5.6-terra",
}
MAX_IMAGES = 4
MAX_TOTAL_BYTES = 2_000_000

RESOLUTION_PROMPT = """You are a bounded visual observer for a photo-selection research prototype.
The next four images are two labelled photo groups, each shown at 2048px and 1024px:
1. 2026-03-29-G012 at 2048px (frames F01-F05)
2. 2026-03-29-G012 at 1024px (same frames)
3. 2026-05-27-G005 at 2048px (frames F01-F04)
4. 2026-05-27-G005 at 1024px (same frames)

Each contact sheet contains full frames and labelled face crops. Do not identify people. Do not infer hidden intent, personality, age, relationships, or internal emotion. Do not judge eye-open/closed, sharpness, exposure, crop defects, or other CV tasks. Observe only broad visible expression, pose/action, camera engagement, interaction, and whether one frame captures a visibly distinct moment. Do not assume any existing keeper is correct.

Compare the 2048px version first, then independently verify whether the 1024px version supports the same claims. If a distinction is not directly visible, abstain. Output exactly one JSON object and no Markdown, with this schema:
{
  "schema_version": 1,
  "groups": [
    {
      "group_id": "...",
      "resolution_consistency": "consistent|partial|inconsistent|unreadable",
      "resolution_notes": "...",
      "semantic_claims": [
        {
          "code": "EXPRESSION_DIFFERENCE|POSE_ACTION_DIFFERENCE|CAMERA_ENGAGEMENT|UNIQUE_MOMENT|NO_RELIABLE_DIFFERENCE",
          "subjects": ["F01"],
          "compared_to": ["F02"],
          "observation": "directly visible, neutral description",
          "confidence": "high|medium|low",
          "resolution_sensitive": true
        }
      ],
      "abstentions": ["..."]
    }
  ],
  "global_assessment": {
    "is_2048_sufficient": true,
    "is_1024_sufficient": true,
    "recommended_minimum": "2048|1024|unresolved",
    "notes": "..."
  }
}
"""

SEMANTIC_PROMPT = """You are the group-level deep judge for a photo-selection research prototype.
The next three images are 2048px labelled contact sheets:
1. 2026-03-29-G008 (frames F01-F02)
2. 2026-03-29-G011 (frames F01-F04)
3. 2026-05-27-G013 (frames F01-F03)

Each sheet contains full frames and, where available, labelled face crops. A MODEL_CONTEXT_JSON block may follow the images. It contains every available per-frame CV measurement, including weak, default-off, and shadow-only signals. Consider all of them jointly with the pixels: a weak signal may matter in combination, but keep its supplied applicability and authority and never turn it into ground truth. Independently recommend the frame(s) that a human should inspect first; do not assume any existing keeper is correct. Select at most three discriminators that most reduce human scanning effort. Each discriminator must cite a visible frame/region, an exact CV ref from MODEL_CONTEXT_JSON, or both. Do not identify people or infer hidden intent, personality, age, relationships, or internal emotion. If evidence is not reliable enough, abstain.

Output exactly one JSON object and no Markdown, with this schema:
{
  "schema_version": 1,
  "groups": [
    {
      "group_id": "...",
      "judgment": {
        "recommended": ["F01"],
        "confidence": "high|medium|low",
        "discriminators": [
          {
            "code": "EXPRESSION_DIFFERENCE|POSE_ACTION_DIFFERENCE|CAMERA_ENGAGEMENT|SCENE_INFORMATION_DIFFERENCE|UNIQUE_MOMENT|EYE_STATE_DIFFERENCE|FACE_QUALITY_DIFFERENCE|TECHNICAL_QUALITY_DIFFERENCE|EXPOSURE_DIFFERENCE|FRAMING_DIFFERENCE|COMPOSITE_ADVANTAGE|NO_RELIABLE_DIFFERENCE",
            "subjects": ["F01"],
            "compared_to": ["F02"],
            "reason": "concise human-readable distinction",
            "confidence": "high|medium|low",
            "visual_refs": [{"frame": "F01", "region": "full_frame|face|subject|background"}],
            "cv_refs": ["F01:EXACT_CODE_FROM_CONTEXT"]
          }
        ]
      },
      "abstentions": ["..."]
    }
  ],
  "global_assessment": {
    "is_2048_sufficient": true,
    "notes": "..."
  }
}
"""

PROBES = {
    "resolution": {
        "prompt": RESOLUTION_PROMPT,
        "image_count": 4,
        "groups": {"2026-03-29-G012", "2026-05-27-G005"},
    },
    "semantic": {
        "prompt": SEMANTIC_PROMPT,
        "image_count": 3,
        "groups": {"2026-03-29-G008", "2026-03-29-G011", "2026-05-27-G013"},
    },
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_text(response: dict) -> str:
    parts = []
    for block in response["output"]["message"]["content"]:
        if "text" in block:
            parts.append(block["text"])
    text = "\n".join(parts).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return text


def run(probe: str, model_id: str, images: list[Path], output: Path,
        cv_context: Path | None = None) -> dict:
    if model_id not in ALLOWED_MODELS:
        raise ValueError(f"model is not in bounded allowlist: {model_id}")
    spec = PROBES[probe]
    if len(images) != spec["image_count"] or len(images) > MAX_IMAGES:
        raise ValueError(f"{probe} probe requires {spec['image_count']} images")
    payloads = []
    total_bytes = 0
    for path in images:
        data = path.read_bytes()
        total_bytes += len(data)
        with Image.open(path) as image:
            width, height = image.size
        payloads.append({
            "path": path, "name": path.name, "bytes": data,
            "sha256": _sha256(data), "width": width, "height": height,
        })
    if total_bytes > MAX_TOTAL_BYTES:
        raise ValueError(f"image payload exceeds {MAX_TOTAL_BYTES} bytes: {total_bytes}")

    prompt = spec["prompt"]
    content = [{"text": prompt}]
    for index, payload in enumerate(payloads, start=1):
        content.extend([
            {"text": f"IMAGE {index}: {payload['name']}"},
            {"image": {"format": "jpeg", "source": {"bytes": payload["bytes"]}}},
        ])
    model_context_sha256 = None
    if cv_context is not None:
        supplied = json.loads(cv_context.read_bytes())
        model_context = supplied.get("model_context") if isinstance(
            supplied, Mapping,
        ) and "model_context" in supplied else supplied
        if not isinstance(model_context, Mapping):
            raise TypeError("--cv-context must contain a model_context object")
        assert_model_context_safe(model_context)
        model_context_bytes = json.dumps(
            model_context, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        model_context_sha256 = _sha256(model_context_bytes)
        content.append({
            "text": "MODEL_CONTEXT_JSON\n" + model_context_bytes.decode("utf-8"),
        })
    client = boto3.client(
        "bedrock-runtime", region_name="us-east-1",
        config=Config(retries={"max_attempts": 1, "mode": "standard"}),
    )
    started = time.perf_counter()
    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": content}],
        inferenceConfig={"maxTokens": 4000},
    )
    latency = time.perf_counter() - started
    text = _json_text(response)
    parsed = json.loads(text)
    if (parsed.get("schema_version") != 1
            or len(parsed.get("groups", [])) != len(spec["groups"])):
        raise ValueError("VLM output failed bounded schema gate")
    if {group.get("group_id") for group in parsed["groups"]} != spec["groups"]:
        raise ValueError("VLM output group IDs do not match the inputs")

    report = {
        "schema_version": 1,
        "mode": "shadow_review_only",
        "probe": probe,
        "model_id": model_id,
        "model_calls": 1,
        "image_count": len(payloads),
        "group_count": len(spec["groups"]),
        "total_image_bytes": total_bytes,
        "prompt_sha256": _sha256(prompt.encode()),
        "model_context_sha256": model_context_sha256,
        "images": [
            {key: payload[key] for key in ("name", "sha256", "width", "height")}
            | {"bytes": len(payload["bytes"])}
            for payload in payloads
        ],
        "latency_seconds": latency,
        "usage": response.get("usage"),
        "stop_reason": response.get("stopReason"),
        "observations": parsed,
        "invariants": {
            "changes_keeper": False,
            "changes_sorting": False,
            "changes_routing": False,
            "deletion_authority": "none",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", choices=sorted(PROBES), default="resolution")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cv-context", type=Path,
        help="prompt-safe model context or a VLM sidecar containing model_context",
    )
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()
    report = run(args.probe, args.model_id, args.images, args.output, args.cv_context)
    print(json.dumps({
        key: report[key] for key in (
            "model_id", "model_calls", "image_count", "group_count",
            "total_image_bytes", "latency_seconds", "usage", "stop_reason",
        )
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
