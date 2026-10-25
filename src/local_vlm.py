"""Opt-in local-trial VLM suggestions through an OpenAI-compatible endpoint.

Usage (normally via ``python -m src.local_trial judge`` / ``start --vlm``)::

    set PHOTO_VLM_API_KEY=...            (optional; environment only)
    python -m src.local_trial judge --output D:\\review --vlm-base-url http://127.0.0.1:4000/v1 --vlm-model <alias>

What this module does, and nothing more:

* Picks a bounded set of review groups (default 8, hard max 20): multi-member
  groups of at most 20 photos. Byte-identical groups stay CV-only and larger
  groups are recorded as manual review -- never silently truncated.
* Per selected group, makes **one** ``POST {base}/chat/completions`` with every
  member as a freshly re-encoded, EXIF-free, full-frame JPEG labelled ``P1..Pn``.
  The provider never sees paths, file names, GPS or other metadata. Originals
  are opened read-only after their inventory size/mtime/SHA-256 identity is
  re-checked; nothing is written next to them.
* Validates the reply strictly (``local-trial-vlm-v1``) and stores a derived
  sidecar under ``<output>/local_vlm/``. Failures are stored and shown too; there
  is no retry, no fallback model/provider and no repair of a malformed answer.
* Cached results are keyed by endpoint identity + model + prompt + exact member
  file id/SHA + sent image hashes; a matching sidecar is reused without a call
  unless ``force`` is given.

The API key is read only from the environment (``PHOTO_VLM_API_KEY``), never
logged, never stored and never accepted from the browser. A schema-valid answer
is a *suggestion for a human*, not evidence of accuracy: nothing here touches the
inventory DB, the AI keeper, the decision tables or the human review state.
"""

from __future__ import annotations

import base64
import hashlib
import io
import ipaddress
import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

SCHEMA = "local-trial-vlm-v1"
DIR_NAME = "local_vlm"
KEY_ENV = "PHOTO_VLM_API_KEY"
BASE_URL_ENV = "PHOTO_VLM_BASE_URL"
MODEL_ENV = "PHOTO_VLM_MODEL"
DEFAULT_GROUPS = 8
MAX_GROUPS = 20
MAX_MEMBERS = 20
DEFAULT_TIMEOUT_S = 120.0
MAX_RESPONSE_BYTES = 1_000_000
MAX_CONTENT_CHARS = 20_000
MAX_GROUP_IMAGE_BYTES = 12_000_000
MAX_REASONS = 3
MAX_REASON_CHARS = 80
MAX_TOKENS = 800
CONFIDENCE = ("high", "medium", "low")

PROMPT = """你是照片整理助手，只给人类审阅者提供参考建议，不做最终决定。
这一组是同一时间附近拍摄的相似照片，标签为 {aliases}（每张图前有对应标签）。
请判断哪些照片值得保留：
- 不要强行只选一张。不同瞬间、不同表情、动作的不同阶段、不同人物组合，都可以同时保留。
- 只有在明显更差（失焦、闭眼、模糊、构图截断、几乎完全重复）时才不推荐。
- 不确定时请弃权（status = "abstain"，keep 为空数组）。
只输出一个 JSON 对象，不要输出任何其它文字：
{{"status": "recommend" 或 "abstain",
 "keep": [推荐保留的标签，如 "P1"，不重复，只能用上面的标签],
 "confidence": "high" | "medium" | "low",
 "reasons": [最多 3 条，每条 {{"text": "不超过 40 字的中文理由或可见差别", "photos": [相关标签]}}]}}"""
PROMPT_VERSION = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()[:16]

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*)\n```\s*$", re.S)


class ConfigError(ValueError):
    """Bad endpoint/model configuration; message is safe to print."""


# --- configuration ---------------------------------------------------------

def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def endpoint_identity(base_url: str) -> str:
    """Validated ``scheme://host[:port]/path`` (no credentials, query or fragment).

    Plain HTTP is accepted only for a loopback host (the user's SSH/port-forward
    tunnel); anything remote must be HTTPS, which urllib verifies by default.
    """
    if not isinstance(base_url, str) or not base_url.strip():
        raise ConfigError("VLM base URL is required (e.g. http://127.0.0.1:4000/v1)")
    parsed = urllib.parse.urlsplit(base_url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ConfigError("VLM base URL must look like http://127.0.0.1:4000/v1")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ConfigError("VLM base URL must not contain user:password, ?query or #fragment; "
                          f"put the key in the {KEY_ENV} environment variable")
    if parsed.scheme == "http" and not _is_loopback(parsed.hostname):
        raise ConfigError("plain http:// is only allowed for a local tunnel (127.0.0.1/localhost); "
                          "use https:// for a remote endpoint")
    host = parsed.hostname if ":" not in parsed.hostname else f"[{parsed.hostname}]"
    netloc = f"{host}:{parsed.port}" if parsed.port else host
    return f"{parsed.scheme}://{netloc}{parsed.path.rstrip('/')}"


def make_config(base_url: Optional[str], model: Optional[str], *, timeout_s: float = DEFAULT_TIMEOUT_S,
                response_format: bool = True, environ: Mapping[str, str] = os.environ) -> Dict[str, Any]:
    base = base_url or environ.get(BASE_URL_ENV) or ""
    model = (model or environ.get(MODEL_ENV) or "").strip()
    if not model:
        raise ConfigError(f"VLM model alias is required (--vlm-model or {MODEL_ENV})")
    if len(model) > 200 or _CONTROL_RE.search(model):
        raise ConfigError("VLM model alias is not a valid name")
    if not (1 <= float(timeout_s) <= 600):
        raise ConfigError("VLM timeout must be between 1 and 600 seconds")
    return {"endpoint": endpoint_identity(base), "model": model, "timeout_s": float(timeout_s),
            "response_format": bool(response_format)}


# --- images ----------------------------------------------------------------

def _max_edge(count: int) -> int:
    return 2048 if count <= 3 else 1536 if count <= 6 else 1024 if count <= 12 else 768


def render_member(path: Path, max_edge: int) -> bytes:
    """Full-frame resized JPEG with no EXIF/ICC/metadata carried over."""
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image.thumbnail((max_edge, max_edge))
        clean = Image.new("RGB", image.size)
        clean.paste(image)
    buffer = io.BytesIO()
    clean.save(buffer, "JPEG", quality=85)
    return buffer.getvalue()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- request / response ------------------------------------------------------

def build_request(config: Mapping[str, Any], images: Sequence[bytes]) -> Dict[str, Any]:
    aliases = [f"P{i + 1}" for i in range(len(images))]
    content: List[Dict[str, Any]] = [{"type": "text", "text": PROMPT.format(aliases="、".join(aliases))}]
    for alias, data in zip(aliases, images):
        content.append({"type": "text", "text": alias})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")}})
    body: Dict[str, Any] = {"model": config["model"], "max_tokens": MAX_TOKENS,
                            "messages": [{"role": "user", "content": content}]}
    if config.get("response_format", True):
        body["response_format"] = {"type": "json_object"}
    return body


def _clean_text(value: Any, limit: int = 300, secret: str = "") -> str:
    text = _CONTROL_RE.sub(" ", str(value))
    if secret:
        text = text.replace(secret, "[redacted]")
    return text[:limit]


def post_chat(config: Mapping[str, Any], body: Mapping[str, Any],
              environ: Mapping[str, str] = os.environ) -> Dict[str, Any]:
    """One POST, no retries. Returns {"ok": True, "json": ...} or a clean error."""
    key = environ.get(KEY_ENV, "")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(config["endpoint"] + "/chat/completions",
                                     data=json.dumps(body).encode("utf-8"), headers=headers,
                                     method="POST")
    try:
        with urllib.request.urlopen(request, timeout=config["timeout_s"]) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            status = response.status
    except urllib.error.HTTPError as exc:
        detail = _clean_text(exc.read(2000).decode("utf-8", "replace"), secret=key)
        kind = ("auth" if exc.code in (401, 403) else
                "model_or_request_rejected" if exc.code in (400, 404, 422) else "http_error")
        return {"ok": False, "error": kind, "http_status": exc.code, "detail": detail}
    except (socket.timeout, TimeoutError):
        return {"ok": False, "error": "timeout", "detail": f"no reply within {config['timeout_s']:g}s"}
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, (socket.timeout, TimeoutError)):
            return {"ok": False, "error": "timeout", "detail": f"no reply within {config['timeout_s']:g}s"}
        return {"ok": False, "error": "offline", "detail": _clean_text(reason, secret=key)}
    except OSError as exc:
        return {"ok": False, "error": "offline", "detail": _clean_text(exc, secret=key)}
    if len(raw) > MAX_RESPONSE_BYTES:
        return {"ok": False, "error": "response_too_large", "http_status": status}
    try:
        return {"ok": True, "http_status": status, "json": json.loads(raw.decode("utf-8"))}
    except (UnicodeDecodeError, ValueError):
        return {"ok": False, "error": "malformed_envelope", "http_status": status}


def validate_answer(content: Any, aliases: Sequence[str]) -> Dict[str, Any]:
    """Strictly parse the model text. Raises ValueError with a short reason."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty model reply")
    if len(content) > MAX_CONTENT_CHARS:
        raise ValueError("model reply too long")
    text = content.strip()
    fenced = _FENCE_RE.match(text)
    if fenced:  # a single ```json fence is unwrapped; nothing else is "fixed"
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except ValueError:
        raise ValueError("model reply is not one JSON object") from None
    if not isinstance(value, dict) or set(value) != {"status", "keep", "confidence", "reasons"}:
        raise ValueError("reply must have exactly status/keep/confidence/reasons")
    status, keep, confidence, reasons = (value[k] for k in ("status", "keep", "confidence", "reasons"))
    allowed = set(aliases)
    if status not in ("recommend", "abstain"):
        raise ValueError("status must be recommend or abstain")
    if confidence not in CONFIDENCE:
        raise ValueError("confidence must be high/medium/low")
    if not isinstance(keep, list) or not all(isinstance(a, str) for a in keep):
        raise ValueError("keep must be a list of photo labels")
    if any(a not in allowed for a in keep):
        raise ValueError("keep names a photo that is not in this group")
    if len(set(keep)) != len(keep):
        raise ValueError("keep repeats a photo")
    if (status == "recommend") != bool(keep):
        raise ValueError("recommend needs >=1 photo; abstain needs an empty keep")
    if not isinstance(reasons, list) or len(reasons) > MAX_REASONS:
        raise ValueError(f"reasons must be a list of at most {MAX_REASONS}")
    clean_reasons = []
    for reason in reasons:
        if (not isinstance(reason, dict) or set(reason) != {"text", "photos"}
                or not isinstance(reason["text"], str) or not reason["text"].strip()
                or len(reason["text"]) > MAX_REASON_CHARS or _CONTROL_RE.search(reason["text"])
                or not isinstance(reason["photos"], list)
                or any(not isinstance(a, str) or a not in allowed for a in reason["photos"])
                or len(set(reason["photos"])) != len(reason["photos"])):
            raise ValueError("each reason needs short text and photos from this group")
        clean_reasons.append({"text": reason["text"].strip(), "photos": list(reason["photos"])})
    order = {a: i for i, a in enumerate(aliases)}
    return {"status": status, "keep": sorted(keep, key=order.__getitem__),
            "confidence": confidence, "reasons": clean_reasons}


# --- sidecars ----------------------------------------------------------------

def sidecar_dir(output: Path) -> Path:
    return Path(output) / DIR_NAME


def sidecar_path(output: Path, group_id: int) -> Path:
    return sidecar_dir(output) / f"group-{int(group_id)}.json"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def cache_key(config: Mapping[str, Any], members: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_canonical({
        "schema": SCHEMA, "endpoint": config["endpoint"], "model": config["model"],
        "prompt": PROMPT_VERSION, "response_format": bool(config.get("response_format", True)),
        "members": [[m["alias"], m["file_id"], m["sha256"], m.get("image_sha256")] for m in members],
    })).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _members_of(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    ordered = sorted(rows, key=lambda r: int(r["file_id"]))
    return [{"alias": f"P{i + 1}", "file_id": int(r["file_id"]), "sha256": r["content_sha256"]}
            for i, r in enumerate(ordered)]


def load_for_group(output: Path, rows: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """UI projection of a group's sidecar, or None when absent/stale/invalid.

    Stale = schema, prompt, member file ids or member SHA-256 differ from the
    current inventory. Recommendations are re-validated against the current
    members before anything is projected.
    """
    path = sidecar_path(output, int(rows[0]["group_id"])) if rows else None
    if path is None or not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    current = _members_of(rows)
    if (not isinstance(record, dict) or record.get("schema") != SCHEMA
            or record.get("prompt_version") != PROMPT_VERSION
            or [[m.get("alias"), m.get("file_id"), m.get("sha256")] for m in record.get("members") or []]
            != [[m["alias"], m["file_id"], m["sha256"]] for m in current]):
        return None
    alias_to_id = {m["alias"]: m["file_id"] for m in current}
    projection = {"schema": SCHEMA, "status": record.get("status"), "model": record.get("model"),
                  "served_model": record.get("served_model"), "endpoint": record.get("endpoint"),
                  "created_at": record.get("created_at"), "aliases": alias_to_id,
                  "human_confirmation": "仅供参考：需要你点“接受VLM建议”才会保存，不会自动改动任何选择。"}
    if record.get("status") == "ok":
        try:
            answer = validate_answer(json.dumps(record.get("answer")), list(alias_to_id))
        except ValueError:
            return None
        projection.update({
            "verdict": answer["status"], "confidence": answer["confidence"],
            "keep_file_ids": [alias_to_id[a] for a in answer["keep"]],
            "reasons": [{"text": r["text"], "file_ids": [alias_to_id[a] for a in r["photos"]]}
                        for r in answer["reasons"]],
        })
    else:
        projection["error"] = record.get("error")
        projection["detail"] = record.get("detail")
    return projection


# --- judging -----------------------------------------------------------------

TRANSPORT_ERRORS = frozenset({"offline", "auth", "timeout", "model_or_request_rejected", "http_error",
                              "response_too_large", "malformed_envelope"})

FRIENDLY = {
    "offline": "连不上 VLM 接口：请确认端口转发/LiteLLM 已启动，地址正确",
    "auth": "VLM 接口拒绝了认证：请检查 PHOTO_VLM_API_KEY 环境变量",
    "model_or_request_rejected": "VLM 接口拒绝了请求：模型别名可能不存在，或不支持图片/response_format",
    "timeout": "VLM 接口超时",
    "invalid_answer": "模型回复不符合格式，本组请人工判断",
}


def _select(data: Any, max_groups: int) -> Dict[str, Any]:
    from . import db

    chosen, skipped = [], []
    for group_id in data.group_ids():
        with data._lock:
            rows = [dict(r) for r in db.group_page_by_id(data._conn, group_id, scope=data.scope)]
        if len(rows) < 2:
            continue
        if len(rows) > MAX_MEMBERS:
            skipped.append({"group_id": group_id, "status": "manual_too_large", "members": len(rows)})
        elif all(r.get("content_sha256") for r in rows) and len({r["content_sha256"] for r in rows}) == 1:
            skipped.append({"group_id": group_id, "status": "cv_only_identical", "members": len(rows)})
        elif len(chosen) < max_groups:
            chosen.append(rows)
        else:
            skipped.append({"group_id": group_id, "status": "not_selected_budget", "members": len(rows)})
    return {"chosen": chosen, "skipped": skipped}


def judge_group(data: Any, rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], *,
                force: bool = False, poster: Callable[..., Dict[str, Any]] = post_chat) -> Dict[str, Any]:
    group_id = int(rows[0]["group_id"])
    members = _members_of(rows)
    base = {"schema": SCHEMA, "group_id": group_id, "endpoint": config["endpoint"],
            "model": config["model"], "prompt_version": PROMPT_VERSION, "members": members,
            "created_at": int(time.time()), "model_calls": 0}
    images: List[bytes] = []
    edge = _max_edge(len(members))
    for member in members:
        record = data.original_record(member["file_id"])
        try:
            stat = record.path.stat() if record else None
            if (record is None or not member["sha256"] or stat.st_size != record.size_bytes
                    or stat.st_mtime_ns != record.mtime_ns or _sha_file(record.path) != member["sha256"]):
                raise ValueError
            image = render_member(record.path, edge)
        except (OSError, ValueError):
            failed = {**base, "status": "failed", "error": "source_changed",
                      "detail": f"{member['alias']} 原图不可读或与扫描时不一致，请重新扫描"}
            _write_json(sidecar_path(data.output, group_id), failed)
            return failed
        member["image_sha256"] = hashlib.sha256(image).hexdigest()
        member["image_bytes"] = len(image)
        images.append(image)
    if sum(len(i) for i in images) > MAX_GROUP_IMAGE_BYTES:
        failed = {**base, "status": "failed", "error": "payload_too_large",
                  "detail": "本组图片总量超过单次请求上限，请人工判断"}
        _write_json(sidecar_path(data.output, group_id), failed)
        return failed
    key = cache_key(config, members)
    path = sidecar_path(data.output, group_id)
    if not force and path.is_file():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached = None
        # Only an answer the model actually gave is reused; a transport failure
        # (tunnel offline, auth, timeout) is attempted again on the next explicit run.
        if (isinstance(cached, dict) and cached.get("cache_key") == key
                and (cached.get("status") == "ok" or cached.get("error") == "invalid_answer")):
            return {**cached, "cache_hit": True}
    started = time.monotonic()
    reply = poster(config, build_request(config, images))
    record = {**base, "cache_key": key, "image_max_edge": edge, "model_calls": 1,
              "latency_s": round(time.monotonic() - started, 3), "http_status": reply.get("http_status")}
    if not reply.get("ok"):
        record.update(status="failed", error=reply["error"], detail=reply.get("detail"))
    else:
        envelope = reply["json"]
        served = envelope.get("model") if isinstance(envelope, dict) else None
        record["served_model"] = _clean_text(served, 200) if isinstance(served, str) else None
        usage = envelope.get("usage") if isinstance(envelope, dict) else None
        if isinstance(usage, dict):
            record["usage"] = {k: usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                               if isinstance(usage.get(k), int)}
        try:
            content = envelope["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = None
        record["raw_content"] = _clean_text(content, MAX_CONTENT_CHARS) if isinstance(content, str) else None
        try:
            record["answer"] = validate_answer(content, [m["alias"] for m in members])
            record["status"] = "ok"
        except ValueError as exc:
            record.update(status="failed", error="invalid_answer", detail=str(exc))
    if record["status"] == "failed" and record["error"] in TRANSPORT_ERRORS:
        # A dead tunnel or bad key must not erase a suggestion the reviewer already
        # has; the failure is still reported in the run summary and console.
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = None
        if isinstance(previous, dict) and previous.get("status") == "ok":
            return {**record, "kept_previous": True}
    _write_json(path, record)
    return record


def run(output: Path, config: Mapping[str, Any], *, max_groups: int = DEFAULT_GROUPS,
        force: bool = False, poster: Callable[..., Dict[str, Any]] = post_chat,
        log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Judge a bounded set of groups; returns (and writes) a run summary."""
    from . import db
    from .review_server import ReviewData

    if not (1 <= int(max_groups) <= MAX_GROUPS):
        raise ConfigError(f"--vlm-max-groups must be between 1 and {MAX_GROUPS}")
    output = Path(output).resolve()
    data = ReviewData(output)
    try:
        selection = _select(data, int(max_groups))
        for item in selection["skipped"]:
            if item["status"] != "not_selected_budget":
                with data._lock:
                    rows = [dict(r) for r in db.group_page_by_id(
                        data._conn, item["group_id"], scope=data.scope)]
                _write_json(sidecar_path(output, item["group_id"]), {
                    "schema": SCHEMA, "group_id": item["group_id"], "status": item["status"],
                    "prompt_version": PROMPT_VERSION, "members": _members_of(rows),
                    "endpoint": config["endpoint"], "model": config["model"],
                    "created_at": int(time.time()), "model_calls": 0,
                    "detail": ("超过 20 张，请人工审阅" if item["status"] == "manual_too_large"
                               else "完全相同的文件，保留 CV 结果")})
        total = len(selection["chosen"])
        log(f"[local_vlm] VLM stage: {total} group(s) to judge with model '{config['model']}' "
            f"at {config['endpoint']} (max {max_groups}; key from {KEY_ENV}: "
            f"{'set' if os.environ.get(KEY_ENV) else 'not set'})")
        results = []
        for index, rows in enumerate(selection["chosen"], 1):
            result = judge_group(data, rows, config, force=force, poster=poster)
            label = ("cache hit, no call" if result.get("cache_hit") else
                     f"FAILED: {FRIENDLY.get(result.get('error'), result.get('error'))} "
                     "(earlier suggestion kept)" if result.get("kept_previous") else
                     "ok" if result["status"] == "ok" else
                     f"FAILED: {FRIENDLY.get(result.get('error'), result.get('error'))}")
            log(f"[local_vlm] group {index}/{total} (#{result['group_id']}): {label}")
            results.append({k: result.get(k) for k in (
                "group_id", "status", "error", "http_status", "model", "served_model",
                "model_calls", "latency_s", "usage")} | {"cache_hit": bool(result.get("cache_hit")),
                                                        "kept_previous": bool(result.get("kept_previous"))})
            if result.get("error") in ("offline", "auth") and index < total:
                log(f"[local_vlm] stopping: {FRIENDLY[result['error']]} (remaining groups are left for manual review)")
                break
    finally:
        data.close()
    summary = {"schema": SCHEMA, "endpoint": config["endpoint"], "requested_model": config["model"],
               "prompt_version": PROMPT_VERSION, "max_groups": int(max_groups),
               "model_calls": sum(r["model_calls"] or 0 for r in results if not r["cache_hit"]),
               "results": results, "skipped": selection["skipped"], "finished_at": int(time.time())}
    _write_json(sidecar_dir(output) / "last_run.json", summary)
    log(f"[local_vlm] done: {sum(r['status'] == 'ok' for r in results)} ok, "
        f"{sum(r['status'] == 'failed' for r in results)} failed, "
        f"{sum(r['cache_hit'] for r in results)} cached, {summary['model_calls']} model call(s)"
        + (f", {sum(r['kept_previous'] for r in results)} kept earlier suggestion"
           if any(r["kept_previous"] for r in results) else ""))
    return summary
