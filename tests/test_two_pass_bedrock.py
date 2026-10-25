"""Network-free tests; fake SDK responses never enter the real artifact tree."""

import copy
import json
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image

from scripts.probe_two_pass_review import fake_judgment, fake_map
from src import two_pass_review as tp

ROOT = Path(__file__).resolve().parents[1]
MODEL = "global.anthropic.claude-sonnet-5-5"
GROUP = "2026-07-25-G004"
RATES = {
    "input_per_million": 3,
    "output_per_million": 15,
    "basis": "unverified test estimate",
}


class Client:
    def __init__(self, packet):
        self.calls = []
        self.response = {
            "output": {"message": {"content": [{"text": json.dumps(packet)}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 100, "outputTokens": 50, "totalTokens": 150},
        }

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return copy.deepcopy(self.response)


@pytest.fixture
def setup_case():
    output = ROOT / "artifacts/two-pass-adapter-tests" / uuid4().hex
    output.mkdir(parents=True)
    rows, sources = [], {}
    for fid in (1, 2):
        path = output / f"source-{fid}.png"
        Image.new("RGB", (1600, 1200), "gray").save(path)
        rows.append(
            {
                "file_id": fid,
                "content_sha256": tp.sha(path.read_bytes()),
                "width": 1600,
                "height": 1200,
                "faces_json": [],
                "quality_meta": {},
            }
        )
        sources[fid] = path
    a = fake_map(rows, "stage")
    client = Client(a)
    adapter = tp.BedrockAdapter(output, model_id=MODEL, rates=RATES, client=client)
    images = {tp.identity(r): Image.open(sources[r["file_id"]]) for r in rows}
    pages = tp.render_pages(images, output)
    payload = tp.request(
        "A",
        MODEL,
        {"frames": list(images), "scene_hint": "multi_subject_or_stage"},
        pages,
        output,
    )
    return output, rows, sources, client, adapter, payload


def test_usage_hash_cost_no_retry_and_restart(setup_case):
    output, rows, _, client, adapter, payload = setup_case
    assert adapter.invoke(payload, group=GROUP, assets_dir=output) == fake_map(
        rows, "stage"
    )
    entry = adapter.calls[0]
    assert entry["estimated_cost_usd"] == 0.00105
    assert entry["usage"]["totalTokens"] == 150
    assert entry["request_sha256"] == tp.sha(
        (output / "pass-A-wire-request.json").read_bytes()
    )
    assert entry["response_sha256"] == tp.sha(
        (output / "pass-A-provider-response.json").read_bytes()
    )
    assert entry["latency_seconds"] >= 0
    restarted = tp.BedrockAdapter(output, model_id=MODEL, rates=RATES, client=client)
    with pytest.raises(tp.ReviewError, match="retry"):
        restarted.invoke(payload, group=GROUP, assets_dir=output)
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(stopReason="max_tokens"),
        lambda r: r.update(usage={}),
        lambda r: r["usage"].update(outputTokens=4097, totalTokens=4197),
        lambda r: r["usage"].update(inputTokens=True),
        lambda r: r["usage"].update(totalTokens=999),
        lambda r: r["usage"].update(cacheReadInputTokens=20),
        lambda r: r["output"]["message"].update(content=[{"text": "{}"}]),
        lambda r: r["output"]["message"].update(content=[{"text": "x" * 12001}]),
        lambda r: r["output"]["message"].update(content=[{"toolUse": {}}]),
    ],
)
def test_fail_closed_response(setup_case, mutation):
    output, _, _, client, adapter, payload = setup_case
    mutation(client.response)
    with pytest.raises(tp.ReviewError):
        adapter.invoke(payload, group=GROUP, assets_dir=output)
    assert len(client.calls) == 1
    assert adapter.calls[0]["state"] == "failed_no_retry"


def test_allowlist_and_hard_total_limit(setup_case):
    output, _, _, client, adapter, payload = setup_case
    assert tp.ALLOWED_BEDROCK_MODELS == {MODEL, "global.openai.gpt-5.6-terra"}
    with pytest.raises(tp.ReviewError, match="allowlist"):
        tp.BedrockAdapter(
            output,
            model_id="global.anthropic.claude-opus-5-5",
            rates=RATES,
            client=client,
        )
    adapter.calls = [{"group": "other", "pass": "A"}] * 4
    with pytest.raises(tp.ReviewError, match="budget"):
        adapter.invoke(payload, group=GROUP, assets_dir=output)
    assert not client.calls


def test_wire_budget_and_tampering_before_call(setup_case, monkeypatch):
    output, _, _, client, adapter, payload = setup_case
    payload["max_output_tokens"] += 1
    with pytest.raises(tp.ReviewError, match="contract"):
        adapter.invoke(payload, group=GROUP, assets_dir=output)
    payload["max_output_tokens"] -= 1
    monkeypatch.setattr(tp, "PRIVACY_INSTRUCTION", "x" * 6001)
    with pytest.raises(tp.ReviewError, match="wire"):
        adapter.invoke(payload, group=GROUP, assets_dir=output)
    assert not client.calls


def test_missing_key_no_cv_no_b(setup_case):
    output, rows, sources, client, adapter, _ = setup_case
    a = fake_map(rows, "stage")
    a["coverage"][0].update(state="missing", uncertainty="missing_key")
    client.response["output"]["message"]["content"][0]["text"] = json.dumps(a)
    result = tp.run_real(rows, sources, output / "case", adapter=adapter, group=GROUP)
    assert result["judgment"]["decision"] == "attention_only"
    assert result["pass_B_status"] == "not_run"
    assert len(client.calls) == 1
    assert json.loads((output / "case/L1.json").read_text())["status"] == "not_produced"


def test_privacy_and_duplicate_json_keys():
    for text in ("Person named Alice.", "Email me a@b.com", "his wife", "姓名"):
        with pytest.raises(tp.ReviewError, match="privacy"):
            tp.privacy_check({"discriminators": [{"text": text}]})
    for text in ('{"x":1,"x":2}', '{"x":NaN}'):
        with pytest.raises(tp.ReviewError):
            tp.strict_json(text)


def test_two_pass_real_transport_mechanics(setup_case):
    output, rows, sources, client, adapter, _ = setup_case
    a = fake_map(rows, "stage")
    original = client.converse

    def converse(**kwargs):
        if client.calls:
            client.response["output"]["message"]["content"][0]["text"] = json.dumps(
                fake_judgment(a)
            )
        return original(**kwargs)

    client.converse = converse
    result = tp.run_real(rows, sources, output / "case", adapter=adapter, group=GROUP)
    assert result["pass_B_status"] == "passed", result["blockers"]
    assert result["external_model_calls"] == 2
    assert result["judgment"]["decision"] == "select"
    assert not result["production_enabled"]


def test_sdk_total_max_attempts_is_one(setup_case, monkeypatch):
    import boto3

    output, _, _, client, adapter, payload = setup_case
    captured = {}

    def make_client(service, **kwargs):
        captured.update(kwargs)
        assert service == "bedrock-runtime"
        return client

    monkeypatch.setattr(boto3, "client", make_client)
    adapter.client = None
    adapter.invoke(payload, group=GROUP, assets_dir=output)
    assert captured["config"].retries["total_max_attempts"] == 1
    assert "max_attempts" not in captured["config"].retries


def test_transport_exception_is_consumed_no_retry(setup_case):
    output, _, _, client, adapter, payload = setup_case

    def failed(**kwargs):
        client.calls.append(kwargs)
        raise TimeoutError("uncertain delivery")

    client.converse = failed
    with pytest.raises(tp.ReviewError, match="TimeoutError"):
        adapter.invoke(payload, group=GROUP, assets_dir=output)
    with pytest.raises(tp.ReviewError, match="retry"):
        adapter.invoke(payload, group=GROUP, assets_dir=output)
    assert len(client.calls) == 1
    assert adapter.calls[0]["usage"] is None


def test_persisted_v1_a_verification_rejects_tampering(tmp_path):
    import shutil
    import sqlite3

    from scripts.revalidate_two_pass_real import verify_saved_a
    from src import db

    original = ROOT / "artifacts/two-pass-real-pilot"
    if not (original / "archive/v1").exists():
        pytest.skip("private original pilot not present")
    clone = tmp_path / "pilot"
    shutil.copytree(original / "archive/v1", clone / "archive/v1")
    shutil.copy2(original / "archive/v1/call-ledger.json", clone / "call-ledger.json")
    shutil.copy2(
        original / "implementation-manifest.json",
        clone / "implementation-manifest.json",
    )
    shutil.copytree(original / GROUP, clone / GROUP)
    inventory = Path(
        "/home/ubuntu/photo-dedup-eval/astra-regime-validation-20260910/development/2026-07-25/inventory.sqlite"
    )
    if not inventory.exists():
        pytest.skip("private frozen database not present")
    with sqlite3.connect(f"file:{inventory}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.group_page_by_id(conn, 4)]
        sources = {
            r["file_id"]: Path(
                conn.execute(
                    "SELECT path FROM files WHERE id=?", (r["file_id"],)
                ).fetchone()[0]
            )
            for r in rows
        }
    packet, validation, _, _, _ = verify_saved_a(clone, GROUP, rows, sources)
    assert validation["eligible_tracks"] == ["T1"]
    assert validation["decision_ceiling"] == "attention_only"
    for path in (
        clone / GROUP / "pass-A-raw.json",
        clone / GROUP / "page-1.jpg",
        clone / "archive/v1/implementation/src/two_pass_review.py",
        clone / "archive/v1/implementation/schemas/two-pass-subject-map-v1.schema.json",
    ):
        data = path.read_bytes()
        path.write_bytes(data + b" ")
        with pytest.raises(tp.ReviewError):
            verify_saved_a(clone, GROUP, rows, sources)
        path.write_bytes(data)
    wrong = copy.deepcopy(rows)
    wrong[0]["content_sha256"] = "f" * 64
    with pytest.raises(tp.ReviewError, match="identity"):
        verify_saved_a(clone, GROUP, wrong, sources)
    assert packet["scene"] == "stage"
