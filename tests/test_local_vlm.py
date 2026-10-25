"""Local-trial OpenAI-compatible VLM add-on: config, transport, validation, sidecars."""

from __future__ import annotations

import http.server
import json
import threading

import pytest

from src import local_trial, local_vlm

ALIASES = ["P1", "P2", "P3"]


def _answer(**over):
    value = {"status": "recommend", "keep": ["P3", "P1"], "confidence": "medium",
             "reasons": [{"text": "表情不同", "photos": ["P1", "P3"]}]}
    value.update(over)
    return json.dumps(value, ensure_ascii=False)


class _Mock:
    """Loopback OpenAI-compatible stub recording every request it receives."""

    def __init__(self, status=200, content=None):
        self.requests = []
        mock = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                mock.requests.append({"path": self.path, "auth": self.headers.get("Authorization"),
                                      "body": body})
                if status != 200:
                    payload = json.dumps({"error": {"message": "bad key sk-SECRET"}}).encode()
                else:
                    payload = json.dumps({"model": "upstream-1", "choices": [
                        {"message": {"content": content or _answer()}}]}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def test_endpoint_identity_rules():
    assert local_vlm.endpoint_identity("http://127.0.0.1:4000/v1/") == "http://127.0.0.1:4000/v1"
    assert local_vlm.endpoint_identity("https://gw.example.com/v1") == "https://gw.example.com/v1"
    for bad in ("http://gw.example.com/v1", "http://u:p@127.0.0.1:4000/v1",
                "http://127.0.0.1:4000/v1?key=abc", "ftp://127.0.0.1/v1", ""):
        with pytest.raises(local_vlm.ConfigError):
            local_vlm.endpoint_identity(bad)
    with pytest.raises(local_vlm.ConfigError):
        local_vlm.make_config("http://127.0.0.1:4000/v1", None, environ={})
    # Any alias is accepted; no model whitelist.
    cfg = local_vlm.make_config(None, "whatever/alias-7", environ={
        local_vlm.BASE_URL_ENV: "http://localhost:4000/v1"})
    assert cfg["model"] == "whatever/alias-7" and cfg["endpoint"] == "http://localhost:4000/v1"


def test_validate_answer_strict():
    assert local_vlm.validate_answer(_answer(), ALIASES)["keep"] == ["P1", "P3"]
    assert local_vlm.validate_answer("```json\n" + _answer() + "\n```", ALIASES)["status"] == "recommend"
    abstain = _answer(status="abstain", keep=[], reasons=[])
    assert local_vlm.validate_answer(abstain, ALIASES)["keep"] == []
    bad = [
        _answer(keep=["P1", "P9"]),                       # foreign alias
        _answer(keep=["1234"]),                           # raw file id, not an alias
        _answer(keep=["P1", "P1"]),                       # duplicate
        _answer(keep=[]),                                 # recommend with nothing
        _answer(status="abstain"),                        # abstain with picks
        _answer(confidence="certain"),
        _answer(reasons=[{"text": "a", "photos": ["P1"]}] * 4),
        _answer(reasons=[{"text": "x" * 200, "photos": ["P1"]}]),
        _answer(reasons=[{"text": "ok", "photos": ["P7"]}]),
        _answer(extra=1),
        "好的：" + _answer(),
        "",
    ]
    for content in bad:
        with pytest.raises(ValueError):
            local_vlm.validate_answer(content, ALIASES)


def test_post_chat_success_auth_and_offline(monkeypatch):
    cfg = None
    mock = _Mock()
    try:
        cfg = local_vlm.make_config(mock.url, "alias", timeout_s=5)
        body = local_vlm.build_request(cfg, [b"\xff\xd8a", b"\xff\xd8b"])
        reply = local_vlm.post_chat(cfg, body, environ={local_vlm.KEY_ENV: "sk-SECRET"})
        assert reply["ok"] and reply["json"]["model"] == "upstream-1"
        sent = mock.requests[0]
        assert sent["path"] == "/v1/chat/completions" and sent["auth"] == "Bearer sk-SECRET"
        assert sent["body"]["response_format"] == {"type": "json_object"}
        texts = [c["text"] for c in sent["body"]["messages"][0]["content"] if c["type"] == "text"]
        assert texts[1:] == ["P1", "P2"]
        local_vlm.post_chat(cfg, body, environ={})
        assert mock.requests[1]["auth"] is None            # no key: no header at all
        no_rf = local_vlm.make_config(mock.url, "alias", response_format=False)
        assert "response_format" not in local_vlm.build_request(no_rf, [b"x"])
    finally:
        mock.close()
    denied = _Mock(status=401)
    try:
        cfg = local_vlm.make_config(denied.url, "alias", timeout_s=5)
        reply = local_vlm.post_chat(cfg, {"model": "alias"}, environ={local_vlm.KEY_ENV: "sk-SECRET"})
        assert reply["error"] == "auth" and "sk-SECRET" not in reply["detail"]
        assert len(denied.requests) == 1                    # no retry
    finally:
        denied.close()
    offline = local_vlm.post_chat(cfg, {"model": "alias"}, environ={})
    assert offline["ok"] is False and offline["error"] == "offline"


class _FakeData:
    def __init__(self, groups):
        self.groups = groups
        self._lock = threading.Lock()
        self._conn = None
        self.scope = None

    def group_ids(self):
        return sorted(self.groups)


def test_selection_caps_groups_and_never_truncates(monkeypatch):
    def rows(gid, n, same=False):
        return [{"group_id": gid, "file_id": gid * 100 + i,
                 "content_sha256": "s" if same else f"{gid}-{i}"} for i in range(n)]

    groups = {gid: rows(gid, 2) for gid in range(1, 26)}
    groups[30] = rows(30, 21)
    groups[31] = rows(31, 3, same=True)
    groups[32] = rows(32, 1)
    monkeypatch.setattr("src.db.group_page_by_id", lambda conn, gid, scope=None: groups[gid])
    picked = local_vlm._select(_FakeData(groups), local_vlm.MAX_GROUPS)
    assert len(picked["chosen"]) == 20 and all(len(g) == 2 for g in picked["chosen"])
    status = {s["group_id"]: s["status"] for s in picked["skipped"]}
    assert status[30] == "manual_too_large" and status[31] == "cv_only_identical"
    assert [g for g, s in status.items() if s == "not_selected_budget"] == [21, 22, 23, 24, 25]
    assert 32 not in status
    with pytest.raises(local_vlm.ConfigError):
        local_vlm.run(None, {"endpoint": "x", "model": "m"}, max_groups=21)


def test_sidecar_loader_rejects_stale_members(tmp_path):
    rows = [{"group_id": 5, "file_id": 1, "content_sha256": "a"},
            {"group_id": 5, "file_id": 2, "content_sha256": "b"}]
    members = local_vlm._members_of(rows)
    record = {"schema": local_vlm.SCHEMA, "group_id": 5, "status": "ok", "model": "m",
              "prompt_version": local_vlm.PROMPT_VERSION, "members": members,
              "answer": {"status": "recommend", "keep": ["P2"], "confidence": "high",
                         "reasons": [{"text": "清晰", "photos": ["P2"]}]}}
    local_vlm._write_json(local_vlm.sidecar_path(tmp_path, 5), record)
    view = local_vlm.load_for_group(tmp_path, rows)
    assert view["keep_file_ids"] == [2] and view["reasons"][0]["file_ids"] == [2]
    changed = [rows[0], {**rows[1], "content_sha256": "CHANGED"}]
    assert local_vlm.load_for_group(tmp_path, changed) is None
    local_vlm._write_json(local_vlm.sidecar_path(tmp_path, 5), {**record, "prompt_version": "old"})
    assert local_vlm.load_for_group(tmp_path, rows) is None
    local_vlm._write_json(local_vlm.sidecar_path(tmp_path, 5),
                          {**record, "answer": {**record["answer"], "keep": ["P3"]}})
    assert local_vlm.load_for_group(tmp_path, rows) is None


def test_cli_requires_explicit_vlm_config(tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    assert local_trial.main(["judge", "--output", str(out), "--vlm-base-url",
                             "http://127.0.0.1:4000/v1"]) == 2
    assert "model alias is required" in capsys.readouterr().err
    assert local_trial.main(["judge", "--output", str(out), "--vlm-base-url",
                             "http://127.0.0.1:4000/v1", "--vlm-model", "m",
                             "--vlm-max-groups", "21"]) == 2
    assert local_trial.main(["judge", "--output", str(out), "--vlm-base-url",
                             "http://127.0.0.1:4000/v1", "--vlm-model", "m"]) == 2
    assert "no finished review" in capsys.readouterr().err


def test_openai_request_does_not_force_temperature_for_arbitrary_models():
    from src import local_vlm
    cfg = local_vlm.make_config('http://127.0.0.1:4000/v1', 'user-vision-alias')
    request = local_vlm.build_request(cfg, [b'image'])
    assert 'temperature' not in request
    assert request['model'] == 'user-vision-alias'
