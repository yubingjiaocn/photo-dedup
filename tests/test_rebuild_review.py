r"""``rebuild_review`` CLI shape + fail-closed gates.

Two properties are gated here, and they are the reason the command exists:

* **One path mapping.** ``--root``/``--output`` alone decide ``paths.root``,
  ``paths.db`` and ``paths.output_dir``, identically to ``run_pipeline``. A
  ``--config`` may only contribute tunables; if it could contribute paths, a
  rebuild could cluster a *different* database than the run it rebuilds, which is
  exactly the failure the removed ``run-config.yaml`` form allowed.
* **Fail closed, before writing.** Missing DB, wrong schema, unfinished Stage 1,
  no usable features, or an inventory bound to another root: refuse, and refuse
  without creating or migrating anything.

Stage 0/1 are never invoked; the tests assert that too, since a rebuild that
re-read the library would defeat its whole purpose.
"""

import sqlite3
import numpy as np
import pytest
import yaml

from src import db, root_scope, rebuild_review, runtime_config
from src.rebuild_review import main, rebuild


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def _valid_inventory(output, photos_root):
    """A minimal but genuinely valid finished Stage 0/1 inventory in ``output``."""
    output.mkdir(parents=True, exist_ok=True)
    db_path = runtime_config.db_path_for(output)
    conn = db.open_db(db_path)
    root_scope.bind(conn, str(photos_root), db_path=str(db_path))
    conn.execute(
        "INSERT INTO files (path, exif_timestamp, size_bytes, width, height, file_kind) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (str(photos_root / "test.jpg"), 1000000, 100, 1920, 1080, "jpg"),
    )
    conn.commit()
    emb = np.random.randn(768).astype(np.float16)
    emb /= np.linalg.norm(emb)
    conn.execute(
        "INSERT INTO features (file_id, phash, content_sha256, dinov2_embedding, "
        "quality_score, quality_meta, faces_json, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (1, (0x1234567812345678).to_bytes(8, "big"), "a" * 64, emb.tobytes(), 50.0,
         '{"exposure":{"clip_hi":0.0,"clip_lo":0.0,"blob_hi":0.0,"blob_lo":0.0,'
         '"anchor_mass":0.5,"entropy_nonclip":5.0,"mass_usable":0.6}}',
         "[]", "done"),
    )
    db.set_meta(conn, "stage1_done_at", "1234567890")
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def library(tmp_path):
    """``(root, output)`` for a valid, rebuildable run."""
    root = tmp_path / "photos"
    root.mkdir()
    output = tmp_path / "review"
    _valid_inventory(output, root)
    return root, output


# --------------------------------------------------------------------------
# CLI surface: --root and --output are required, --config is optional
# --------------------------------------------------------------------------

@pytest.mark.parametrize("argv", [
    [],
    ["--root", "R"],
    ["--output", "O"],
    ["--config", "c.yaml"],
    ["--config", "c.yaml", "--root", "R"],
    ["--config", "c.yaml", "--output", "O"],
])
def test_missing_root_or_output_is_a_usage_error(argv, capsys):
    """Both paths are mandatory; a config alone can no longer drive a rebuild."""
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "required" in err


def test_config_only_invocation_is_rejected(tmp_path, capsys):
    """The removed ``--config run-config.yaml --root ...`` form must not work.

    It used to be the *documented* way to rebuild. Accepting it silently would
    resolve paths from the file again, so it has to be a hard usage error.
    """
    run_config = tmp_path / "run-config.yaml"
    run_config.write_text("paths:\n  db: /tmp/x.sqlite\n", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        main(["--config", str(run_config), "--root", str(tmp_path)])
    assert excinfo.value.code == 2
    assert "--output" in capsys.readouterr().err


def test_help_documents_the_new_shape(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    text = capsys.readouterr().out
    assert "--root" in text and "--output" in text and "--config" in text
    assert "run-config" not in text


def test_missing_config_file_reports_error_without_running_stages(tmp_path, monkeypatch, capsys):
    def explode(**_kwargs):
        raise AssertionError("no stage may run when --config is unusable")

    monkeypatch.setattr(rebuild_review.stage2_cluster, "run", explode)
    status = main([
        "--root", str(tmp_path), "--output", str(tmp_path / "out"),
        "--config", str(tmp_path / "nope.yaml"),
    ])
    assert status == 1
    assert "config not found" in capsys.readouterr().err


# --------------------------------------------------------------------------
# path derivation: --root/--output win, config supplies tunables only
# --------------------------------------------------------------------------

def _capture_stage_configs(monkeypatch):
    """Record the config each stage is handed, and prove Stage 0/1 stay unused."""
    seen = []

    def record(name):
        def fake(*_args, **kwargs):
            data = yaml.safe_load(
                open(kwargs["config_path"], encoding="utf-8").read()
            )
            seen.append((name, kwargs, data))
            return {}

        return fake

    monkeypatch.setattr(rebuild_review.stage2_cluster, "run", record("stage2"))
    monkeypatch.setattr(rebuild_review.stage3_report, "run", record("stage3"))
    return seen


def test_paths_are_derived_from_root_and_output(library, monkeypatch):
    """The three paths every stage needs come from the two CLI arguments."""
    root, output = library
    seen = _capture_stage_configs(monkeypatch)

    assert rebuild(str(root), str(output)) == 0

    assert [name for name, _, _ in seen] == ["stage2", "stage3"]
    for _name, kwargs, data in seen:
        assert data["paths"]["root"] == str(root.resolve())
        assert data["paths"]["db"] == str(output.resolve() / "inventory.sqlite")
        assert data["paths"]["output_dir"] == str(output.resolve())
        # Every stage is scoped to the CLI root, not to whatever the DB is bound to.
        assert kwargs["root_override"] == str(root.resolve())


def test_config_paths_are_ignored_but_its_tunables_are_passed_through(library, monkeypatch):
    """A ``--config`` is a tunables base; its ``paths`` never win over the CLI."""
    root, output = library
    decoy = output.parent / "decoy"
    tunables = output.parent / "tunables.yaml"
    tunables.write_text(yaml.safe_dump({
        "paths": {
            "root": str(decoy / "photos"),
            "db": str(decoy / "other.sqlite"),
            "output_dir": str(decoy),
        },
        "cluster": {"dinov2_threshold": 0.977, "burst_window_seconds": 11},
        "quality": {"weight_iqa": 0.42},
        "decision": {"profile": "aggressive"},
    }), encoding="utf-8")
    seen = _capture_stage_configs(monkeypatch)

    assert rebuild(str(root), str(output), config_path=str(tunables)) == 0

    for _name, _kwargs, data in seen:
        # paths: overridden
        assert data["paths"]["root"] == str(root.resolve())
        assert data["paths"]["db"] == str(output.resolve() / "inventory.sqlite")
        assert data["paths"]["output_dir"] == str(output.resolve())
        assert str(decoy) not in yaml.safe_dump(data["paths"])
        # tunables: preserved exactly
        assert data["cluster"]["dinov2_threshold"] == 0.977
        assert data["cluster"]["burst_window_seconds"] == 11
        assert data["quality"]["weight_iqa"] == 0.42
        assert data["decision"]["profile"] == "aggressive"
    assert not decoy.exists()


def test_rebuild_does_not_restate_stage1_parameters(library, monkeypatch):
    """A rebuild must not rewrite how the existing feature cache was computed."""
    root, output = library
    tunables = output.parent / "tunables.yaml"
    tunables.write_text(yaml.safe_dump({
        "features": {"backend": "stub", "thumbnails": {"enabled": False, "max_px": 64}},
    }), encoding="utf-8")
    seen = _capture_stage_configs(monkeypatch)

    rebuild(str(root), str(output), config_path=str(tunables))

    for _name, _kwargs, data in seen:
        assert data["features"]["backend"] == "stub"
        thumbs = data["features"]["thumbnails"]
        # Verbatim from the config: a rebuild adds no backend/thumbnail override of
        # its own (unlike run_pipeline, which owns Stage 1). Defaults still fill in
        # keys the config omitted (jpeg_quality) -- that is load_config, not us.
        assert thumbs["enabled"] is False
        assert thumbs["max_px"] == 64


def test_paths_with_spaces_survive_round_trip(tmp_path, monkeypatch):
    """Windows users have ``C:\\photo review`` paths; quoting must be enough."""
    root = tmp_path / "My Photos 2026"
    root.mkdir()
    output = tmp_path / "photo review out"
    _valid_inventory(output, root)
    seen = _capture_stage_configs(monkeypatch)

    assert rebuild(str(root), str(output)) == 0

    for _name, _kwargs, data in seen:
        assert data["paths"]["root"] == str(root.resolve())
        assert data["paths"]["db"] == str(output.resolve() / "inventory.sqlite")
        assert data["paths"]["output_dir"] == str(output.resolve())


def test_relative_root_and_output_are_resolved(library, monkeypatch):
    root, output = library
    monkeypatch.chdir(output.parent)
    seen = _capture_stage_configs(monkeypatch)

    assert rebuild(root.name, output.name) == 0

    for _name, _kwargs, data in seen:
        assert data["paths"]["db"] == str(output.resolve() / "inventory.sqlite")


# --------------------------------------------------------------------------
# fail closed
# --------------------------------------------------------------------------

def test_missing_db_fails_closed(tmp_path, monkeypatch):
    """No inventory in ``--output`` -> refuse; Stage 2 must never create one."""
    root = tmp_path / "photos"
    root.mkdir()
    output = tmp_path / "empty-output"
    seen = _capture_stage_configs(monkeypatch)

    with pytest.raises(FileNotFoundError, match="inventory DB not found"):
        rebuild(str(root), str(output))

    assert seen == []
    assert not runtime_config.db_path_for(output).exists()


def test_missing_schema_fails_closed(tmp_path, monkeypatch):
    root = tmp_path / "photos"
    root.mkdir()
    output = tmp_path / "review"
    output.mkdir()
    conn = sqlite3.connect(str(runtime_config.db_path_for(output)))
    conn.execute("CREATE TABLE dummy (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    seen = _capture_stage_configs(monkeypatch)

    with pytest.raises(ValueError, match="missing essential tables"):
        rebuild(str(root), str(output))
    assert seen == []


def test_stage1_not_done_fails_closed(tmp_path, monkeypatch):
    root = tmp_path / "photos"
    root.mkdir()
    output = tmp_path / "review"
    output.mkdir()
    db_path = runtime_config.db_path_for(output)
    conn = db.open_db(db_path)
    root_scope.bind(conn, str(root), db_path=str(db_path))
    conn.commit()
    conn.close()
    seen = _capture_stage_configs(monkeypatch)

    with pytest.raises(ValueError, match="has no stage1_done_at"):
        rebuild(str(root), str(output))
    assert seen == []


def test_no_features_fails_closed(tmp_path, monkeypatch):
    root = tmp_path / "photos"
    root.mkdir()
    output = tmp_path / "review"
    output.mkdir()
    db_path = runtime_config.db_path_for(output)
    conn = db.open_db(db_path)
    root_scope.bind(conn, str(root), db_path=str(db_path))
    db.set_meta(conn, "stage1_done_at", "1234567890")
    conn.commit()
    conn.close()
    seen = _capture_stage_configs(monkeypatch)

    with pytest.raises(ValueError, match="has 0 features rows"):
        rebuild(str(root), str(output))
    assert seen == []


def test_root_mismatch_fails_closed(library, monkeypatch, tmp_path):
    """An output directory belongs to one root; rebuilding it as another refuses.

    This gate only became reachable when ``--root``/``--output`` replaced the
    config: previously a hand-written ``run-config.yaml`` could name any db.
    """
    _root, output = library
    other = tmp_path / "other-photos"
    other.mkdir()
    seen = _capture_stage_configs(monkeypatch)

    with pytest.raises(root_scope.ParameterError):
        rebuild(str(other), str(output))
    assert seen == []


def test_main_reports_failures_as_status_not_traceback(tmp_path, capsys):
    root = tmp_path / "photos"
    root.mkdir()
    status = main(["--root", str(root), "--output", str(tmp_path / "missing-out")])
    assert status == 1
    assert "inventory DB not found" in capsys.readouterr().err


def test_main_reports_root_mismatch_with_distinct_status(library, tmp_path, capsys):
    _root, output = library
    other = tmp_path / "other-photos"
    other.mkdir()
    status = main(["--root", str(other), "--output", str(output)])
    assert status == 2
    assert capsys.readouterr().err.strip() != ""


# --------------------------------------------------------------------------
# end to end: real Stage 2 + Stage 3 against a real inventory
# --------------------------------------------------------------------------

def test_valid_inventory_rebuilds_stage2_and_stage3(library, monkeypatch):
    """The happy path writes the review outputs and reads no photos."""
    root, output = library
    import src.stage0_inventory as stage0
    import src.stage1_features as stage1
    monkeypatch.setattr(stage0, "run", lambda **_k: (_ for _ in ()).throw(
        AssertionError("rebuild must not run Stage 0")))
    monkeypatch.setattr(stage1, "run", lambda **_k: (_ for _ in ()).throw(
        AssertionError("rebuild must not run Stage 1")))

    assert rebuild(str(root), str(output)) == 0

    assert (output / "review.html").exists()
    assert (output / "delete_local.txt").exists()


def test_main_end_to_end_matches_rebuild(library):
    root, output = library
    assert main(["--root", str(root), "--output", str(output)]) == 0
    assert (output / "review.html").exists()


def test_verify_inventory_returns_feature_count(library):
    root, output = library
    count = rebuild_review.verify_inventory(runtime_config.db_path_for(output), root)
    assert count == 1
