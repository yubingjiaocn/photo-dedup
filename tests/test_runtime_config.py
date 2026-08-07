r"""The shared ``--root``/``--output`` -> paths mapping used by both CLIs.

This module is the single place that answers "which database does this run use?".
The tests therefore gate the *invariant*, not the implementation: for any
``(root, output)``, both entry points must end up with

    paths.root       = ROOT
    paths.db         = OUTPUT/inventory.sqlite
    paths.output_dir = OUTPUT

regardless of what a ``--config`` says, and :func:`src.config.load_config` must
resolve the written file back to exactly those three paths (that is what the
stages will do).
"""

from pathlib import Path

import pytest
import yaml

from src import runtime_config
from src.config import load_config


def test_db_filename_is_fixed():
    """One output directory owns one inventory filename, in both commands."""
    assert runtime_config.DB_FILENAME == "inventory.sqlite"
    assert runtime_config.db_path_for("/tmp/out") == Path("/tmp/out/inventory.sqlite")


def test_resolve_paths_makes_absolute_and_derives_db(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "photos").mkdir()
    root, output, db = runtime_config.resolve_paths("photos", "out")
    assert root == (tmp_path / "photos").resolve()
    assert output == (tmp_path / "out").resolve()
    assert db == (tmp_path / "out").resolve() / "inventory.sqlite"


def test_build_overrides_paths_from_cli(tmp_path):
    config = runtime_config.build(tmp_path / "photos", tmp_path / "out")
    assert config["paths"]["root"] == str((tmp_path / "photos").resolve())
    assert config["paths"]["db"] == str((tmp_path / "out").resolve() / "inventory.sqlite")
    assert config["paths"]["output_dir"] == str((tmp_path / "out").resolve())


def test_build_keeps_base_config_tunables_but_replaces_its_paths(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({
        "paths": {"root": "/decoy/photos", "db": "/decoy/other.sqlite",
                  "output_dir": "/decoy/out", "trash": "/decoy/trash"},
        "cluster": {"dinov2_threshold": 0.955},
        "quality": {"weight_iqa": 0.71},
    }), encoding="utf-8")

    config = runtime_config.build(tmp_path / "photos", tmp_path / "out", base_config=base)

    assert config["paths"]["root"] == str((tmp_path / "photos").resolve())
    assert config["paths"]["db"] == str((tmp_path / "out").resolve() / "inventory.sqlite")
    assert config["paths"]["output_dir"] == str((tmp_path / "out").resolve())
    assert config["cluster"]["dinov2_threshold"] == 0.955
    assert config["quality"]["weight_iqa"] == 0.71
    # Non-derived paths are left alone: only the three the CLI decides are replaced.
    assert config["paths"]["trash"] == "/decoy/trash"


def test_build_leaves_stage1_knobs_alone_unless_asked(tmp_path):
    """A stage 2+3 rebuild must not restate how features were computed."""
    default = runtime_config.build(tmp_path / "photos", tmp_path / "out")
    repo_default = load_config().as_dict()["features"]
    assert default["features"]["backend"] == repo_default["backend"]
    assert default["features"]["thumbnails"] == repo_default["thumbnails"]


def test_build_applies_backend_and_thumb_px_when_given(tmp_path):
    config = runtime_config.build(
        tmp_path / "photos", tmp_path / "out", backend="stub", thumb_px=123
    )
    assert config["features"]["backend"] == "stub"
    assert config["features"]["thumbnails"]["enabled"] is True
    assert config["features"]["thumbnails"]["max_px"] == 123


def test_written_config_resolves_back_to_the_cli_paths(tmp_path):
    """What the stages read must be what the CLI meant -- verified, not assumed."""
    root, output = tmp_path / "photos", tmp_path / "out"
    with runtime_config.written(root, output) as path:
        cfg = load_config(path)
        assert cfg.root_path == root.resolve()
        assert cfg.declared_root == root.resolve()
        assert cfg.db_path == output.resolve() / "inventory.sqlite"
        assert cfg.output_dir == output.resolve()
        written_path = Path(path)
        assert written_path.is_file()
    # Temporary: the user's config.yaml is never touched, and nothing is left behind.
    assert not written_path.exists()


def test_written_config_handles_paths_with_spaces(tmp_path):
    root = tmp_path / "My Photos 2026"
    output = tmp_path / "photo review out"
    with runtime_config.written(root, output) as path:
        cfg = load_config(path)
        assert cfg.root_path == root.resolve()
        assert cfg.db_path == output.resolve() / "inventory.sqlite"
        assert cfg.output_dir == output.resolve()


def test_written_config_does_not_modify_the_repository_config(tmp_path):
    from src.config import DEFAULT_CONFIG_PATH
    before = DEFAULT_CONFIG_PATH.read_bytes()
    with runtime_config.written(tmp_path / "photos", tmp_path / "out"):
        pass
    assert DEFAULT_CONFIG_PATH.read_bytes() == before


def test_verify_rejects_a_config_pointing_elsewhere(tmp_path):
    """The guard that would catch a future regression in ``build``."""
    stray = tmp_path / "stray.yaml"
    stray.write_text(yaml.safe_dump({
        "paths": {"root": str(tmp_path / "photos"),
                  "db": str(tmp_path / "elsewhere" / "inventory.sqlite"),
                  "output_dir": str(tmp_path / "out")},
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="paths.db"):
        runtime_config.verify(stray, tmp_path / "photos", tmp_path / "out")


def test_verify_rejects_a_config_declaring_another_root(tmp_path):
    stray = tmp_path / "stray.yaml"
    stray.write_text(yaml.safe_dump({
        "paths": {"root": str(tmp_path / "other"),
                  "db": str(tmp_path / "out" / "inventory.sqlite"),
                  "output_dir": str(tmp_path / "out")},
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="declares root"):
        runtime_config.verify(stray, tmp_path / "photos", tmp_path / "out")
