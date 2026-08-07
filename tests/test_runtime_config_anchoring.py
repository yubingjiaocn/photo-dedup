"""A generated runtime config must not change what the tunables file meant.

``runtime_config.written`` puts the derived config in a temporary directory. Two
things break silently if that move is careless, and both are pinned here:

* an explicitly named ``--config`` that does not exist would be *defaulted* by
  ``load_config``, tuning a run by values the user never wrote;
* ``features.scene_routing`` model/prompt paths are documented to resolve beside
  the config file, and :mod:`src.siglip_runtime` resolves them against
  ``Config.source.parent`` -- which for a generated config is the temp directory,
  so a relative path would drift away from the artifacts it named.

CLI shape and path derivation live in ``test_runtime_config.py``,
``test_rebuild_review.py`` and ``test_run_pipeline.py``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from src import rebuild_review, run_pipeline, runtime_config
from src.config import load_config
from src.siglip_router import prompt_bank_hash
from src.siglip_runtime import artifact_sha256, build_siglip_provider

BANK_PATH = Path(__file__).parents[1] / "research" / "siglip_prompt_bank_v1.yaml"


# --------------------------------------------------------------------------
# an explicit --config that does not exist is an error, never the defaults
# --------------------------------------------------------------------------

def test_base_config_path_rejects_a_missing_explicit_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="config file not found"):
        runtime_config.base_config_path(tmp_path / "absent.yaml")


def test_base_config_path_defaults_to_the_repository_config():
    assert runtime_config.base_config_path(None) == load_config().source


def test_build_rejects_a_missing_explicit_config(tmp_path):
    """Without this, load_config() silently returns the packaged defaults."""
    with pytest.raises(FileNotFoundError, match="config file not found"):
        runtime_config.build(
            tmp_path / "photos", tmp_path / "review",
            base_config=tmp_path / "absent.yaml",
        )


def test_build_rejects_a_directory_given_as_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="config file not found"):
        runtime_config.build(
            tmp_path / "photos", tmp_path / "review", base_config=tmp_path
        )


def test_run_pipeline_run_rejects_a_missing_explicit_config(tmp_path, monkeypatch):
    """Direct API callers get the same refusal the CLI gives, before any stage."""
    def explode(**_kwargs):
        raise AssertionError("no stage may run when --config is unusable")

    monkeypatch.setattr(run_pipeline.stage0_inventory, "run", explode)
    root = tmp_path / "photos"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="config file not found"):
        run_pipeline.run(
            str(root), str(tmp_path / "review"), backend="stub",
            config_path=str(tmp_path / "absent.yaml"),
        )


def test_rebuild_rejects_a_missing_explicit_config(tmp_path, monkeypatch):
    def explode(**_kwargs):
        raise AssertionError("no stage may run when --config is unusable")

    monkeypatch.setattr(rebuild_review.stage2_cluster, "run", explode)
    with pytest.raises(FileNotFoundError, match="config file not found"):
        rebuild_review.rebuild(
            str(tmp_path / "photos"), str(tmp_path / "review"),
            config_path=str(tmp_path / "absent.yaml"),
        )


def test_run_pipeline_cli_reports_a_missing_config_without_running_stages(
    tmp_path, monkeypatch, capsys
):
    def explode(**_kwargs):
        raise AssertionError("no stage may run when --config is unusable")

    monkeypatch.setattr(run_pipeline.stage0_inventory, "run", explode)
    root = tmp_path / "photos"
    root.mkdir()
    output = tmp_path / "review"

    status = run_pipeline.main([
        "--root", str(root), "--output", str(output),
        "--config", str(tmp_path / "absent.yaml"), "--backend", "stub", "--no-serve",
    ])
    assert status == 1
    assert "config not found" in capsys.readouterr().err
    # Refused before the output directory or the diagnostic log could appear.
    assert not output.exists()


# --------------------------------------------------------------------------
# scene_routing artifact paths stay anchored to the base config's directory
# --------------------------------------------------------------------------

def _scene_routing_base(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A base config declaring enabled scene_routing with *relative* paths."""
    base_dir = tmp_path / "conf"
    base_dir.mkdir()
    model = base_dir / "models" / "siglip"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"model_type":"siglip"}', encoding="utf-8")
    prompt = base_dir / "banks" / "bank.yaml"
    prompt.parent.mkdir()
    shutil.copyfile(BANK_PATH, prompt)

    base = base_dir / "tunables.yaml"
    base.write_text(yaml.safe_dump({
        "features": {
            "scene_routing": {
                "enabled": True,
                "mode": "shadow",
                "model": {
                    "path": "./models/siglip",
                    "revision": "local-revision-abc",
                    "sha256": artifact_sha256(model),
                },
                "prompt": {"path": "./banks/bank.yaml",
                           "sha256": prompt_bank_hash(prompt)},
                "runtime": {"device": "cpu", "precision": "float32", "batch_size": 8},
            }
        }
    }), encoding="utf-8")
    return base, model, prompt


def test_relative_scene_routing_paths_become_absolute_beside_the_base_config(tmp_path):
    base, model, prompt = _scene_routing_base(tmp_path)
    routing = runtime_config.build(
        tmp_path / "photos", tmp_path / "review", base_config=base
    )["features"]["scene_routing"]

    assert routing["model"]["path"] == str(model.resolve())
    assert routing["prompt"]["path"] == str(prompt.resolve())
    # The rest of the block is passed through untouched.
    assert routing["enabled"] is True
    assert routing["model"]["revision"] == "local-revision-abc"
    assert routing["runtime"]["batch_size"] == 8


def test_enabled_scene_routing_resolves_artifacts_from_a_temp_runtime_config(tmp_path):
    """The regression: the provider must load the model beside the base config.

    Unanchored, ``siglip_runtime`` looks under the temporary directory and fails
    with "must exist" instead of loading the artifacts the user declared.
    """
    base, model, prompt = _scene_routing_base(tmp_path)
    with runtime_config.written(
        tmp_path / "photos", tmp_path / "review", base_config=base
    ) as config_path:
        assert config_path.parent != base.parent  # genuinely a different directory
        cfg = load_config(config_path)
        provider = build_siglip_provider(cfg, backend=object(), processor=object())

        assert provider.runtime.model_path == model.resolve()
        assert provider.runtime.prompt_path == prompt.resolve()
        # Hashes still match, so the pinned identity survived the round trip.
        assert artifact_sha256(provider.runtime.model_path) == provider.runtime.model_sha256
        assert prompt_bank_hash(provider.runtime.prompt_path) == provider.runtime.prompt_sha256


def test_rebuild_hands_stages_an_anchored_scene_routing_config(tmp_path, monkeypatch):
    """A Stage 2+3 rebuild must not relocate the artifacts either."""
    base, model, prompt = _scene_routing_base(tmp_path)
    root = tmp_path / "photos"
    output = tmp_path / "review"
    seen = {}

    def capture(*, config_path, **_kwargs):
        routing = load_config(config_path).features["scene_routing"]
        seen["model"] = routing["model"]["path"]
        seen["prompt"] = routing["prompt"]["path"]
        return {}

    monkeypatch.setattr(rebuild_review, "verify_inventory", lambda *_a, **_k: 1)
    monkeypatch.setattr(rebuild_review.stage2_cluster, "run", capture)
    monkeypatch.setattr(rebuild_review.stage3_report, "run", lambda **_k: {})

    assert rebuild_review.rebuild(str(root), str(output), config_path=str(base)) == 0
    assert seen["model"] == str(model.resolve())
    assert seen["prompt"] == str(prompt.resolve())


def test_absolute_scene_routing_paths_are_left_exactly_as_written(tmp_path):
    absolute = (tmp_path / "abs" / "model").resolve()
    base = tmp_path / "tunables.yaml"
    base.write_text(yaml.safe_dump({
        "features": {"scene_routing": {"enabled": True,
                                       "model": {"path": str(absolute)}}}
    }), encoding="utf-8")

    routing = runtime_config.build(
        tmp_path / "photos", tmp_path / "review", base_config=base
    )["features"]["scene_routing"]
    assert routing["model"]["path"] == str(absolute)


def test_url_and_malformed_scene_routing_paths_are_not_rewritten(tmp_path):
    """Only relative local paths are anchored; bad values stay rejectable as-is."""
    base = tmp_path / "tunables.yaml"
    base.write_text(yaml.safe_dump({
        "features": {
            "scene_routing": {
                "enabled": True,
                "model": {"path": "https://example.invalid/model",
                          "revision": "r", "sha256": "0" * 64},
                "prompt": {"path": "", "sha256": "0" * 64},
                "runtime": {"device": "cpu", "precision": "float32", "batch_size": 1},
            }
        }
    }), encoding="utf-8")

    routing = runtime_config.build(
        tmp_path / "photos", tmp_path / "review", base_config=base
    )["features"]["scene_routing"]
    assert routing["model"]["path"] == "https://example.invalid/model"
    assert routing["prompt"]["path"] == ""

    # A URL is still refused for being a URL, not silently turned into a path.
    with runtime_config.written(
        tmp_path / "photos", tmp_path / "review", base_config=base
    ) as config_path:
        with pytest.raises(ValueError, match="must be a local filesystem path"):
            build_siglip_provider(load_config(config_path), backend=object())


def test_missing_or_disabled_scene_routing_is_untouched(tmp_path):
    """The anchoring step must not invent keys for configs that omit the block."""
    base = tmp_path / "tunables.yaml"
    base.write_text("cluster:\n  dinov2_threshold: 0.9\n", encoding="utf-8")
    data = runtime_config.build(
        tmp_path / "photos", tmp_path / "review", base_config=base
    )
    default_routing = load_config().features["scene_routing"]
    routing = data["features"]["scene_routing"]
    assert routing["enabled"] == default_routing["enabled"]

    empty = tmp_path / "empty-routing.yaml"
    empty.write_text(
        yaml.safe_dump({"features": {"scene_routing": {"enabled": False}}}),
        encoding="utf-8",
    )
    built = runtime_config.build(
        tmp_path / "photos", tmp_path / "review", base_config=empty
    )["features"]["scene_routing"]
    assert built["enabled"] is False


def test_packaged_config_scene_routing_paths_are_anchored_into_the_repo(tmp_path):
    """The shipped config.yaml declares ./models and ./research relative paths."""
    repo = Path(__file__).parents[1]
    routing = runtime_config.build(
        tmp_path / "photos", tmp_path / "review"
    )["features"]["scene_routing"]

    for section in ("model", "prompt"):
        value = Path(routing[section]["path"])
        assert value.is_absolute()
        assert value.is_relative_to(repo)
