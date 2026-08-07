"""Shared ``--root``/``--output`` runtime config builder for the CLI entry points.

Why this module exists
----------------------
``run_pipeline`` (stages 0-3) and ``rebuild_review`` (stages 2+3) both operate on
one photo root and one output directory, and both must derive *exactly the same*
three paths from them::

    paths.root       = ROOT
    paths.db         = OUTPUT/inventory.sqlite
    paths.output_dir = OUTPUT

They used to assemble that themselves -- and ``rebuild_review`` did not assemble
it at all, it demanded a hand-maintained ``run-config.yaml`` whose ``paths.db``
had to happen to match the directory ``run_pipeline`` had written. Two path
assemblies for one invariant is how a rebuild ends up clustering a *different*
database than the run it was meant to rebuild. So the assembly lives here once.

``config.yaml`` keeps exactly one job in both commands: the **tunables base**
(thresholds, backend defaults, model settings). Any ``paths`` it declares are
overridden by ``ROOT``/``OUTPUT``, because the command line is the thing the user
just typed and therefore the thing they mean.

The builder never writes into the user's ``config.yaml``; :func:`written` puts the
derived config in a temporary file that lives only as long as the stages run.
That temporary location is also why :func:`build` re-anchors the declared
``features.scene_routing`` artifact paths: those are documented to resolve
*beside the config file*, and :mod:`src.siglip_runtime` resolves them against
``Config.source.parent``. Left relative, a run would look for the model under the
temporary directory instead of beside the base config.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import yaml

from .config import DEFAULT_CONFIG_PATH, load_config

#: The one inventory filename an output directory may hold. Stage 0/1 write it,
#: stages 2/3 (and a rebuild) must read that same file or refuse to run.
DB_FILENAME = "inventory.sqlite"

PathLike = str | os.PathLike


def db_path_for(output: PathLike) -> Path:
    """The database an output directory owns: ``OUTPUT/inventory.sqlite``."""
    return Path(output) / DB_FILENAME


def resolve_paths(root: PathLike, output: PathLike) -> tuple[Path, Path, Path]:
    """Return absolute ``(root, output, db)`` for a ``--root``/``--output`` pair."""
    root_path = Path(root).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    return root_path, output_path, db_path_for(output_path)


def base_config_path(base_config: Optional[PathLike]) -> Path:
    """The tunables file that will be read, explicit or default.

    An explicitly named file that does not exist is an error, not a reason to
    fall back to defaults: :func:`src.config.load_config` would silently return
    the packaged defaults, and the user would get a run tuned by values they
    never wrote.
    """
    if base_config is None:
        return DEFAULT_CONFIG_PATH
    path = Path(base_config).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    return path


def _anchor_scene_routing(data: Dict[str, Any], base_dir: Path) -> None:
    """Make declared ``scene_routing`` model/prompt paths absolute.

    ``config.yaml`` documents these as resolving beside the config file, and
    :mod:`src.siglip_runtime` implements that with ``Config.source.parent`` -- which
    for a generated runtime config is a temporary directory. Only a declared,
    relative, local path is rewritten: a URL or a malformed value is left exactly
    as written so ``siglip_runtime`` still rejects it for what it is.
    """
    features = data.get("features")
    if not isinstance(features, dict):
        return
    routing = features.get("scene_routing")
    if not isinstance(routing, dict):
        return
    routing = dict(routing)
    changed = False
    for section in ("model", "prompt"):
        block = routing.get(section)
        if not isinstance(block, dict):
            continue
        value = block.get("path")
        if not isinstance(value, str) or not value.strip() or "://" in value:
            continue
        path = Path(value)
        if path.is_absolute():
            continue
        block = dict(block)
        block["path"] = str((base_dir / path).resolve())
        routing[section] = block
        changed = True
    if changed:
        features["scene_routing"] = routing


def build(
    root: PathLike,
    output: PathLike,
    *,
    base_config: Optional[PathLike] = None,
    backend: Optional[str] = None,
    thumb_px: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the runtime config dict for one ``(root, output)`` pair.

    ``base_config`` supplies tunables only (default: the repository
    ``config.yaml``); its ``paths`` are replaced, and an explicitly named file that
    does not exist raises :class:`FileNotFoundError` rather than silently falling
    back to defaults. ``backend``/``thumb_px`` are Stage 1 knobs and are only
    touched when a caller passes them, so a stage 2+3 rebuild cannot silently
    rewrite the settings the existing feature cache was computed with.
    """
    root_path, output_path, db_path = resolve_paths(root, output)
    source = base_config_path(base_config)
    data = load_config(source).as_dict()
    data["paths"] = dict(data.get("paths") or {})
    data["paths"].update(
        root=str(root_path),
        db=str(db_path),
        output_dir=str(output_path),
    )
    if backend is not None or thumb_px is not None:
        data["features"] = dict(data.get("features") or {})
    if backend is not None:
        data["features"]["backend"] = backend
    if thumb_px is not None:
        thumb_cfg = dict(data["features"].get("thumbnails") or {})
        thumb_cfg["enabled"] = True
        thumb_cfg["max_px"] = int(thumb_px)
        data["features"]["thumbnails"] = thumb_cfg
    # The generated config lives in a temp dir; keep artifact paths pointing at
    # the files they named beside the base config.
    data["features"] = dict(data.get("features") or {})
    _anchor_scene_routing(data, source.resolve().parent)
    return data


def verify(config_path: PathLike, root: PathLike, output: PathLike) -> None:
    """Re-read the written config and assert it resolves to the CLI's paths.

    Cheap, and it closes the gap the old two-config setup left open: a stage must
    never be handed a database other than ``OUTPUT/inventory.sqlite``, and the
    only way to know what a stage will resolve is to resolve it the same way the
    stage does (:func:`src.config.load_config`).
    """
    root_path, output_path, db_path = resolve_paths(root, output)
    cfg = load_config(config_path)
    if cfg.db_path != db_path:
        raise RuntimeError(
            f"runtime config resolves paths.db to {cfg.db_path}, expected {db_path}"
        )
    if cfg.output_dir != output_path:
        raise RuntimeError(
            f"runtime config resolves paths.output_dir to {cfg.output_dir}, "
            f"expected {output_path}"
        )
    declared = cfg.declared_root
    if declared is None or Path(declared) != root_path:
        raise RuntimeError(
            f"runtime config declares root {declared}, expected {root_path}"
        )


@contextlib.contextmanager
def written(
    root: PathLike,
    output: PathLike,
    *,
    base_config: Optional[PathLike] = None,
    backend: Optional[str] = None,
    thumb_px: Optional[int] = None,
) -> Iterator[Path]:
    """Yield a temporary runtime config path for ``(root, output)``.

    Verified before it is yielded, and deleted afterwards.
    """
    config = build(
        root, output, base_config=base_config, backend=backend, thumb_px=thumb_px
    )
    with tempfile.TemporaryDirectory(prefix="photo-dedup-") as temp_dir:
        config_path = Path(temp_dir) / "runtime-config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        verify(config_path, root, output)
        yield config_path
