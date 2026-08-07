"""The compiled review front-end, and how it reaches an output directory.

The review UI is a Preact application under ``frontend/`` built by Vite. Its
compiled output -- ``frontend/dist/`` -- is committed to the repository and is
the *runtime source of truth*: Stage 3 copies it into the review output
directory, and the local server serves it. **No Node.js is needed to run the
pipeline**; a build is only needed to *change* the UI (see ``frontend/README``).

Two things live here so both Stage 3 (which writes the output) and the review
server (which serves and allow-lists it) agree on one contract:

* :func:`dist_dir` -- where the committed build is in a source checkout.
* :func:`install_into` -- copy the build into an output directory and write
  ``review_assets.json``, a manifest naming exactly which files may be served
  statically. The server reads that manifest instead of trusting the directory
  listing, so the output folder stays a fail-closed allow-list: a stray file
  dropped beside the manifest is not reachable just because it exists.

The manifest is a small JSON object::

    {"schema": 1, "entry": "review.html", "assets": ["assets/review-<hash>.js", ...]}

``entry`` is always ``review.html``; ``assets`` are the hashed chunks Vite
emitted. Everything the browser loads is one of these, so the allow-list is
exactly the manifest's ``entry`` plus its ``assets``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

MANIFEST_NAME = "review_assets.json"
ENTRY_NAME = "review.html"
ASSETS_DIRNAME = "assets"
_SCHEMA = 1


def dist_dir() -> Path:
    """Absolute path of the committed Vite build in this checkout.

    ``src/`` and ``frontend/`` are siblings under the repository root, so the
    build is ``<repo>/frontend/dist``. Resolved, not guessed, so a symlinked or
    relocated checkout still finds it.
    """
    return (Path(__file__).resolve().parents[1] / "frontend" / "dist").resolve()


def build_exists(dist: Optional[Path] = None) -> bool:
    """True when a usable committed build is present (entry HTML + assets dir)."""
    root = dist_dir() if dist is None else Path(dist)
    return (root / ENTRY_NAME).is_file() and (root / ASSETS_DIRNAME).is_dir()


def _collect_assets(dist: Path) -> List[str]:
    """Relative POSIX paths of every emitted asset under ``assets/``, sorted.

    Only regular files inside the single ``assets/`` directory are listed; the
    build never nests deeper, and refusing anything else keeps the manifest a
    faithful description of what Vite produced.
    """
    assets_dir = dist / ASSETS_DIRNAME
    names: List[str] = []
    for path in sorted(assets_dir.rglob("*")):
        if path.is_file():
            names.append(path.relative_to(dist).as_posix())
    return names


def manifest_for(dist: Optional[Path] = None) -> Dict[str, object]:
    """Build the manifest object describing the committed build."""
    root = dist_dir() if dist is None else Path(dist)
    if not build_exists(root):
        raise FileNotFoundError(
            f"review front-end build not found at {root}. It ships committed; "
            f"if it is missing, run `npm ci && npm run build` in frontend/."
        )
    return {"schema": _SCHEMA, "entry": ENTRY_NAME, "assets": _collect_assets(root)}


def install_into(output_dir: Path, dist: Optional[Path] = None) -> Dict[str, object]:
    """Copy the committed build into ``output_dir`` and write the manifest.

    Returns the manifest that was written. The entry HTML lands as
    ``output/review.html`` (unchanged URL) and the hashed chunks as
    ``output/assets/*``. The previous build's ``assets/`` is cleared first so a
    rebuild never leaves an orphaned old chunk that the manifest no longer names.
    """
    root = dist_dir() if dist is None else Path(dist)
    manifest = manifest_for(root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Replace assets/ wholesale, then copy the entry HTML.
    dest_assets = output_dir / ASSETS_DIRNAME
    if dest_assets.exists():
        shutil.rmtree(dest_assets)
    shutil.copytree(root / ASSETS_DIRNAME, dest_assets)
    shutil.copy2(root / ENTRY_NAME, output_dir / ENTRY_NAME)

    with (output_dir / MANIFEST_NAME).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


def load_manifest(output_dir: Path) -> Optional[Dict[str, object]]:
    """Read a previously installed manifest, or ``None`` if it is absent/broken."""
    path = Path(output_dir) / MANIFEST_NAME
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or value.get("schema") != _SCHEMA:
        return None
    if value.get("entry") != ENTRY_NAME or not isinstance(value.get("assets"), list):
        return None
    return value


def allowed_static_paths(output_dir: Path) -> frozenset[str]:
    """The exact set of URL paths the server may serve statically.

    Reads ``review_assets.json`` and returns ``{entry} | set(assets)``. When the
    manifest is missing or unreadable, only the entry HTML is allowed -- the
    fail-closed default, matching the old single-file page.
    """
    manifest = load_manifest(output_dir)
    if manifest is None:
        return frozenset({ENTRY_NAME})
    assets = [a for a in manifest["assets"] if isinstance(a, str)]  # type: ignore[union-attr]
    return frozenset({ENTRY_NAME, *assets})
