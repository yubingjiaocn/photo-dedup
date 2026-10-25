"""Build the distributable local-trial ZIP from an explicit allowlist.

Usage::

    python scripts/package_local_trial.py --dest <dir outside the repo>

Contents: the ``src`` modules reachable (statically, including function-level
imports) from the local-trial entry points, the schema files those modules name,
the committed ``frontend/dist`` build, requirements/setup/config/launchers and
``LOCAL_TRIAL.md``. Nothing else: no photos, inventories, outputs, artifacts,
fixtures, testdata, research, models, labels, secrets, ``.git``, venvs or
``__pycache__``. Entries are sorted with fixed timestamps, so the same tree
gives byte-identical ZIPs. ``SHA256SUMS.txt`` is written inside and beside it.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Set

REPO = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "photo-dedup-local-trial-v2-vlm"
ENTRY_MODULES = ("local_trial", "local_trial_export", "run_pipeline", "serve_review",
                 "review_server", "stage1_setup")
TOP_FILES = ("requirements.txt", "setup_windows.ps1", "setup_windows.bat",
             "start-local-review.ps1", "start-local-review.bat", "config.yaml",
             "LOCAL_TRIAL.md")
FORBIDDEN_PARTS = {".git", ".venv", "venv", "__pycache__", "artifacts", "output", "fixtures",
                   "testdata", "research", "models", "node_modules", "tests"}
FORBIDDEN_SUFFIXES = {".sqlite", ".jpg", ".jpeg", ".png", ".heic", ".mp4", ".onnx", ".pt",
                      ".safetensors", ".env", ".pem", ".key"}
_SCHEMA_RE = re.compile(r"schemas/([\w.-]+\.json)")
_FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def _imports(path: Path) -> Set[str]:
    """Sibling ``src`` modules a file imports anywhere (module or function level)."""
    names: Set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1:
                if node.module:
                    names.add(node.module.split(".")[0])
                else:
                    names.update(alias.name for alias in node.names)
            elif node.level == 0 and node.module and node.module.split(".")[0] == "src":
                parts = node.module.split(".")
                if len(parts) > 1:
                    names.add(parts[1])
                else:
                    names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "src" and len(parts) > 1:
                    names.add(parts[1])
    return {n for n in names if (REPO / "src" / f"{n}.py").is_file()}


def module_closure() -> List[str]:
    seen: Set[str] = set()
    todo = list(ENTRY_MODULES)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        todo.extend(_imports(REPO / "src" / f"{name}.py") - seen)
    return sorted(seen)


def file_list() -> List[str]:
    modules = module_closure()
    files = {"src/__init__.py", *(f"src/{m}.py" for m in modules)}
    for m in modules:
        text = (REPO / "src" / f"{m}.py").read_text(encoding="utf-8")
        files.update(f"schemas/{name}" for name in _SCHEMA_RE.findall(text)
                     if (REPO / "schemas" / name).is_file())
        if '"schemas").glob' in text:
            files.update(f"schemas/{p.name}" for p in (REPO / "schemas").glob("*.json"))
        if 'with_name("human_audit_ui.html")' in text:
            files.add("src/human_audit_ui.html")
    files.update(p.relative_to(REPO).as_posix()
                 for p in (REPO / "frontend" / "dist").rglob("*") if p.is_file())
    files.update(TOP_FILES)
    out = sorted(files)
    for rel in out:
        parts = set(Path(rel).parts)
        if parts & FORBIDDEN_PARTS or Path(rel).suffix.lower() in FORBIDDEN_SUFFIXES:
            raise SystemExit(f"refusing to package forbidden path: {rel}")
        if not (REPO / rel).is_file():
            raise SystemExit(f"allowlisted file missing: {rel}")
    return out


def build(dest: Path) -> Dict[str, str]:
    dest = dest.resolve()
    if dest == REPO or REPO in dest.parents:
        raise SystemExit("--dest must be outside the repository")
    dest.mkdir(parents=True, exist_ok=True)
    files = file_list()
    sums = {rel: hashlib.sha256((REPO / rel).read_bytes()).hexdigest() for rel in files}
    sums_text = "".join(f"{sums[rel]}  {rel}\n" for rel in files)
    zip_path = dest / f"{PACKAGE_NAME}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel, data in [*((r, (REPO / r).read_bytes()) for r in files),
                          ("SHA256SUMS.txt", sums_text.encode("utf-8"))]:
            info = zipfile.ZipInfo(f"{PACKAGE_NAME}/{rel}", date_time=_FIXED_TIME)
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
    (dest / "SHA256SUMS.txt").write_text(sums_text, encoding="utf-8")
    zip_sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    (dest / f"{PACKAGE_NAME}.zip.sha256").write_text(f"{zip_sha}  {zip_path.name}\n",
                                                     encoding="utf-8")
    return {"zip": str(zip_path), "sha256": zip_sha, "files": str(len(files))}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--list", action="store_true", help="print the allowlist and exit")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(file_list()))
        return 0
    result = build(args.dest)
    print(f"[package] {result['files']} files -> {result['zip']}")
    print(f"[package] sha256 {result['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
