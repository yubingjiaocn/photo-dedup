"""Small-folder local review trial: one safe entry point for the Windows launcher.

Usage::

    python -m src.local_trial start  --root "E:\\照片\\2026-10-01" --output "D:\\photo-review\\2026-10-01"
    python -m src.local_trial reopen --output "D:\\photo-review\\2026-10-01"
    python -m src.local_trial export --output "D:\\photo-review\\2026-10-01"
    python -m src.local_trial judge  --output "D:\\photo-review\\2026-10-01" --vlm-base-url http://127.0.0.1:4000/v1 --vlm-model <alias>

``start`` runs the existing review-only pipeline (default ``--limit 100``, torch
backend) and then serves the existing local review UI. If that output already
finished a run, it reopens the saved review instead of scanning again (pass
``--rescan`` to re-run). ``reopen`` never scans. ``export`` writes data-only
selection records (see :mod:`src.local_trial_export`).

VLM suggestions are opt-in only: ``start --vlm`` (scan, then judge, then open)
or ``judge`` (judge an existing output, then exit). They call one user-supplied
OpenAI-compatible endpoint for a bounded set of groups (see :mod:`src.local_vlm`);
the key comes only from ``PHOTO_VLM_API_KEY``. ``reopen`` never calls a model.

Guards added on top of the pipeline's own one-root-per-output check: the output
may not sit inside the photo root (or vice versa) nor inside this program's
folder, and the stub backend needs an explicit ``--smoke-stub``. Nothing here
imports ``execute_local``; originals are only read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import local_vlm

DEFAULT_LIMIT = 100
APP_DIR = Path(__file__).resolve().parent.parent


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def check_paths(root: Optional[Path], output: Path) -> Optional[str]:
    """Return a human-readable refusal, or None when the paths are safe."""
    output = output.expanduser().resolve()
    if _inside(output, APP_DIR):
        return f"output must not be inside the program folder {APP_DIR}: {output}"
    if root is not None:
        root = root.expanduser().resolve()
        if not root.is_dir():
            return f"photo root not found: {root}"
        if _inside(output, root):
            return f"output must not be inside the photo root (originals stay untouched): {output}"
        if _inside(root, output):
            return f"photo root must not be inside the output directory: {root}"
    return None


def _finished(output: Path) -> bool:
    return (output / "inventory.sqlite").is_file() and (output / "review_summary.json").is_file()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Local photo review trial (review-only; never moves, deletes or edits photos).")
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start", help="scan a small new folder, then open the review")
    start.add_argument("--root", required=True, help="photo folder to review (read-only)")
    start.add_argument("--output", required=True, help="separate folder for review data")
    start.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                       help=f"process at most N photos (default {DEFAULT_LIMIT})")
    start.add_argument("--smoke-stub", action="store_true",
                       help="use the stub backend (smoke test only, NOT real quality)")
    start.add_argument("--rescan", action="store_true",
                       help="re-run the pipeline even if this output already finished")
    for name, helptext in (("reopen", "open a saved review without scanning"),
                           ("export", "export saved selections as data-only records"),
                           ("judge", "ask the configured VLM about a few groups, then exit")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--output", required=True, help="existing review output folder")
    start.add_argument("--vlm", action="store_true",
                       help="after the scan, ask the configured VLM about a few groups")
    for p in (start, sub.choices["judge"]):
        p.add_argument("--vlm-base-url", help="OpenAI-compatible base URL, e.g. http://127.0.0.1:4000/v1 "
                       "(or PHOTO_VLM_BASE_URL); key only via PHOTO_VLM_API_KEY")
        p.add_argument("--vlm-model", help="model alias as your gateway names it (or PHOTO_VLM_MODEL)")
        p.add_argument("--vlm-max-groups", type=int, default=local_vlm.DEFAULT_GROUPS,
                       help=f"groups to judge (default {local_vlm.DEFAULT_GROUPS}, max {local_vlm.MAX_GROUPS})")
        p.add_argument("--vlm-timeout", type=float, default=local_vlm.DEFAULT_TIMEOUT_S,
                       help="seconds to wait for one reply (no retries)")
        p.add_argument("--vlm-no-response-format", action="store_true",
                       help="do not send response_format=json_object (for gateways that reject it)")
        p.add_argument("--vlm-force", action="store_true",
                       help="ask again even when a cached answer matches")
    for p in (start, sub.choices["reopen"]):
        p.add_argument("--port", type=int, default=0)
        p.add_argument("--no-open", action="store_true", help="do not open the browser")
    args = parser.parse_args(argv)

    output = Path(args.output).expanduser().resolve()
    refusal = check_paths(Path(args.root) if args.command == "start" else None, output)
    if refusal:
        print(f"[local_trial][ERROR] {refusal}", file=sys.stderr)
        return 2

    if args.command == "export":
        from . import local_trial_export
        return local_trial_export.main(["--output", str(output)])

    vlm_config: Optional[Dict[str, Any]] = None
    if args.command == "judge" or (args.command == "start" and args.vlm):
        try:
            if not (1 <= args.vlm_max_groups <= local_vlm.MAX_GROUPS):
                raise local_vlm.ConfigError(
                    f"--vlm-max-groups must be between 1 and {local_vlm.MAX_GROUPS}")
            vlm_config = local_vlm.make_config(
                args.vlm_base_url, args.vlm_model, timeout_s=args.vlm_timeout,
                response_format=not args.vlm_no_response_format)
        except local_vlm.ConfigError as exc:
            print(f"[local_trial][ERROR] {exc}", file=sys.stderr)
            return 2

    if args.command == "judge":
        if not _finished(output):
            print(f"[local_trial][ERROR] no finished review in {output}; run start first",
                  file=sys.stderr)
            return 2
        return _judge(output, vlm_config, args)

    serve_args = ["--output", str(output), "--port", str(args.port)]
    if args.no_open:
        serve_args.append("--no-open")

    if args.command == "start":
        from . import root_scope
        try:
            root_scope.preflight(output / "inventory.sqlite", args.root)
        except root_scope.ParameterError as exc:
            print(f"[local_trial][ERROR] {exc}", file=sys.stderr)
            return 2

    from . import serve_review
    if args.command == "reopen" or (_finished(output) and not args.rescan):
        if args.command == "start":
            print(f"[local_trial] {output} already has a finished review; reopening it "
                  "without scanning (use --rescan to scan again)")
            if vlm_config is not None:
                code = _judge(output, vlm_config, args)
                if code:
                    return code
        else:
            print("[local_trial] reopening saved review; saved VLM suggestions are shown "
                  "as-is, no model is called")
        return serve_review.main(serve_args)

    if args.limit is not None and args.limit <= 0:
        print("[local_trial][ERROR] --limit must be a positive number", file=sys.stderr)
        return 2
    backend = "stub" if args.smoke_stub else "torch"
    if backend == "stub":
        print("[local_trial] WARNING: stub backend = smoke test only; grouping/quality "
              "hints are NOT the real torch model results")
    from . import run_pipeline
    pipeline_args = ["--root", str(Path(args.root).expanduser().resolve()), *serve_args,
                     "--backend", backend, "--limit", str(args.limit)]
    if vlm_config is None:
        return run_pipeline.main(pipeline_args)
    print("[local_trial] step 1/3: scanning photos (no model calls yet)")
    code = run_pipeline.main([*pipeline_args, "--no-serve"])
    if code:
        return code
    print("[local_trial] step 2/3: VLM suggestions")
    code = _judge(output, vlm_config, args)
    if code:
        return code
    print("[local_trial] step 3/3: opening review")
    return serve_review.main(serve_args)


def _judge(output: Path, config: Dict[str, Any], args: argparse.Namespace) -> int:
    """Run the bounded VLM stage. Model failures are recorded, not fatal."""
    try:
        local_vlm.run(output, config, max_groups=args.vlm_max_groups, force=args.vlm_force)
    except (local_vlm.ConfigError, OSError, RuntimeError, ValueError) as exc:
        print(f"[local_trial][ERROR] VLM stage: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
