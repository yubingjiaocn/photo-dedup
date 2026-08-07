"""Real-browser verification for the review workbench (delegating shim).

The review UI is now a compiled Preact application under ``frontend/``. Its
real-browser acceptance test lives beside the source it exercises, at
``frontend/harness/browser_check.py``, and drives the *built* app served by the
real ``src.review_server`` for both viewers (Panzoom in production, and
OpenSeadragon via ``?viewer=osd`` for the comparison gate). This module stays as
the documented entry point and simply forwards to it.

    # start a fixture server, then:
    python3 frontend/harness/fixture_server.py --port 18931 &
    python3 scripts/verification/album_viewer_browser.py http://127.0.0.1:18931/

See ``frontend/README.md`` and ``FRONTEND_DECISION.md`` for what the harness
asserts and why (centred fit, wheel/drag/F/1, dual-pane sync, hold-C blink, the
exact /2->/1->/5 request sequence, three-blink zero-request release, old-group
release, and zero page errors).
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[2] / "frontend" / "harness" / "browser_check.py"


def main() -> None:
    if not HARNESS.is_file():
        raise SystemExit(f"production harness not found: {HARNESS}")
    # A bare base URL is expected (the harness appends review.html itself); accept
    # the historical .../review.html form too by trimming it.
    argv = list(sys.argv)
    if len(argv) > 1 and argv[1].endswith("/review.html"):
        argv[1] = argv[1].rsplit("review.html", 1)[0]
    sys.argv = argv
    runpy.run_path(str(HARNESS), run_name="__main__")


if __name__ == "__main__":
    main()
