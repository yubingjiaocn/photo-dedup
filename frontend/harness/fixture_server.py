"""Serve the committed production build + the real review API from one origin.

This is the production harness server: it builds the same synthetic review
output the pytest suite and the manual album viewer use
(``scripts/verification/album_viewer_fixture.build`` -- 4 groups of 3, a group
with no AI keeper, one missing thumbnail), installs the committed Preact build
into it exactly as Stage 3 does, and serves it through the *real*
``src.review_server`` handler. Nothing here stubs the server: the browser test
drives the production HTTP surface, static allow-list included.

    python3 frontend/harness/fixture_server.py [--port N] [--keep]

Prints the base URL on stdout, then serves until interrupted.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.verification.album_viewer_fixture import build  # noqa: E402
from src import review_server  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18931)
    parser.add_argument("--keep", action="store_true", help="print and keep the temp dir")
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="photo-dedup-frontend-harness-"))
    output = work / "output"
    try:
        # build() installs the committed dist via review_page.install_review_ui,
        # so the server serves exactly what Stage 3 would ship.
        build(output, work / "photos")
        server, url = review_server.start_server(output, port=args.port)
        print(url.rsplit("/review.html", 1)[0] + "/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            data = getattr(server, "review_data", None)
            if data is not None:
                data.close()
    finally:
        if args.keep:
            print(work)
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
