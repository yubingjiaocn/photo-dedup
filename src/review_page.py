"""Install the compiled review UI into an output directory.

The review front-end used to be a single ``review.html`` this module built by
string-substituting an inlined stylesheet and script. It is now a Preact
application under ``frontend/``, compiled by Vite; its committed build in
``frontend/dist/`` is the runtime source of truth (see :mod:`src.review_assets`
and ``frontend/README``). Stage 3 calls :func:`install_review_ui` to copy that
build into the output directory and write the ``review_assets.json`` manifest the
server allow-lists against.

Two boundaries survive the move to a build step, unchanged:

* **No original photo is opened to produce this page.** The UI fetches cached
  thumbnails through ``/api/thumb/<id>.jpg`` and an original only through
  ``/api/original/<id>`` while its photo is on screen. Installing the UI copies
  static files and touches no image.
* **The browser only ever sees ``file_id``/``basename``.** No source path is
  rendered, and the only endpoints used are the read-only API plus the single
  ``/api/action`` mutation.

Diagnostics are no longer embedded in the page at all: stage timings and the
thumbnail-cache report live in ``performance.txt`` and ``review_summary.json``,
which is where those numbers are read. The reviewer's page carries none of them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from . import review_assets


def install_review_ui(output_dir: Path) -> Dict[str, Any]:
    """Copy the committed front-end build into ``output_dir``.

    Writes ``review.html`` (unchanged URL), the hashed ``assets/*`` chunks, and
    the ``review_assets.json`` manifest naming exactly what may be served
    statically. Returns that manifest. Raises :class:`FileNotFoundError` if the
    committed build is missing -- that is a broken checkout, not a runtime state
    to paper over.
    """
    return review_assets.install_into(Path(output_dir))
