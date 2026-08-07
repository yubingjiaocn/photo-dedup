"""Review page rendering: the paged UI shell plus a cached-thumbnail fallback.

Split out of :mod:`src.stage3_report` to keep both files small; the markup, CSS
and browser script itself live in :mod:`src.review_template`. Two rules govern
everything here:

* **No original photo is opened to build this page.** Static fallback tiles
  reference ``output/thumbs/<id>.jpg`` written by Stage 1, or say plainly that
  the thumbnail is unavailable. In the served page, originals are fetched only
  through ``/api/original/<id>`` while the album viewer is explicitly open.
* **The full views are not embedded.** The template ships an empty grid that the
  local server fills page by page from SQLite, so a 100k-photo library does not
  become a 100k-entry HTML document. The static block is only a small
  ``file://`` fallback.

The UI text is Chinese; internal state names (MAYBE/UNKNOWN/KEEP/AUTO_REMOVE and
group types) stay English to match the database and the manifests.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict

from .review_template import PAGE_TEMPLATE

DEFAULT_REVIEW_LIMIT = 200

# Kept as a private alias so existing readers of this module still find the
# template under the old name.
_PAGE_TEMPLATE = PAGE_TEMPLATE


def cached_thumb_uri(output_dir: Path, file_id: int | None) -> str | None:
    """Relative URL of an existing SSD thumbnail, or None.

    Deliberately never opens an original photo: if Stage 1 did not manage to
    write the thumbnail, the review shows an explicit placeholder instead.
    """
    from . import thumbnails

    if file_id is None:
        return None
    if not thumbnails.thumb_path(thumbnails.thumbs_dir(output_dir), int(file_id)).is_file():
        return None
    return f"{thumbnails.DIR_NAME}/{int(file_id)}{thumbnails.SUFFIX}"


_STATIC_NOTICE = (
    "静态后备页面。请启动本地服务器（默认开启）以分页浏览完整的全部时间线、"
    "MAYBE、UNKNOWN 和 GROUPS 视图。"
)


def _tile(output_dir: Path, rec: Dict[str, Any]) -> str:
    """Static-fallback tile using only an already-cached SSD thumbnail."""
    decision = str(rec.get("decision") or "UNKNOWN")
    uri = cached_thumb_uri(output_dir, rec.get("file_id"))
    if uri:
        image = f'<img loading="lazy" src="{html.escape(uri)}">'
    else:
        image = '<div class="miss">缩略图不可用<br>thumbnail unavailable</div>'
    score = rec.get("quality_score")
    score_txt = f"{score:.1f}" if isinstance(score, (int, float)) else "?"
    return (
        f'<div class="card {html.escape(decision.lower())}">{image}<div class="cap">'
        f'<span class="tag">{html.escape(decision)}</span><br>'
        f'{html.escape(str(rec.get("basename") or ""))}<br>'
        f'{rec.get("width") or "?"}x{rec.get("height") or "?"} &middot; 质量={score_txt} '
        f'&middot; 人脸={rec.get("face_count") if rec.get("face_count") is not None else "?"}<br>'
        f'{html.escape(str(rec.get("reason") or ""))}</div></div>'
    )


def render_html(
    data: Dict[str, Any],
    output_dir: Path,
    review_limit: int = DEFAULT_REVIEW_LIMIT,
    performance_panel: str = "",
) -> str:
    """Render the paged review page plus a small cached-thumbnail fallback.

    ``review_limit`` bounds only the static ``file://`` fallback tiles; 0 omits
    them entirely. The paged views are always complete because they are served
    from the DB index, not embedded here.
    """
    if review_limit < 0:
        raise ValueError("review_limit must be 0 or greater")
    queue = data["queues"]["MAYBE"] or data["queues"]["UNKNOWN"]
    blocks = [_tile(Path(output_dir), rec) for rec in queue[:review_limit]]
    return (
        PAGE_TEMPLATE
        .replace("PERFORMANCE_PANEL", performance_panel or
                 "性能指标由一键流水线写入（参见 performance.txt）。")
        .replace("STATIC_NOTICE", _STATIC_NOTICE)
        .replace(
            "SUMMARY_TEXT",
            f'分组 {len(data["groups"])} 个 &middot; '
            f'建议自动删除（AUTO_REMOVE）{len(data["delete_paths"])} 个文件 &middot; '
            f'MAYBE {len(data["queues"]["MAYBE"])} &middot; '
            f'UNKNOWN {len(data["queues"]["UNKNOWN"])} &middot; '
            f'约可释放 {data["total_delete_bytes"] / (1024 ** 3):.2f} GiB',
        )
        .replace("STATIC_FALLBACK", "".join(blocks))
    )


_PANEL_START = '<div class="notice" id="perf">'
_PANEL_END = "</div>"


def rewrite_performance_panel(review_html: Path, panel: str) -> bool:
    """Replace the performance panel once final stage timings are known."""
    review_html = Path(review_html)
    try:
        text = review_html.read_text(encoding="utf-8")
    except OSError:
        return False
    start = text.find(_PANEL_START)
    if start < 0:
        return False
    body_start = start + len(_PANEL_START)
    end = text.find(_PANEL_END, body_start)
    if end < 0:
        return False
    review_html.write_text(text[:body_start] + panel + text[end:], encoding="utf-8")
    return True
