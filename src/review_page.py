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
* **Diagnostics are generated but not displayed.** The performance panel is
  still written (and later rewritten in place by the pipeline) so
  ``review.html`` remains a complete report artifact, but the reviewer's page
  hides it along with the storage-mode notice. Algorithm diagnostics (quality
  score, face count, grouping reason) are reachable only through the workbench's
  collapsible 推荐依据 panel, for the one group on screen; fallback tiles carry
  none of them.

The UI text is Chinese; internal values (MAYBE/UNKNOWN/KEEP/AUTO_REMOVE and
group types) are translated for display with the original value kept beside them
in small type, so the page and the database stay easy to line up.
"""

from __future__ import annotations

import html
import re
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
    "静态后备页面。请启动本地服务器（默认开启）以使用未审队列工作台，"
    "以及分页浏览完整的全部时间线、待确认和未知视图。"
)

# Display names for the decisions a fallback tile can carry. The internal value
# is still shown, in small type, so a tile can be matched to the database.
_DECISION_LABELS = {
    "KEEP": "保留",
    "AUTO_REMOVE": "建议删除",
    "MAYBE": "待确认",
    "UNKNOWN": "未知",
}


def _tile(output_dir: Path, rec: Dict[str, Any]) -> str:
    """Static-fallback tile using only an already-cached SSD thumbnail.

    Shows what a reviewer needs to recognise the photo (decision, name, pixel
    size, capture time) and no algorithm diagnostics: quality score, face count
    and the internal grouping ``reason`` stay out of the UI.
    """
    decision = str(rec.get("decision") or "UNKNOWN")
    label = _DECISION_LABELS.get(decision.upper(), decision)
    uri = cached_thumb_uri(output_dir, rec.get("file_id"))
    if uri:
        image = f'<img loading="lazy" src="{html.escape(uri)}">'
    else:
        image = '<div class="miss">缩略图不可用<br>thumbnail unavailable</div>'
    return (
        f'<div class="card {html.escape(decision.lower())}">{image}<div class="cap">'
        f'<span class="tag">{html.escape(label)}</span>'
        f'<span class="enum">{html.escape(decision)}</span><br>'
        f'{html.escape(str(rec.get("basename") or ""))}<br>'
        f'{rec.get("width") or "?"}x{rec.get("height") or "?"}<br>'
        f'{html.escape(str(rec.get("exif_datetime") or "无拍摄时间"))}</div></div>'
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
            f'建议自动删除 {len(data["delete_paths"])} 个文件 &middot; '
            f'待确认 {len(data["queues"]["MAYBE"])} &middot; '
            f'未知 {len(data["queues"]["UNKNOWN"])} &middot; '
            f'约可释放 {data["total_delete_bytes"] / (1024 ** 3):.2f} GiB',
        )
        .replace("STATIC_FALLBACK", "".join(blocks))
    )


_PANEL_START = '<div class="notice diag" id="perf" hidden>'
_PANEL_END = "</div>"
# The panel is hidden from the reviewer but still rewritten in place, so match the
# opening tag by its id instead of its exact attribute list. That keeps pages
# produced by earlier versions (plain ``<div class="notice" id="perf">``)
# rewritable too.
_PANEL_OPEN_RE = re.compile(r'<div\b[^>]*\bid="perf"[^>]*>')


def rewrite_performance_panel(review_html: Path, panel: str) -> bool:
    """Replace the performance panel once final stage timings are known."""
    review_html = Path(review_html)
    try:
        text = review_html.read_text(encoding="utf-8")
    except OSError:
        return False
    opening = _PANEL_OPEN_RE.search(text)
    if opening is None:
        return False
    body_start = opening.end()
    end = text.find(_PANEL_END, body_start)
    if end < 0:
        return False
    review_html.write_text(text[:body_start] + panel + text[end:], encoding="utf-8")
    return True
