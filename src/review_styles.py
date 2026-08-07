"""Stylesheet for the review workbench.

Split out of :mod:`src.review_template` so the markup, the CSS and the browser
script each stay readable on their own. No build step is involved: the string is
inlined into ``review.html`` exactly as it is written here.

Colour language (used consistently by every state in the UI):

===============  =========================  ===================================
``--focus``      yellow                     what the keyboard is pointing at
``--done``       green                      finished (accepted or picked)
``--ai``         blue/violet                the machine's recommendation
``--later``      orange                     deferred ("稍后")
``--danger``     red                        destructive only (建议删除)
``--neutral``    grey-blue                  everything undecided
===============  =========================  ===================================
"""

from __future__ import annotations

STYLES = r"""
:root{
 --focus:#fc3;--done:#3c6;--ai:#8ab4ff;--later:#f93;--danger:#d15;--neutral:#7d8ba0;
 --bg:#111;--panel:#1a1a1a;--line:#3a3a3a;--ink:#eee;--dim:#9aa4b2;
 --tile:clamp(220px,24vw,420px);--tileh:clamp(180px,19vw,340px);
 --fsw:clamp(74px,7.4vw,132px);--fsh:clamp(56px,5.6vw,100px)}
body.dense{--tile:clamp(120px,12vw,200px);--tileh:clamp(96px,9vw,150px);
 --fsw:clamp(56px,5vw,92px);--fsh:clamp(42px,3.8vw,70px)}
*{box-sizing:border-box}
body{font-family:system-ui,"Microsoft YaHei",Arial,sans-serif;margin:0;padding:14px 16px 16px;
 background:var(--bg);color:var(--ink)}
h1{font-size:17px;margin:0 0 10px;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
h1 .sub{font-size:12px;color:var(--dim);font-weight:normal}
.stats,.notice{color:var(--dim);margin:8px 0;font-size:12.5px;line-height:1.6}
.notice{padding:9px 11px;background:#18232b;border-radius:6px}
/* Diagnostics stay in the document (the pipeline rewrites #perf) but out of sight. */
.diag{display:none!important}
/* The one message the reviewer must not miss: their saved decisions could not be
   read. Prominent, dismissable by fixing the cause, and never blocking review. */
.statewarn{margin:8px 0;padding:10px 13px;border-radius:6px;font-size:13px;
 line-height:1.65;background:#3a2410;border:1px solid var(--later);
 border-left:5px solid var(--later);color:#ffe2c9}
.statewarn[hidden]{display:none}
.statewarn b{color:#fff}
.statewarn code{background:#22303c;border-radius:3px;padding:1px 5px;
 font-family:ui-monospace,monospace;color:#ffd9b3;word-break:break-all}
nav{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:8px 0}
nav .sep{color:#555}
nav .grouplabel{font-size:12px;color:var(--dim)}
button,select{padding:6px 11px;background:#222;color:var(--ink);border:1px solid #555;
 border-radius:5px;cursor:pointer;font-size:13px}
button:hover:not(:disabled){border-color:#888}
button.on{background:var(--focus);color:#000;font-weight:bold;border-color:var(--focus)}
button:disabled{opacity:.4;cursor:not-allowed}
.tabs button{padding:7px 14px;font-size:13.5px}
.tabs button .n{font-weight:bold;margin-left:5px}
.tabs button[data-queue=DONE].on{background:var(--done);border-color:var(--done)}
.tabs button[data-queue=LATER].on{background:var(--later);border-color:var(--later)}

/* --- workbench ---------------------------------------------------------- */
/* Sized to the viewport and laid out with flex, so the action bar and the
   shortcut hint are always on screen: the stage gives up height, never the
   controls. */
#workbench{display:none;flex-direction:column;gap:8px;margin-top:6px;
 min-height:0;height:calc(100vh - 128px)}
#workbench.show{display:flex}
#workbench.overlay{position:fixed;inset:0;z-index:60;background:rgba(0,0,0,.95);
 padding:12px;margin:0;height:100vh}
.wbhead{display:flex;gap:10px;align-items:center;flex-wrap:wrap;flex:0 0 auto;
 border:1px solid var(--line);border-left:4px solid var(--focus);
 background:var(--panel);border-radius:6px;padding:8px 11px}
.wbhead .pos{font-size:15px;font-weight:bold;color:var(--focus)}
.wbhead .meta{font-size:12.5px;color:var(--dim);flex:1;min-width:160px}
.wbhead .gtype{color:var(--ink)}
.wbhead .enum{font-size:10.5px;color:#6f7a88}
.wbhead.done{border-left-color:var(--done)}
.wbhead.later{border-left-color:var(--later)}
.evidence{display:none;border:1px solid var(--line);border-radius:6px;background:#151b22;
 padding:9px 11px;font-size:12.5px;color:var(--dim);flex:0 0 auto;
 max-height:26vh;overflow:auto}
.evidence.open{display:block}
.evidence table{border-collapse:collapse;width:100%}
.evidence td,.evidence th{padding:3px 8px;border-bottom:1px solid #26303a;text-align:left;
 font-weight:normal}
.evidence th{color:#7f8b99;font-size:11.5px}
.evidence .name{color:var(--ink)}
.evidence .enum{font-size:10.5px;color:#6f7a88;margin-left:5px}
.stage{display:grid;grid-template-columns:1fr;gap:8px;flex:1 1 auto;
 min-height:120px}
.stage.compare{grid-template-columns:1fr 1fr}
.pane{position:relative;overflow:hidden;background:#000;border:1px solid var(--line);
 border-radius:6px;display:flex;flex-direction:column;min-width:0;min-height:0}
.pane.hide{display:none}
.pane .label{position:absolute;left:0;right:0;top:0;z-index:2;font-size:12px;padding:5px 8px;
 background:linear-gradient(rgba(0,0,0,.75),rgba(0,0,0,0));color:#dfe6ee;
 pointer-events:none;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pane .frame{flex:1 1 auto;min-height:0;display:flex;align-items:center;justify-content:center;
 cursor:zoom-in;touch-action:none}
.pane .frame.zoomed{cursor:grab}
.pane .frame.grabbing{cursor:grabbing}
.pane img{max-width:100%;max-height:100%;object-fit:contain;transform-origin:center center;
 will-change:transform;user-select:none;-webkit-user-drag:none}
.pane.ai{border-color:var(--ai)}
.zoombar{display:flex;gap:8px;align-items:center;font-size:12px;color:var(--dim);
 flex-wrap:wrap;flex:0 0 auto}
.zoombar .val{color:var(--focus);font-variant-numeric:tabular-nums}
.filmstrip{display:flex;gap:6px;overflow-x:auto;overflow-y:hidden;padding:6px 2px 2px;
 border-top:1px solid #2a2a2a;justify-content:safe center;flex:0 0 auto}
.fs-item{flex:0 0 auto;border:2px solid #444;border-radius:5px;padding:2px;background:#161616;
 cursor:pointer;text-align:center}
.fs-item.current{border-color:var(--focus);box-shadow:0 0 10px rgba(255,204,51,.55)}
.fs-item img{display:block;width:var(--fsw);height:var(--fsh);object-fit:contain;background:#000;
 border-radius:3px}
.fs-miss{width:var(--fsw);height:var(--fsh);background:#221c1c;color:#c88;font-size:10px;
 display:flex;align-items:center;justify-content:center;border-radius:3px}
.fs-meta{font-size:10px;color:#bbb;margin-top:2px;white-space:nowrap}
.fs-idx{color:#9cf}
.actionbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;flex:0 0 auto;
 border:1px solid var(--line);background:var(--panel);border-radius:6px;padding:8px 11px}
.actionbar button{font-size:13.5px;padding:8px 13px}
#bAccept{border-color:var(--ai);color:#dce8ff}
#bPick{border-color:var(--done);color:#d7f5e3}
#bMark{border-color:var(--later);color:#ffe2c9}
.actionbar .hint{flex:1;min-width:220px;text-align:right;font-size:11.5px;color:var(--dim);
 line-height:1.7}
.actionbar .hint kbd{background:#242c36;border:1px solid #3d4854;border-radius:3px;
 padding:0 4px;margin:0 2px;font-family:inherit;color:var(--focus)}
.donepanel{display:none;border:1px solid var(--done);border-left:4px solid var(--done);
 background:#132018;border-radius:6px;padding:16px 18px}
.donepanel.show{display:block}
.donepanel h2{margin:0 0 8px;font-size:17px;color:var(--done)}
.donepanel p{margin:0 0 12px;color:var(--dim);font-size:13px}

/* --- browse list -------------------------------------------------------- */
#listWrap{display:none}
#listWrap.show{display:block}
.row{display:grid;grid-template-columns:repeat(auto-fill,minmax(var(--tile),1fr));gap:12px;
 align-items:start}
.card{position:relative;min-width:0;border:1px solid #555;padding:6px;border-radius:6px;
 background:var(--panel);cursor:zoom-in}
.card:hover{border-color:#888}
.card.focused{border-color:var(--focus);box-shadow:0 0 8px rgba(255,204,51,.4)}
.card img{display:block;width:100%;height:var(--tileh);object-fit:contain;background:#000;
 border-radius:3px}
.miss{width:100%;height:var(--tileh);background:#221c1c;color:#c88;font-size:12px;
 display:flex;align-items:center;justify-content:center;text-align:center;border-radius:3px}
.cap{font-size:12px;color:#bbb;word-break:break-word;margin-top:5px;line-height:1.5}
body.dense .cap{font-size:11px;line-height:1.35}
.tag{font-weight:bold;color:#fc9}
.keep{border-color:var(--ai)}.maybe{border-color:var(--neutral)}
.unknown{border-color:var(--neutral)}.ungrouped{border-color:var(--neutral)}
.auto_remove{border-color:var(--danger)}
.badge{display:inline-block;font-size:10px;padding:1px 5px;margin-left:4px;border-radius:8px;
 background:#333;color:var(--ink);white-space:nowrap;vertical-align:middle}
.badge.ai{background:#24365c;color:#cfe0ff}
.badge.human{background:#1e4232;color:#c8f2d9}
.badge.cur{background:var(--focus);color:#000;font-weight:bold}
.badge.later{background:#4a3419;color:#ffd9b3}

/* --- help + toast ------------------------------------------------------- */
.help{position:fixed;inset:0;z-index:100;background:rgba(0,0,0,.9);display:none;
 align-items:center;justify-content:center;padding:20px}.help.open{display:flex}
.help-box{background:var(--panel);border:1px solid #555;border-radius:8px;padding:20px;
 max-width:640px;max-height:90vh;overflow-y:auto}
.help-box h2{margin-top:0;color:var(--focus)}
.help-box table{width:100%;border-collapse:collapse;margin:10px 0}
.help-box td{padding:7px;border-bottom:1px solid #333;font-size:13px}
.help-box td:first-child{color:var(--focus);font-family:monospace;white-space:nowrap}
.help-box .grouphead{color:var(--focus);font-weight:bold}
#toast{position:fixed;left:50%;bottom:26px;transform:translateX(-50%);z-index:120;
 background:#222c38;border:1px solid #48586a;color:var(--ink);padding:9px 16px;
 border-radius:20px;font-size:13px;opacity:0;pointer-events:none;transition:opacity .18s}
#toast.show{opacity:1}
@media(max-width:820px){:root{--tile:min(100%,300px)}
 #workbench{height:auto}.stage{min-height:44vh}
 .stage.compare{grid-template-columns:1fr}}
"""
