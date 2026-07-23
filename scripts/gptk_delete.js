/* ===========================================================================
 * gptk_delete.js  --  cloud-side deletion for Photo Dedup
 * ===========================================================================
 * Trashes the same photos you deleted locally, in Google Photos, by driving
 * the Google-Photos-Toolkit (GPTK) private API from the browser console.
 *
 * WHY THIS EXISTS
 *   Google's official Library API (since 2025-03) can no longer delete
 *   manually-uploaded photos. GPTK talks to the same private endpoints the
 *   web UI uses, which still can. Deleted items go to the Google Photos trash
 *   (recoverable for 60 days), so this is reversible.
 *
 * PREREQUISITES
 *   1. Install Tampermonkey and the "Google Photos Toolkit" userscript
 *      (github.com/xob0t/Google-Photos-Toolkit). Confirm `window.gptkApi`
 *      exists after loading photos.google.com.
 *   2. Open https://photos.google.com and wait until it is fully loaded.
 *   3. Open DevTools console (F12).
 *
 * HOW TO RUN
 *   1. Paste the ENTIRE contents of your output/delete_cloud.json between the
 *      brackets of DELETE_CLOUD below (it is a JSON array).
 *   2. Leave DRY_RUN = true for the first pass -> it only reports matches.
 *   3. Paste this whole file into the console and press Enter.
 *   4. Review the match report. If it looks right, set DRY_RUN = false and
 *      run again to actually move matches to the trash.
 * ===========================================================================
 */

(async () => {
  // ----- CONFIG -------------------------------------------------------------
  const DRY_RUN = true;          // true = report only; false = actually trash
  const BATCH_SIZE = 100;        // items trashed per API call
  const BATCH_DELAY_MS = 2000;   // pause between batches (rate-limit friendly)
  const TIME_TOLERANCE_MS = 120 * 1000; // +/- 2 min when matching by timestamp

  // >>> PASTE output/delete_cloud.json CONTENTS BETWEEN THESE BRACKETS <<<
  const DELETE_CLOUD = [
    // { "filename": "IMG_20260204_194535.jpg",
    //   "exif_datetime": "2026-02-04T19:45:35", "size_bytes": 3200000 },
  ];
  // --------------------------------------------------------------------------

  const log = (...a) => console.log("%c[gptk-delete]", "color:#4c9", ...a);
  const warn = (...a) => console.warn("[gptk-delete]", ...a);

  if (typeof window.gptkApi === "undefined") {
    warn("window.gptkApi not found. Install the GPTK userscript and reload photos.google.com.");
    return;
  }
  if (!Array.isArray(DELETE_CLOUD) || DELETE_CLOUD.length === 0) {
    warn("DELETE_CLOUD is empty. Paste your delete_cloud.json array first.");
    return;
  }
  log(`Loaded ${DELETE_CLOUD.length} target items. DRY_RUN=${DRY_RUN}`);

  // ----- helpers ------------------------------------------------------------
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const normName = (n) => String(n || "").trim().toLowerCase();
  const parseMs = (s) => {
    if (!s) return null;
    const t = Date.parse(String(s).replace(" ", "T"));
    return Number.isNaN(t) ? null : t;
  };

  // Normalize a GPTK item across known field-name variants.
  const normItem = (it) => ({
    dedupKey: it.dedupKey || it.dedup_key || it.mediaKey || it.media_key,
    fileName: it.fileName || it.filename || it.name,
    ts: it.timestamp || it.creationTimestamp || it.takenTimestamp || it.descriptionTimestamp,
  });

  // ----- 1. page through the whole cloud library ----------------------------
  log("Fetching cloud library (paged by uploaded date)...");
  const cloud = [];
  let cursor = undefined;
  let page = 0;
  // GPTK's getItemsByUploadedDate returns a page + a next cursor. Field names
  // vary by version, so probe several. TODO: confirm against your GPTK build.
  while (true) {
    let resp;
    try {
      resp = await window.gptkApi.getItemsByUploadedDate(cursor);
    } catch (e) {
      warn("getItemsByUploadedDate failed:", e);
      break;
    }
    const items = resp?.items || resp?.mediaItems || resp || [];
    if (!items.length) break;
    for (const it of items) cloud.push(normItem(it));
    page += 1;
    const next = resp?.nextPageId ?? resp?.nextPageTimestamp ?? resp?.cursor ?? resp?.nextCursor;
    log(`  page ${page}: +${items.length} (total ${cloud.length})`);
    if (!next || next === cursor) break;
    cursor = next;
    await sleep(300);
  }
  log(`Cloud library size: ${cloud.length}`);

  // ----- 2. build an index by filename --------------------------------------
  const byName = new Map();
  for (const it of cloud) {
    const k = normName(it.fileName);
    if (!k) continue;
    if (!byName.has(k)) byName.set(k, []);
    byName.get(k).push(it);
  }

  // ----- 3. match targets ---------------------------------------------------
  const toTrash = [];
  let matched = 0, missed = 0, ambiguous = 0;
  for (const target of DELETE_CLOUD) {
    const cands = byName.get(normName(target.filename)) || [];
    if (cands.length === 0) { missed++; continue; }
    const tMs = parseMs(target.exif_datetime);
    let pick = null;
    if (cands.length === 1) {
      pick = cands[0];
    } else if (tMs != null) {
      // choose the candidate whose timestamp is closest and within tolerance
      let best = null, bestDiff = Infinity;
      for (const c of cands) {
        const cMs = Number(c.ts) || null;
        if (cMs == null) continue;
        const diff = Math.abs(cMs - tMs);
        if (diff < bestDiff) { bestDiff = diff; best = c; }
      }
      if (best && bestDiff <= TIME_TOLERANCE_MS) pick = best;
    }
    if (pick && pick.dedupKey) { toTrash.push(pick.dedupKey); matched++; }
    else { ambiguous++; }
  }

  log(`Match report: matched=${matched} missed=${missed} ambiguous/unmatched=${ambiguous}`);
  if (matched === 0) { warn("Nothing matched -- aborting."); return; }

  if (DRY_RUN) {
    log(`DRY RUN: would trash ${toTrash.length} items. Set DRY_RUN=false to apply.`);
    console.table(toTrash.slice(0, 20).map((k) => ({ dedupKey: k })));
    return;
  }

  // ----- 4. trash in batches ------------------------------------------------
  log(`Trashing ${toTrash.length} items in batches of ${BATCH_SIZE}...`);
  let done = 0, failed = 0;
  for (let i = 0; i < toTrash.length; i += BATCH_SIZE) {
    const batch = toTrash.slice(i, i + BATCH_SIZE);
    try {
      await window.gptkApi.moveItemsToTrash(batch);
      done += batch.length;
      log(`  trashed ${done}/${toTrash.length}`);
    } catch (e) {
      failed += batch.length;
      warn(`  batch ${i / BATCH_SIZE} failed:`, e);
    }
    await sleep(BATCH_DELAY_MS);
  }
  log(`Done. trashed=${done} failed=${failed}. Check Google Photos > Trash (60-day recovery).`);
})();
